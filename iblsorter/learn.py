import logging
import os
from math import sqrt, ceil

import numpy as np
import cupy as cp
from tqdm.auto import tqdm

from iblsorter.cptools import svdecon, svdecon_cpu, median, free_gpu_memory, ones
from iblsorter.cluster import isolated_peaks_new, get_SpikeSample, getClosestChannels
from iblsorter.utils import Bunch, get_cuda, _extend, LargeArrayWriter, plot_diagnostics

logger = logging.getLogger(__name__)


def extractTemplatesfromSnippets(
    data_loader=None, probe=None, params=None, Nbatch=None, nPCs=None
):
    # this function is very similar to extractPCfromSnippets.
    # outputs not just the PC waveforms, but also the template "prototype",
    # basically k-means clustering of 1D waveforms.

    NT = params.NT
    # skip every this many batches
    nskip = params.nskip
    nPCs = nPCs or params.nPCs
    nt0min = params.nt0min

    k = 0
    # preallocate matrix to hold 1D spike snippets
    # dd = cp.zeros((params.nt0, int(5e4)), dtype=np.float32, order='F')
    dds = []

    for ibatch in tqdm(range(0, Nbatch, nskip), desc="Extracting templates"):
        # load data to GPU
        dataRAW = data_loader.load_batch(ibatch)

        # find isolated spikes from each batch
        row, col, mu = isolated_peaks_new(dataRAW, params)

        # for each peak, get the voltage snippet from that channel
        c = get_SpikeSample(dataRAW, row, col, params)

        # if k + c.shape[1] > dd.shape[1]:
        #     dd = cp.pad(dd, (0, dd.shape[1]), mode='constant')

        # dd[:, k:k + c.shape[1]] = c
        dds.append(c)
        k = k + c.shape[1]
        if k > 1e5:
            break

    # discard empty samples
    # dd = dd[:, :k]
    dd = cp.asfortranarray(cp.concatenate(dds, axis=1).astype(np.float32))

    # initialize the template clustering with random waveforms
    uu = np.random.permutation(dd.shape[1])[:nPCs]
    wTEMP = dd[:, uu]
    wTEMP = wTEMP / cp.sum(wTEMP ** 2, axis=0) ** 0.5  # normalize them

    for i in range(10):
        # at each iteration, assign the waveform to its most correlated cluster
        cc = cp.dot(wTEMP.T, dd)
        imax = cp.argmax(cc, axis=0)
        amax = cc[imax, np.arange(cc.shape[1])]
        for j in range(nPCs):
            # weighted average to get new cluster means
            wTEMP[:, j] = cp.dot(dd[:, imax == j], amax[imax == j].T)
        wTEMP = wTEMP / cp.sum(wTEMP ** 2, axis=0) ** 0.5  # unit normalize

    # the PCs are just the left singular vectors of the waveforms
    U, Sv, V = svdecon(dd)

    # take as many as needed
    wPCA = U[:, :nPCs]

    # adjust the arbitrary sign of the first PC so its negativity is downward
    wPCA[:, 0] = -wPCA[:, 0] * cp.sign(wPCA[nt0min, 0])

    return wTEMP, wPCA


# TODO: design - if we just need a couple of params then we can just pass those in.
def getKernels(params):
    # this function makes upsampling kernels for the temporal components.
    # those are used for interpolating the biggest negative peak,
    # and aligning the template to that peak with sub-sample resolution
    # needs nup, the interpolation factor (default = 10)
    # also needs sig, the interpolation smoothness (default = 1)

    nup = params.nup
    sig = params.sig

    nt0min = params.nt0min
    nt0 = params.nt0

    xs = cp.arange(1, nt0 + 1)
    ys = cp.linspace(0.5, nt0 + 0.5, nt0 * nup + 1)[:-1]

    # these kernels are just standard kriging interpolators

    # first compute distances between the sample coordinates
    # for some reason, this seems to be circular, although the waveforms are not circular
    # I think the reason had to do with some constant offsets in some channels?
    d = cp.mod(xs[:, np.newaxis] - xs[np.newaxis, :] + nt0, nt0)
    d = cp.minimum(d, nt0 - d)
    # the kernel covariance uses a squared exponential of spatial scale sig
    Kxx = cp.exp(-(d ** 2) / sig ** 2)

    # do the same for the kernel similarities between upsampled "test" timepoints and
    # the original coordinates
    d = cp.mod(ys[:, np.newaxis] - xs[np.newaxis, :] + nt0, nt0)
    d = cp.minimum(d, nt0 - d)
    Kyx = cp.exp(-(d ** 2) / sig ** 2)

    # the upsampling matrix is given by the following formula,
    # with some light diagonal regularization of the matrix inversion
    B = cp.dot(Kyx, cp.linalg.inv(Kxx + 0.01 * cp.eye(nt0)))
    B = B.reshape((nup, nt0, nt0), order="F")

    # A is just a slice through this upsampling matrix corresponding to the most negative point
    # this is used to compute the biggest negative deflection (after upsampling)
    A = cp.squeeze(B[:, nt0min - 1, :])
    B = cp.transpose(B, [1, 2, 0])

    return A.astype(np.float64), B.astype(np.float64)


def getMeUtU(iU, iC, mask, Nnearest, Nchan):
    # function [UtU, maskU, iList] = getMeUtU(iU, iC, mask, Nnearest, Nchan)
    # this function determines if two templates share any channels
    # iU are the channels that each template is assigned to, one main channel per template
    # iC has as column K the list of neigboring channels for channel K
    # mask are the weights assigned for the corresponding neighboring channels
    # in iC (gaussian-decaying)

    Nfilt = iU.size

    # create a sparse matrix with ones if a channel K belongs to a template
    U = np.zeros((Nchan, Nfilt), dtype=np.float32, order="F")

    # use the template primary channel to obtain its neighboring channels from iC
    ix = cp.asnumpy(iC[:, iU]).squeeze() + np.arange(0, Nchan * Nfilt, Nchan).astype(np.int32)

    # WARNING: transpose because the indexing assumes F ordering.
    U.T.flat[ix] = 1  # use this as an awkward index into U

    # if this is 0, the templates had not pair of channels in common
    UtU = (cp.dot(U.T, U) > 0).astype(np.int32)

    # we also return the masks for each template, picked from the corresponding mask of
    # their primary channel
    maskU = mask[:, iU]

    # # sort template pairs in order of how many channels they share
    # isort = cp.argsort(UtU, axis=0)[::-1]
    # iList = isort[:Nnearest, :]  # take the Nnearest templates for each template

    return UtU, maskU  # , iList


def getMeWtW2(W, U0, Nnearest=None):
    # this function compute the correlation between any two pairs of templates

    # it relies on the fact that the W and U0 are unit normalized, so that the product of a
    # template with itself is 1, as it should be if we're trying to calculate correlations

    # takes input the temporal and spatial factors of the low-rank template, as
    # well as the number of most similar template pairs desired to be output in
    # iList

    nt0, Nfilt, Nrank = W.shape
    WtW = cp.zeros((Nfilt, Nfilt), dtype=np.float32, order="F")

    # since the templates are factorized into orthonormal components, we can compute dot products
    # one dimension at a time
    for i in range(Nrank):
        for j in range(Nrank):
            #  this computes the spatial dot product
            utu0 = cp.dot(U0[:, :, i].T, U0[:, :, j])
            #  this computes the temporal dot product
            wtw0 = cp.dot(W[:, :, i].T, W[:, :, j])

            # the element-wise product of these is added to the matrix of correlatioons
            WtW = WtW + wtw0 * utu0

    # also return a list of most correlated template pairs
    isort = cp.argsort(WtW, axis=0)[::-1]

    if Nnearest:
        # if we don't have enough templates yet, just wrap the indices around the range 1:Nfilt
        iNear = cp.mod(cp.arange(Nnearest), Nfilt)
        iList = isort[iNear, :]  # return the list of pairs for each template
        return WtW, iList
    else:
        return WtW


def custom_lexsort(arrays):
    """
    Lexsort of 1D Cupy arrays, last array given is used for the primary sort order, second-last
    for secondary sort order and so on
    :param arrays: List of 1D Cupy arrays, all lengths must match
    :return: Cupy array of indices that sort the arrays according to the above
    """
    # Check arrays are one-dimensional and have equal length
    for array in arrays:
        assert array.ndim == 1
    n = arrays[0].shape[0]
    for array in arrays[1:]:
        assert array.shape[0] == n

    # Concatenate arrays and pass to cupy lexsort function
    lex_array = cp.concatenate([array.reshape(1, -1) for array in arrays], axis=0)
    return cp.lexsort(lex_array)


def mexGetSpikes2(Params, drez, wTEMP, iC):
    code, constants = get_cuda("mexGetSpikes2")

    NT = int(Params[0])
    Nchan = int(Params[9])
    nt0 = int(Params[4])
    # Nnearest = int(Params[5])
    Nrank = int(Params[14])

    maxFR = constants.maxFR
    Nthreads = constants.Nthreads

    # tpB = (8, 2 * nt0 - 1)
    # tpF = (16, Nnearest)
    tpS = (nt0, 16)

    d_Params = cp.asarray(Params, dtype=np.float64, order="F")
    d_data = cp.asarray(drez, dtype=np.float32, order="F")
    d_W = cp.asarray(wTEMP, dtype=np.float32, order="F")
    d_iC = cp.asarray(iC, dtype=np.int32, order="F")

    d_counter = cp.zeros(2, dtype=np.int32, order="F")
    d_dout = cp.zeros((NT, Nchan), dtype=np.float32, order="F")
    d_dfilt = cp.zeros((Nrank, NT, Nchan), dtype=np.float32, order="F")
    d_err = cp.zeros(NT, dtype=np.float32, order="F")
    d_kkmax = cp.zeros((NT, Nchan), dtype=np.int32, order="F")
    d_kk = cp.zeros(NT, dtype=np.int32, order="F")
    d_ftype = cp.zeros(NT, dtype=np.int32, order="F")
    d_st = cp.zeros(maxFR, dtype=np.int32, order="F")
    d_id = cp.zeros(maxFR, dtype=np.int32, order="F")
    d_x = cp.zeros(maxFR, dtype=np.float32, order="F")
    d_st1 = cp.zeros(maxFR, dtype=np.int32, order="F")
    d_id1 = cp.zeros(maxFR, dtype=np.int32, order="F")

    counter = np.zeros(2, dtype=np.int32, order="F")

    # filter the data with the temporal templates
    Conv1D = cp.RawKernel(code, "Conv1D")
    Conv1D((Nchan,), (Nthreads,), (d_Params, d_data, d_W, d_dfilt))

    # sum each template across channels, square, take max
    sumChannels = cp.RawKernel(code, "sumChannels")
    sumChannels(
        (int(NT / Nthreads),), (Nthreads,), (d_Params, d_dfilt, d_dout, d_kkmax, d_iC)
    )

    # compute the best filter
    bestFilter = cp.RawKernel(code, "bestFilter")
    bestFilter(
        (int(NT / Nthreads),),
        (Nthreads,),
        (d_Params, d_dout, d_err, d_ftype, d_kkmax, d_kk),
    )

    # ignore peaks that are smaller than another nearby peak
    cleanup_spikes = cp.RawKernel(code, "cleanup_spikes")
    cleanup_spikes(
        (int(NT / Nthreads),),
        (Nthreads,),
        (d_Params, d_err, d_ftype, d_x, d_st, d_id, d_counter),
    )

    # ignore peaks that are smaller than another nearby peak
    cleanup_heights = cp.RawKernel(code, "cleanup_heights")
    cleanup_heights(
        (1 + int(maxFR // 32),),
        (32,),
        (d_Params, d_x, d_st, d_id, d_st1, d_id1, d_counter),
    )

    # Order of peaks found can vary depending on thread execution times, sort for determinism
    n_spikes = int(min(maxFR, d_counter[1]))
    sorted_ids = custom_lexsort([d_id1[:n_spikes], d_st1[:n_spikes]])
    d_st1[:n_spikes] = d_st1[:n_spikes][sorted_ids]
    d_id1[:n_spikes] = d_id1[:n_spikes][sorted_ids]

    # add new spikes to 2nd counter
    counter[0] = d_counter[1]
    counter[0] = min(maxFR, counter[0])

    d_WU = cp.zeros((nt0, Nchan, counter[0]), dtype=np.float32, order="F")
    # d_WU1 = cp.zeros((nt0, Nchan, counter[0]), dtype=np.float32, order='F')

    # update dWU here by adding back to subbed spikes
    extract_snips = cp.RawKernel(code, "extract_snips")
    extract_snips((Nchan,), tpS, (d_Params, d_st1, d_id1, d_counter, d_data, d_WU))

    # QUESTION: why a copy here??
    # if counter[0] > 0:
    #     d_WU1[...] = d_WU[...]

    del (
        d_ftype,
        d_kkmax,
        d_err,
        d_st,
        d_id,
        d_st1,
        d_x,
        d_kk,
        d_id1,
        d_counter,
        d_Params,
        d_dfilt,
    )
    return d_WU, d_dout


def mexSVDsmall2(Params, dWU, W, iC, iW, Ka, Kb):
    code, constants = get_cuda("mexSVDsmall2")

    Nthreads = constants.Nthreads

    Nfilt = int(Params[1])
    nt0 = int(Params[4])
    Nrank = int(Params[6])
    Nchan = int(Params[9])

    d_Params = cp.asarray(Params, dtype=np.float64, order="F")

    d_dWU = cp.asarray(dWU, dtype=np.float64, order="F")
    d_iC = cp.asarray(iC, dtype=np.int32, order="F")
    d_iW = cp.asarray(iW, dtype=np.int32, order="F")

    d_A = cp.asarray(Ka, dtype=np.float64, order="F")
    d_B = cp.asarray(Kb, dtype=np.float64, order="F")

    d_U = cp.zeros((Nchan, Nfilt, Nrank), dtype=np.float64, order="F")
    d_mu = cp.zeros(Nfilt, dtype=np.float64, order="F")

    d_W = cp.asarray(W, dtype=np.float64, order="F")

    d_wtw = cp.zeros((nt0, nt0, Nfilt), dtype=np.float64, order="F")
    d_dWUb = cp.zeros((nt0, Nchan, Nfilt), dtype=np.float64, order="F")

    tpS = (nt0, int(Nthreads // nt0))
    tpK = (Nrank, int(Nthreads // Nrank))

    blankdWU = cp.RawKernel(code, "blankdWU")
    blankdWU((Nfilt,), tpS, (d_Params, d_dWU, d_iC, d_iW, d_dWUb))

    # compute dWU * dWU'
    getwtw = cp.RawKernel(code, "getwtw")
    getwtw((Nfilt,), tpS, (d_Params, d_dWUb, d_wtw))

    # get W by power svd iterations
    getW = cp.RawKernel(code, "getW")
    getW((Nfilt,), (nt0,), (d_Params, d_wtw, d_W))

    # compute U by W' * dWU
    getU = cp.RawKernel(code, "getU")
    getU((Nfilt,), tpK, (d_Params, d_dWUb, d_W, d_U))

    # normalize U, get S, get mu, renormalize W
    reNormalize = cp.RawKernel(code, "reNormalize")
    reNormalize((Nfilt,), (nt0,), (d_Params, d_A, d_B, d_W, d_U, d_mu))

    del d_wtw, d_Params, d_dWUb

    return d_W, d_U, d_mu


def mexMPnu8(Params, dataRAW, U, W, mu, iC, iW, UtU, iList, wPCA, params):
    code, constants = get_cuda("mexMPnu8")
    maxFR = int(constants.maxFR)
    nmaxiter = int(constants.nmaxiter)
    Nthreads = int(constants.Nthreads)

    NT = int(Params[0])  # NT = (unsigned int) Params[0];
    Nfilt = int(Params[1])
    nt0 = int(Params[4])
    Nnearest = int(Params[5])
    Nrank = int(Params[6])
    NchanU = int(Params[10])
    Nchan = int(Params[9])

    d_Params = cp.asarray(Params, dtype=np.float64, order="F")

    d_draw = cp.asarray(dataRAW, dtype=np.float32, order="F")
    d_U = cp.asarray(U, dtype=np.float32, order="F")
    d_W = cp.asarray(W, dtype=np.float32, order="F")
    d_mu = cp.asarray(mu, dtype=np.float32, order="F")
    d_iC = cp.asarray(iC, dtype=np.int32, order="F")
    d_iW = cp.asarray(iW, dtype=np.int32, order="F")
    d_UtU = cp.asarray(UtU, dtype=bool, order="F")
    d_iList = cp.asarray(iList, dtype=np.int32, order="F")
    d_wPCA = cp.asarray(wPCA, dtype=np.float32, order="F")

    d_nsp = cp.zeros(Nfilt, dtype=np.int32, order="F")
    d_dWU = cp.zeros((nt0, Nchan, Nfilt), dtype=np.float64, order="F")

    d_dout = cp.zeros((2 * NT, Nfilt), dtype=np.float32, order="F")
    d_data = cp.zeros((NT, Nfilt, Nrank), dtype=np.float32, order="F")
    d_err = cp.zeros(NT, dtype=np.float32, order="F")
    d_ftype = cp.zeros(NT, dtype=np.int32, order="F")
    d_eloss = cp.zeros(NT, dtype=np.float32, order="F")
    d_st = cp.zeros(maxFR, dtype=np.int32, order="F")
    d_id = cp.zeros(maxFR, dtype=np.int32, order="F")
    d_x = cp.zeros(maxFR, dtype=np.float32, order="F")
    d_y = cp.zeros(maxFR, dtype=np.float32, order="F")
    d_z = cp.zeros(maxFR, dtype=np.float32, order="F")

    d_counter = cp.zeros(2, dtype=np.int32, order="F")
    d_count = cp.zeros(nmaxiter, dtype=np.int32, order="F")
    d_feat = cp.zeros((Nnearest, maxFR), dtype=np.float32, order="F")
    d_featPC = cp.zeros((NchanU, Nrank, maxFR), dtype=np.float32, order="F")

    d_idx = cp.zeros(maxFR, dtype=np.int32, order="F")

    counter = np.zeros(2, dtype=np.int32, order="F")

    tpF = (16, Nnearest)
    tpS = (nt0, 16)
    tpPC = (NchanU, Nrank)

    # filter the data with the spatial templates
    spaceFilter = cp.RawKernel(code, "spaceFilter")
    spaceFilter((Nfilt,), (Nthreads,), (d_Params, d_draw, d_U, d_iC, d_iW, d_data))

    # filter the data with the temporal templates
    timeFilter = cp.RawKernel(code, "timeFilter")
    timeFilter((Nfilt,), (Nthreads,), (d_Params, d_data, d_W, d_dout))

    # compute the best filter
    bestFilter = cp.RawKernel(code, "bestFilter")
    bestFilter(
        (int(NT // Nthreads),),
        (Nthreads,),
        (d_Params, d_dout, d_mu, d_err, d_eloss, d_ftype),
    )

    if params.stable_mode and not params.deterministic_mode:
        d_draw64 = cp.array(d_draw, dtype=np.float64)

    # loop to find and subtract spikes
    for k in range(int(Params[3])):
        # ignore peaks that are smaller than another nearby peak
        cleanup_spikes = cp.RawKernel(code, "cleanup_spikes")
        cleanup_spikes(
            (int(NT // Nthreads),),
            (Nthreads,),
            (
                d_Params,
                d_dout,
                d_mu,
                d_err,
                d_eloss,
                d_ftype,
                d_st,
                d_id,
                d_x,
                d_y,
                d_z,
                d_counter,
            ),
        )

        # add new spikes to 2nd counter
        counter[:] = cp.asnumpy(d_counter[:])
        if counter[0] > maxFR:
            logger.warning("Firing rate limit hit for batch.")
            counter[0] = maxFR
            d_counter[0] = counter[0]

        # extract template features before subtraction
        if Params[12] > 1:
            extractFEAT = cp.RawKernel(code, "extractFEAT")
            extractFEAT(
                (64,),
                tpF,
                (d_Params, d_st, d_id, d_counter, d_dout, d_iList, d_mu, d_feat),
            )

        if params.deterministic_mode:
            if params.stable_mode:
                d_stSort = d_st[counter[1]:counter[0]] # cudaMemcpy( d_stSort, d_st+counter[1], (counter[0] - counter[1])*sizeof(int), cudaMemcpyDeviceToDevice );
                d_idx[:counter[0]-counter[1]] = cp.argsort(d_stSort) # cdp_simple_quicksort<<< 1, 1 >>>(d_stSort, d_idx, 0, counter[0] - counter[1] - 1, 0);
            else:
                raise ValueError("Stablemode required for deterministic calculations.")
                # This is allowed in the MATLAB version runtime but it doesn't really make sense
                # and isn't recommended so let's not anyone fall into the trap without knowing.
                # d_idx = cp.arange(0, counter[0] - counter[1])

            if Nchan < Nthreads:
                subtract_spikes_v2 = cp.RawKernel(code, "subtract_spikes_v2")
                subtract_spikes_v2(
                    (1,),
                    (Nchan,),
                    (d_Params, d_st, d_idx, d_id, d_y, d_counter, d_draw, d_W, d_U),
                )
            else:
                subtract_spikes_v2 = cp.RawKernel(code, "subtract_spikes_v2")
                subtract_spikes_v2(
                    (Nchan / Nthreads,),
                    (Nthreads,),
                    (d_Params, d_st, d_idx, d_id, d_y, d_counter, d_draw, d_W, d_U),
                )

            spaceFilterUpdate = cp.RawKernel(code, "spaceFilterUpdate")
            spaceFilterUpdate(
                (Nfilt,),
                (2 * nt0 - 1,),
                (
                    d_Params,
                    d_draw,
                    d_U,
                    d_UtU,
                    d_iC,
                    d_iW,
                    d_data,
                    d_st,
                    d_id,
                    d_counter,
                ),
            )
        else:
            if params.stable_mode:
                subtract_spikes_v4 = cp.RawKernel(code, "subtract_spikes_v4")
                subtract_spikes_v4(
                    (Nfilt,),
                    tpS,
                    (d_Params, d_st, d_id, d_y, d_counter, d_draw64, d_W, d_U),
                )

                spaceFilterUpdate_v2 = cp.RawKernel(code, "spaceFilterUpdate_v2")
                spaceFilterUpdate_v2(
                    (Nfilt,),
                    (2 * nt0 - 1,),
                    (
                        d_Params,
                        d_draw64,
                        d_U,
                        d_UtU,
                        d_iC,
                        d_iW,
                        d_data,
                        d_st,
                        d_id,
                        d_counter,
                    ),
                )
            else:
                # subtract spikes from raw data here
                subtract_spikes = cp.RawKernel(code, "subtract_spikes")
                subtract_spikes(
                    (Nfilt,),
                    tpS,
                    (d_Params, d_st, d_id, d_y, d_counter, d_draw, d_W, d_U),
                )

                # filter the data with the spatial templates
                spaceFilterUpdate = cp.RawKernel(code, "spaceFilterUpdate")
                spaceFilterUpdate(
                    (Nfilt,),
                    (2 * nt0 - 1,),
                    (
                        d_Params,
                        d_draw,
                        d_U,
                        d_UtU,
                        d_iC,
                        d_iW,
                        d_data,
                        d_st,
                        d_id,
                        d_counter,
                    ),
                )

        # filter the data with the temporal templates
        timeFilterUpdate = cp.RawKernel(code, "timeFilterUpdate")
        timeFilterUpdate(
            (Nfilt,),
            (2 * nt0 - 1,),
            (d_Params, d_data, d_W, d_UtU, d_dout, d_st, d_id, d_counter),
        )

        if counter[0] - counter[1] > 0:
            bestFilterUpdate = cp.RawKernel(code, "bestFilterUpdate")
            bestFilterUpdate(
                (counter[0] - counter[1],),
                (2 * nt0 - 1,),
                (
                    d_Params,
                    d_dout,
                    d_mu,
                    d_err,
                    d_eloss,
                    d_ftype,
                    d_st,
                    d_id,
                    d_counter,
                ),
            )

        d_count[k + 1] = d_counter[0]

        # update 1st counter from 2nd counter
        d_counter[1] = d_counter[0]

    if params.stable_mode and not params.deterministic_mode:
        d_draw = cp.array(d_draw64, dtype=np.float32)

    # compute PC features from residuals + subtractions
    # TODO: design - let's not use numeric indexing into the Params array. It's much more difficult to read.
    if Params[12] > 0:
        computePCfeatures = cp.RawKernel(code, "computePCfeatures")
        computePCfeatures(
            (Nfilt,),
            tpPC,
            (
                d_Params,
                d_counter,
                d_draw,
                d_st,
                d_id,
                d_y,
                d_W,
                d_U,
                d_mu,
                d_iW,
                d_iC,
                d_wPCA,
                d_featPC,
            ),
        )

    if params.stable_mode:
        # d_idx = array of time sorted indices
        d_idx[:counter[0]] = cp.argsort(d_st[:counter[0]]) # cdp_simple_quicksort<<< 1, 1 >>>(d_stSort, d_idx, 0, counter[0] - counter[1] - 1, 0);
    else:
        d_idx = cp.arange(0, counter[0])

    # update dWU here by adding back to subbed spikes.
    average_snips = cp.RawKernel(code, "average_snips")
    average_snips(
        (Nfilt,),
        tpS,
        (
            d_Params,
            d_st,
            d_idx,
            d_id,
            d_x,
            d_y,
            d_counter,
            d_draw,
            d_W,
            d_U,
            d_dWU,
            d_nsp,
            d_mu,
            d_z,
        ),
    )

    if counter[0] < maxFR:
        minSize = counter[0]
    else:
        minSize = maxFR

    del d_counter, d_Params, d_ftype, d_err, d_eloss, d_z, d_dout, d_data

    # Sort to ensure determinism as different thread execution times leads to spikes being found
    # in a random order

    sorted_idx = custom_lexsort([d_y[:minSize], d_id[:minSize], d_st[:minSize]])

    # Indexing a f-contiguous array returns a c-contiguous array so we need to recast back into
    # fortran order
    return (
        d_st[:minSize][sorted_idx],
        d_id[:minSize][sorted_idx],
        d_y[:minSize][sorted_idx],
        cp.asfortranarray(d_feat[..., :minSize][..., sorted_idx]),
        d_dWU,
        d_draw,
        d_nsp,
        cp.asfortranarray(d_featPC[..., :minSize][..., sorted_idx]),
        d_x[:minSize][sorted_idx],
    )


def mexWtW2(W1, W2, UtU):
    """
    Calculates the correlations between two sets of temporal components at different time lags and
    multiplies these by the correlations of their spatial components to get the template
    correlations at all time lags

    :param W1: Temporal components,
                cupy array with shape (n_times, n_templates)
    :param W2: Temporal components,
                cupy array with shape (n_times, n_templates)
    :param UtU: Correlations between spatial components,
                cupy array with shape (n_templates, n_templates)
    :return: Template correlations across all time lags,
                cupy array with shape (n_templates, n_templates, 2*n_times-1), fortran order
    """

    d_W1 = cp.asarray(W1, dtype=np.float32, order="F")
    d_W2 = cp.asarray(W2, dtype=np.float32, order="F")
    d_UtU = cp.asarray(UtU, dtype=np.float32, order="F")

    code, constants = get_cuda("mexWtW2")
    nblock = constants.nblock

    n_times, n_templates = d_W1.shape

    WtW = cp.zeros((n_templates, n_templates, 2 * n_times - 1), dtype=np.float32, order="F")

    # Setting parameters for CUDA kernel
    grid = (1 + int(n_templates // nblock), 1 + int(n_templates // nblock))
    block = (nblock, nblock)

    # CUDA function that calculates the correlations
    crossFilter = cp.RawKernel(code, "crossFilter")
    crossFilter(grid, block, (d_W1, d_W2, d_UtU, WtW, n_templates, n_times))

    assert cp.sum(cp.isnan(WtW)) == 0

    return WtW


def getMeWtW(W, U0, Nnearest=None):
    # this function computes correlation between templates at ALL timelags from each other
    # takes the max over timelags to obtain a similarity score
    # also returns lists of most similar templates to each template
    # takes as input the low-rank factorization of templates (W for time and U0
    # for space)

    # W is timesamples (default = 61 ), by number of templates, by rank (default = 3)
    nt0, Nfilt, Nrank = W.shape

    Params = [1, Nfilt, 0, 0, 0, 0, 0, 0, 0, nt0]

    # initialize correlation matrix for all timelags
    WtW = cp.zeros((Nfilt, Nfilt, 2 * nt0 - 1), dtype=np.float32, order="F")
    for i in range(Nrank):
        for j in range(Nrank):
            # the dot product factorizes into separable products for each spatio-temporal component
            utu0 = cp.dot(U0[:, :, i].T, U0[:, :, j])  # spatial products
            # temporal convolutions get multiplied with the spatial products
            wtw0 = mexWtW2(W[:, :, i], W[:, :, j], utu0)
            # add it to the full correlation array
            WtW = WtW + wtw0

    # the maximum across timelags accounts for sample alignment mismatch
    cc = cp.max(WtW, axis=2)

    if Nnearest:
        isort = cp.argsort(cc, axis=0)[::-1]
        # if we don't have enough templates yet, just wrap the indices around the range 1:Nfilt
        iNear = cp.mod(cp.arange(Nnearest), Nfilt)
        iList = isort[iNear, :]  # return the list of pairs for each template
        return WtW, iList
    else:
        return WtW


def triageTemplates2(params, iW, C2C, W, U, dWU, mu, nsp, ndrop):

    # This function checks if some templates should be dropped
    # either because they are very similar to another template,
    # or because they are not catching any spikes, (low mean firing rate).
    # Takes as inputs almost all the information that determines templates, and
    # outputs the same variables back after removing some clusters.

    # this is the firing rate threshold
    m0 = params.minFR * params.NT / params.fs
    idrop = nsp < m0  # drop any templates with firing rate below this

    # remove those templates everywhere
    W = cp.asfortranarray(W[:, ~idrop, :])
    U = cp.asfortranarray(U[:, ~idrop, :])
    dWU = cp.asfortranarray(dWU[:, :, ~idrop])
    mu = mu[~idrop]
    nsp = nsp[~idrop]
    # keep track of how many templates have been removed this way
    ndrop[0] = 0.9 * ndrop[0] + 0.1 * idrop.sum()

    # compute pairwise correlations between templates
    cc = getMeWtW2(W, U, None)
    cc = cc - cp.diag(cp.diag(cc))  # exclude the diagonal

    sd = sqrt(10)  # this is hard-coded here

    # compute a score for the separation of the means
    r0 = 4 * sd / cp.abs(mu[:, np.newaxis] - mu[np.newaxis, :])
    # determine which template has more spikes (that one survives)
    rdir = (nsp[:, np.newaxis] - nsp[np.newaxis, :]) < 0
    # for each pair of template, score their similarity by their template correlation,
    # and amplitude separation
    ipair = (cc > 0.9) & (r0 > 1) & rdir
    # for each template, find its most similar other template
    amax = cp.max(ipair, axis=1)
    # if this score is 1, then all the criteria have bene met for dropping this template
    idrop = amax > 0

    # remove these templates everywhere like before
    W = cp.asfortranarray(W[:, ~idrop, :])
    U = cp.asfortranarray(U[:, ~idrop, :])
    dWU = cp.asfortranarray(dWU[:, :, ~idrop])
    mu = mu[~idrop]
    nsp = nsp[~idrop]
    # keep track of how many templates have been removed this way
    ndrop[1] = 0.9 * ndrop[1] + 0.1 * idrop.sum()

    return W, U, dWU, mu, nsp, ndrop


def learnAndSolve8b(ctx, sanity_plots=False, plot_widgets=None, plot_pos=None):
    """This is the main optimization. Takes the longest time and uses the GPU heavily."""

    Nbatch = ctx.intermediate.Nbatch
    params = ctx.params
    probe = ctx.probe
    ir = ctx.intermediate
    data_loader = ir.data_loader

    iorig = ir.iorig

    # TODO: move_to_config
    NrankPC = 6  # this one is the rank of the PCs, used to detect spikes with threshold crossings
    Nrank = 3  # this one is the rank of the templates

    if params.seed is not None:
        np.random.seed(params.seed)

    wTEMP, wPCA = extractTemplatesfromSnippets(
        data_loader=data_loader, probe=probe, params=params, Nbatch=Nbatch, nPCs=NrankPC
    )

    # move these to the GPU
    wPCA = cp.asarray(wPCA[:, :Nrank], dtype=np.float32, order="F")
    wTEMP = cp.asarray(wTEMP, dtype=np.float32, order="F")
    wPCAd = cp.asarray(
        wPCA, dtype=np.float64, order="F"
    )  # convert to double for extra precision

    nt0 = params.nt0
    nt0min = params.nt0min
    nBatches = Nbatch
    NT = params.NT
    Nfilt = params.Nfilt
    Nchan = probe.Nchan

    # [CR 2024-04-02]: add support for optional overlap at the beginning of the batch
    overlap_samples = params.overlap_samples

    # two variables for the same thing? number of nearest channels to each primary channel
    # TODO: unclear - let's fix this
    NchanNear = min(probe.Nchan, 32)
    Nnearest = min(probe.Nchan, 32)

    # decay of gaussian spatial mask centered on a channel
    sigmaMask = params.sigmaMask

    # find the closest NchanNear channels, and the masks for those channels
    iC, mask, C2C = getClosestChannels(probe, sigmaMask, NchanNear)

    # batch order schedule is a random permutation of all batches
    ischedule = np.random.permutation(nBatches)
    i1 = np.arange(nBatches)

    irounds = np.concatenate((ischedule, i1))

    niter = irounds.size

    # these two flags are used to keep track of what stage of model fitting we're at
    # flag_final = 0
    flag_resort = 1

    # this is the absolute temporal offset in seconds corresponding to the start of the
    # spike sorted time segment
    t0 = 0  # ceil(params.trange(1) * ops.fs)

    nInnerIter = 60  # this is for SVD for the power iteration

    # schedule of learning rates for the model fitting part
    # starts small and goes high, it corresponds approximately to the number of spikes
    # from the past that were averaged to give rise to the current template
    pmi = np.exp(
        -1.0 / np.linspace(params.momentum[0], params.momentum[1], niter - nBatches)
    )

    Nsum = min(Nchan, 7)  # how many channels to extend out the waveform in mexgetspikes
    # lots of parameters passed into the CUDA scripts
    Params = np.array(
        [
            NT,
            Nfilt,
            params.Th[0],
            nInnerIter,
            nt0,
            Nnearest,
            Nrank,
            params.lam,
            pmi[0],
            Nchan,
            NchanNear,
            params.nt0min,
            1,
            Nsum,
            NrankPC,
            params.Th[0],
        ],
        dtype=np.float64,
    )

    # W0 has to be ordered like this
    W0 = cp.transpose(
        cp.atleast_3d(cp.asarray(wPCA, dtype=np.float64, order="F")), [0, 2, 1]
    )

    # initialize the list of channels each template lives on
    iList = cp.zeros((Nnearest, Nfilt), dtype=np.int32, order="F")

    # initialize average number of spikes per batch for each template
    nsp = cp.zeros((0, 1), dtype=np.float64, order="F")

    # this flag starts 0, is set to 1 later
    Params[12] = 0

    # kernels for subsample alignment
    Ka, Kb = getKernels(params)

    p1 = 0.95  # decay of nsp estimate in each batch

    ntot = 0
    # this keeps track of dropped templates for debugging purposes
    ndrop = np.zeros(2, dtype=np.float32, order="F")

    # this is the minimum firing rate that all templates must maintain, or be dropped
    m0 = params.minFR * params.NT / params.fs

    # allocate variables when switching to extraction phase
    # this holds spike times, clusters and other info per spike
    st3 = []  # cp.zeros((int(1e7), 5), dtype=np.float32, order='F')

    # these ones store features per spike
    # Nnearest is the number of nearest templates to store features for
    fW = LargeArrayWriter(
        ctx.path("fW", ext=".dat"), dtype=np.float32, shape=(Nnearest, -1)
    )
    # NchanNear is the number of nearest channels to take PC features from
    fWpc = LargeArrayWriter(
        ctx.path("fWpc", ext=".dat"), dtype=np.float32, shape=(NchanNear, Nrank, -1)
    )

    if params.low_memory:
        feature_path = ctx.context_path / 'spike_features'
        os.mkdir(feature_path)

    for ibatch in tqdm(range(niter), desc="Optimizing templates"):
        # korder is the index of the batch at this point in the schedule
        korder = int(irounds[ibatch])
        # k is the index of the batch in absolute terms
        logger.debug("Batch %d/%d, %d templates.", ibatch, niter, Nfilt)

        if ibatch < niter - nBatches:
            # obtained pm for this batch
            Params[8] = float(pmi[ibatch])
            pm = pmi[ibatch] * ones((Nfilt,), dtype=np.float64, order="F")

        # loading a single batch

        # [CR 2024-04-02]: add overlap when loading the data as a work-round to the spike-hole bug
        # This is the only place where we use the overlap option of load_batch()
        overlap = overlap_samples
        dataRAW = ir.data_loader.load_batch(korder, overlap_samples=overlap)
        # NOTE: we check the batch size: this will only work if (a) overlap is disabled for the
        # first batch, and (b) we enforce the overlap to be smaller than the batch size.
        # NOTE: we assume there are at least 2 batches
        if korder == 0:
            assert dataRAW.shape[0] == NT + 1 * overlap
        elif korder < nBatches - 1:
            assert dataRAW.shape[0] == NT + 2 * overlap
        elif korder == nBatches - 1:
            assert dataRAW.shape[0] <= NT + 2 * overlap
        else:
            # NOTE: this should never occur
            raise ValueError(f"korder = {korder}")
        # HACK: we need to pass the new data shape to the CUDA functions: this goes via the
        # first element of the Params array, normally NT, but here it is NT + overlap
        Params[0] = dataRAW.shape[0]

        if ibatch == 0:
            # only on the first batch, we first get a new set of spikes from the residuals,
            # which in this case is the unmodified data because we start with no templates
            # CUDA function to get spatiotemporal clips from spike detections

            # [CR 2024-04-02]: warning, dataRAW contains the overlap, but here, we need to
            # run this function without the overlap as the CUDA code does not expect it.
            dataRAW_no_overlap = ir.data_loader.load_batch(korder, overlap_samples=0)
            assert dataRAW_no_overlap.shape[0] <= NT  # NOTE: might be smaller if last batch
            Params[0] = dataRAW_no_overlap.shape[0]
            dWU, cmap = mexGetSpikes2(Params, dataRAW_no_overlap, wTEMP, iC)

            dWU = cp.asarray(dWU, dtype=np.float64, order="F")

            # project these into the wPCA waveforms
            dWU = cp.reshape(
                cp.dot(
                    wPCAd, cp.dot(wPCAd.T, dWU.reshape((dWU.shape[0], -1), order="F"))
                ),
                dWU.shape,
                order="F",
            )

            # initialize the low-rank decomposition with standard waves
            W = W0[:, cp.ones(dWU.shape[2], dtype=np.int32), :]
            Nfilt = W.shape[1]  # update the number of filters/templates
            # initialize the number of spikes for new templates with the minimum allowed value,
            # so it doesn't get thrown back out right away
            nsp = _extend(nsp, 0, Nfilt, m0)
            Params[1] = Nfilt  # update in the CUDA parameters

        if flag_resort:
            # this is a flag to resort the order of the templates according to best peak
            # channel
            # this is important in order to have cohesive memory requests from the GPU RAM
            # max channel (either positive or negative peak)
            iW = cp.argmax(cp.abs(dWU[nt0min - 1, :, :]), axis=0)
            # iW = int32(squeeze(iW))

            isort = cp.argsort(iW)  # sort by max abs channel
            iW = cp.asfortranarray(iW[isort])
            W = cp.asfortranarray(W[
                :, isort, :
            ])  # user ordering to resort all the other template variables
            dWU = cp.asfortranarray(dWU[:, :, isort])
            nsp = cp.asfortranarray(nsp[isort])

        # decompose dWU by svd of time and space (via covariance matrix of 61 by 61 samples)
        # this uses a "warm start" by remembering the W from the previous iteration
        W, U, mu = mexSVDsmall2(Params, dWU, W, iC, iW, Ka, Kb)

        # UtU is the gram matrix of the spatial components of the low-rank SVDs
        # it tells us which pairs of templates are likely to "interfere" with each other
        # such as when we subtract off a template
        # this needs to change (but I don't know why!)
        UtU, maskU = getMeUtU(iW, iC, mask, Nnearest, Nchan)

        # main CUDA function in the whole codebase. does the iterative template matching
        # based on the current templates, gets features for these templates if requested
        # (featW, featPC),
        # gets scores for the template fits to each spike (vexp), outputs the average of
        # waveforms assigned to each cluster (dWU0),
        # and probably a few more things I forget about
        st0, id0, x0, featW, dWU0, drez, nsp0, featPC, vexp = mexMPnu8(
            Params, dataRAW, U, W, mu, iC, iW, UtU, iList, wPCA, params
        )

        """
        [CR 2024-04-02] shape of the output arrays of mexMPnu8
        (ex: nspikes=3190, nc=384)

        st0: int32      (nspikes,) number of samples since the beginning of the batch
        id0: int32      (nspikes,) template ID?
        x0:  float32    (nspikes,)
        featW: float32  (32, nspikes)
        dWU0: float64   (61, nc, Y) (ex: Y=140)
        drez: float32   (65600, nc)
        nsp0: int32     (Y,)
        featPC: float32 (32, 3, nspikes)
        vexp: float32   (nspikes,)
        """

        # [CR 2024-04-02]: if there was an overlap, we need to remove the spikes in the overlap
        # area! The following variables depend on nspikes: st0, id0, x0, featW, featPC, vexp

        if overlap > 0:
            overlap_idx = (overlap < st0) & (st0 < NT + overlap)
            n_before = len(st0)
            st0 = st0[overlap_idx] - overlap    # (nspikes,)
            assert np.all(st0 < NT)
            n_after = len(st0)
            logger.debug("Removed %d spikes in the overlap", n_before - n_after)
            id0 = id0[overlap_idx]              # (nspikes,)
            x0 = x0[overlap_idx]                # (nspikes,)
            vexp = vexp[overlap_idx]            # (nspikes,)
            featW = featW[:, overlap_idx]       # (32, nspikes)
            featPC = featPC[..., overlap_idx]   # (32, 3, nspikes)

        logger.debug("Found %d spikes.", x0.size)

        # Sometimes nsp can get transposed (think this has to do with it being
        # a single element in one iteration, to which elements are added
        # nsp, nsp0, and pm must all be row vectors (Nfilt x 1), so force nsp
        # to be a row vector.
        # nsp = cp.atleast_2d(nsp)
        # nsprow, nspcol = nsp.shape
        # if nsprow < nspcol:
        #     nsp = nsp.T
        nsp = nsp.squeeze()

        # updates the templates as a running average weighted by recency
        # since some clusters have different number of spikes, we need to apply the
        # exp(pm) factor several times, and fexp is the resulting update factor
        # for each template
        fexp = np.exp(nsp0 * cp.log(pm[:Nfilt]))
        fexp = cp.reshape(fexp, (1, 1, -1), order="F")

        # disable template updates during extraction phase
        if ibatch <= niter - nBatches - 1:
            dWU = dWU * fexp + (1 - fexp) * (
                dWU0 / cp.reshape(cp.maximum(1, nsp0), (1, 1, -1), order="F")
            )

        # nsp just gets updated according to the fixed factor p1
        nsp = nsp * p1 + (1 - p1) * nsp0

        if ibatch == niter - nBatches - 1:
            # if we reached this point, we need to disable secondary template updates
            # like dropping, and adding new templates. We need to memorize the state of the
            # templates at this timepoint, and set the processing mode to "extraction and
            # tracking"

            flag_resort = 0  # no need to resort templates by channel any more
            # flag_final = 1  # this is the "final" pass

            # final clean up, triage templates one last time
            W, U, dWU, mu, nsp, ndrop = triageTemplates2(
                params, iW, C2C, W, U, dWU, mu, nsp, ndrop
            )

            # final number of templates
            Nfilt = W.shape[1]
            Params[1] = Nfilt

            # final covariance matrix between all templates
            WtW, iList = getMeWtW(W, U, Nnearest)

            # iW is the final channel assigned to each template
            iW = cp.asfortranarray(cp.argmax(cp.abs(dWU[nt0min - 1, :, :]), axis=0))

            # extract ALL features on the last pass
            Params[
                12
            ] = 2  # this is a flag to output features (PC and template features)

            # different threshold on last pass?
            Params[2] = params.Th[
                -1
            ]  # usually the threshold is much lower on the last pass

            # memorize the state of the templates
            logger.debug("Memorized middle timepoint.")
            ir.W, ir.dWU, ir.U, ir.mu = W, dWU, U, mu
            ir.Wraw = cp.zeros(
                (U.shape[0], W.shape[0], U.shape[1]), dtype=np.float64, order="F"
            )
            for n in range(U.shape[1]):
                # temporarily use U rather Urot until I have a chance to test it
                ir.Wraw[:, :, n] = mu[n] * cp.dot(U[:, n, :], W[:, n, :].T)

            if params.low_memory:
                # Initialise array writers for the spike features of each cluster
                feature_writers = [
                    LargeArrayWriter(
                        feature_path / f'spike_features_{i}',
                        dtype=np.float32,
                        shape=(NchanNear, Nrank, -1)
                    ) for i in range(Nfilt)
                ]

        if ibatch < niter - nBatches - 1:
            # during the main "learning" phase of fitting a model
            if ibatch % 5 == 0:
                # this drops templates based on spike rates and/or similarities to
                # other templates
                W, U, dWU, mu, nsp, ndrop = triageTemplates2(
                    params, iW, C2C, W, U, dWU, mu, nsp, ndrop
                )

            Nfilt = W.shape[1]  # update the number of filters
            Params[1] = Nfilt

            # this adds new templates if they are detected in the residual
            dWU0, cmap = mexGetSpikes2(Params, drez, wTEMP, iC)

            if dWU0.shape[2] > 0:
                # new templates need to be integrated into the same format as all templates
                # apply PCA for smoothing purposes
                dWU0 = cp.reshape(
                    cp.dot(
                        wPCAd,
                        cp.dot(
                            wPCAd.T,
                            dWU0.reshape(
                                (dWU0.shape[0], dWU0.shape[1] * dWU0.shape[2]),
                                order="F",
                            ),
                        ),
                    ),
                    dWU0.shape,
                    order="F",
                )
                dWU = cp.asfortranarray(cp.concatenate((dWU, dWU0), axis=2))

                m = dWU0.shape[2]
                # initialize temporal components of waveforms
                W = cp.asfortranarray(
                    _extend(
                    W, Nfilt, Nfilt + m, W0[:, cp.ones(m, dtype=np.int32), :], axis=1
                    )
                )

                # initialize the number of spikes with the minimum allowed
                nsp = _extend(nsp, Nfilt, Nfilt + m, params.minFR * NT / params.fs)
                # initialize the amplitude of this spike with a lowish number
                mu = _extend(mu, Nfilt, Nfilt + m, 10)

                # if the number of filters exceed the maximum allowed, clip it
                Nfilt = min(params.Nfilt, W.shape[1])
                Params[1] = Nfilt

                W = W[:, :Nfilt, :]  # remove any new filters over the maximum allowed
                dWU = dWU[
                    :, :, :Nfilt
                ]  # remove any new filters over the maximum allowed
                nsp = nsp[:Nfilt]  # remove any new filters over the maximum allowed
                mu = mu[:Nfilt]  # remove any new filters over the maximum allowed

        if ibatch > niter - nBatches - 1:
            # during the final extraction pass, this keeps track of all spikes and features

            # we carefully assign the correct absolute times to spikes found in this batch
            toff = nt0min + t0 + NT*korder
            st = toff + st0

            st30 = np.c_[
                cp.asnumpy(st),  # spike times
                cp.asnumpy(id0),  # spike clusters (0-indexing)
                cp.asnumpy(x0),  # template amplitudes
                cp.asnumpy(vexp),  # residual variance of this spike
                korder * np.ones(st.size),  # batch from which this spike was found
            ]
            # Check the number of spikes.
            assert st30.shape[0] == featW.shape[1] == featPC.shape[2]

            # Sort by spike times
            ix_sort = np.argsort(st30[:,0])
            st30 = np.asfortranarray(st30[ix_sort])
            featW = cp.asfortranarray(featW[:, ix_sort])
            featPC = cp.asfortranarray(featPC[:, :, ix_sort])

            st3.append(st30)
            fW.append(featW)
            fWpc.append(featPC)

            if params.low_memory:
                # Save spike features individually for each cluster
                for i in range(Nfilt):
                    sub_array = cp.asfortranarray(featPC[:, :, id0 == i])
                    feature_writers[i].append(sub_array)

            ntot = ntot + x0.size  # keeps track of total number of spikes so far

        if ibatch % 100 == 0:
            # this is some of the relevant diagnostic information to be printed during training
            # logger.info(
            #     (
            #         "%d / %d batches, %d units, nspks: %2.4f, mu: %2.4f, "
            #         "nst0: %d, merges: %2.4f, %2.4f"
            #     ),
            #     ibatch,
            #     niter,
            #     Nfilt,
            #     nsp.sum(),
            #     median(mu),
            #     st0.size,
            #     *ndrop
            # )

            if sanity_plots:
                assert plot_widgets is not None, "if sanity_plots is set, then plot_widgets cannot be None"
                plot_diagnostics(W, U, mu, nsp, plot_widgets[plot_pos])

        free_gpu_memory()

    # Close the large array writers and save the JSON metadata files to disk.
    fW.close()
    fWpc.close()

    if params.low_memory:
        # Close the large array writers for each cluster's spike features
        for writer in feature_writers:
            writer.close()

    # just display the total number of spikes
    logger.info("Found a total of %d spikes.", ntot)

    # Save results to the ctx.intermediate object.
    ir.st3 = np.concatenate(st3, axis=0)

    # the similarity score between templates is simply the correlation,
    # taken as the max over several consecutive time delays
    ir.simScore = cp.asnumpy(cp.max(WtW, axis=2))

    # NOTE: these are now already saved by LargeArrayWriter
    # fWa = np.concatenate(fW, axis=-1)
    # fWpca = np.concatenate(fWpc, axis=-1)

    # the template features are stored in cProj, like in Kilosort1
    # ir.cProj = fWa.T
    # the neihboring templates idnices are stored in iNeigh
    ir.iNeigh = cp.asnumpy(iList)

    #  permute the PC projections in the right order
    # ir.cProjPC = np.transpose(fWpca, (2, 1, 0))
    # iNeighPC keeps the indices of the channels corresponding to the PC features
    ir.iNeighPC = cp.asnumpy(iC[:, iW])

    # Number of spikes.
    assert ir.st3.shape[0] == fW.shape[-1] == fWpc.shape[-1]

    # Save cluster times and IDs at this stage if requested
    if params.save_temp_files:
        np.save(ctx.context_path / 'temp_splits' / 'st3_learn.npy', ir.st3)

    # # this whole next block is just done to compress the compressed templates
    # # we separately svd the time components of each template, and the spatial components
    # # this also requires a careful decompression function, available somewhere in the GUI code
    # nKeep = min(Nchan * 3, 20)  # how many PCs to keep
    # W_a = np.zeros((nt0 * Nrank, nKeep, Nfilt), dtype=np.float32)
    # W_b = np.zeros((nBatches, nKeep, Nfilt), dtype=np.float32)
    # U_a = np.zeros((Nchan * Nrank, nKeep, Nfilt), dtype=np.float32)
    # U_b = np.zeros((nBatches, nKeep, Nfilt), dtype=np.float32)
    #
    # for j in tqdm(range(Nfilt), desc="Compressing templates"):
    #     # do this for every template separately
    #     WA = np.reshape(ir.WA[:, j, ...], (-1, nBatches), order="F")
    #     # svd on the GPU was faster for this, but the Python randomized CPU version
    #     # might be faster still
    #     # WA = gpuArray(WA)
    #     A, B, C = svdecon_cpu(WA)
    #     # W_a times W_b results in a reconstruction of the time components
    #     W_a[:, :, j] = np.dot(A[:, :nKeep], B[:nKeep, :nKeep])
    #     W_b[:, :, j] = C[:, :nKeep]
    #
    #     UA = np.reshape(ir.UA[:, j, ...], (-1, nBatches), order="F")
    #     # UA = gpuArray(UA)
    #     A, B, C = svdecon_cpu(UA)
    #     # U_a times U_b results in a reconstruction of the time components
    #     U_a[:, :, j] = np.dot(A[:, :nKeep], B[:nKeep, :nKeep])
    #     U_b[:, :, j] = C[:, :nKeep]
    #
    # logger.info("Finished compressing time-varying templates.")

    #TODO
    # Keep st3, wPCA, wTEMP, simScore, iNeigh, iNeighPC, W, U, dWU
    #

    return Bunch(
        wPCA=wPCA[:, :Nrank],
        wTEMP=wTEMP,
        st3=ir.st3,
        simScore=ir.simScore,
        # cProj=ir.cProj,
        # cProjPC=ir.cProjPC,
        iNeigh=ir.iNeigh,
        iNeighPC=ir.iNeighPC,
        W=ir.W,
        U=ir.U,
        dWU=ir.dWU,
        mu=ir.mu,
        # W_a=W_a,
        # W_b=W_b,
        # U_a=U_a,
        # U_b=U_b,
    )

def compress_templates(ctx):
    """
    Individually compress the temporal and spatial components of the spike templates
    """
    Nbatch = ctx.intermediate.Nbatch
    params = ctx.params
    probe = ctx.probe
    ir = ctx.intermediate

    Nchan = probe.Nchan
    nt0 = params.nt0
    Nfilt = ir.UA.shape[1]
    Nbatch = ir.Nbatch

    Nrank = 3

    nKeep = min(Nchan * 3, 20)  # how many PCs to keep
    W_a = np.zeros((nt0 * Nrank, nKeep, Nfilt), dtype=np.float32)
    W_b = np.zeros((Nbatch, nKeep, Nfilt), dtype=np.float32)
    U_a = np.zeros((Nchan * Nrank, nKeep, Nfilt), dtype=np.float32)
    U_b = np.zeros((Nbatch, nKeep, Nfilt), dtype=np.float32)

    for j in tqdm(range(Nfilt), desc="Compressing templates"):
        # do this for every template separately
        WA = np.reshape(ir.WA[:, j, ...], (-1, Nbatch), order="F")
        # svd on the GPU was faster for this, but the Python randomized CPU version
        # might be faster still
        # WA = gpuArray(WA)
        A, B, C = svdecon_cpu(WA)
        # W_a times W_b results in a reconstruction of the time components
        W_a[:, :, j] = np.dot(A[:, :nKeep], B[:nKeep, :nKeep])
        W_b[:, :, j] = C[:, :nKeep]

        UA = np.reshape(ir.UA[:, j, ...], (-1, Nbatch), order="F")
        # UA = gpuArray(UA)
        try:
            A, B, C = svdecon_cpu(UA)
        except np.linalg.LinAlgError:
            logger.error("SVD did not converge")
            continue
        # U_a times U_b results in a reconstruction of the time components
        U_a[:, :, j] = np.dot(A[:, :nKeep], B[:nKeep, :nKeep])
        U_b[:, :, j] = C[:, :nKeep]

    logger.info("Finished compressing time-varying templates.")

    #TODO
    # Write these to the disk

    return Bunch(
        W_a=W_a,
        W_b=W_b,
        U_a=U_a,
        U_b=U_b,
    )
