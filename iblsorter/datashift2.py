from math import floor
import logging
import os

from tqdm.auto import tqdm, trange
import numpy as np
import cupy as cp
import dartsort
from dredge import dredge_ap
import iblsorter.qc
from scipy.interpolate import Akima1DInterpolator
from scipy.sparse import coo_matrix
import spikeinterface.full as si
import torch

from .postprocess import my_conv2_cpu
from .learn import extractTemplatesfromSnippets
from .utils import get_cuda, Bunch

logger = logging.getLogger(__name__)


def getClosestChannels2(ycup, xcup, yc, xc, NchanClosest):
    # this function outputs the closest channels to each channel,
    # as well as a Gaussian-decaying mask as a function of pairwise distances
    # sigma is the standard deviation of this Gaussian-mask
    # compute distances between all pairs of channels
    xc = cp.asarray(xc, dtype=np.float32, order='F')
    yc = cp.asarray(yc, dtype=np.float32, order='F')
    xcup = cp.asarray(xcup, dtype=np.float32, order='F')
    ycup = cp.asarray(ycup, dtype=np.float32, order='F')
    C2C = ((xc[:, np.newaxis] - xcup[:].T.flatten()[:, np.newaxis].T) ** 2 + (
                yc[:, np.newaxis] - ycup[:].T.flatten()[:, np.newaxis].T) ** 2)
    C2C = cp.sqrt(C2C)
    Nchan, NchanUp = C2C.shape

    # sort distances
    isort = cp.argsort(C2C, axis=0)

    # take NchanCLosest neighbors for each primary channel
    iC = isort[:NchanClosest, :]

    # in some cases we want a mask that decays as a function of distance between pairs of channels
    # this is an awkward indexing to get the corresponding distances
    ix = iC + cp.arange(0, Nchan * NchanUp, Nchan)
    dist = C2C.T.ravel()[ix]

    return iC, dist


def get_batch(params, probe, ibatch, Nbatch, proc) -> cp.ndarray:
    batchstart = np.arange(0, params.NT * Nbatch + 1, params.NT).astype(np.int64)

    offset = probe.Nchan * batchstart[ibatch]
    dat = proc.flat[offset : offset + params.NT * probe.Nchan].reshape(
        (-1, probe.Nchan), order="F"
    )

    # move data to GPU and scale it back to unit variance
    dataRAW = cp.asarray(dat, dtype=np.float32) / params.scaleproc
    return dataRAW


def spikedetector3(Params, drez, wTEMP, iC, dist, v2, iC2, dist2):
    code, constants = get_cuda("spikedetector3")

    NT = int(Params[0])
    Nchan = int(Params[1])
    nt0 = int(Params[2])
    Nnearest = int(Params[3])
    Nrank = int(Params[4])
    NchanUp = int(Params[7])

    Nthreads = constants.Nthreads
    NrankMax = constants.NrankMax
    maxFR = constants.maxFR
    nt0max = constants.nt0max
    NchanMax = constants.NchanMax
    nsizes = constants.nsizes

    # tpB = (8, 2 * nt0 - 1)
    # tpF = (16, Nnearest)
    tpS = (nt0, 16)

    d_Params = cp.asarray(Params, dtype=np.float64, order="F")
    d_data = cp.asarray(drez, dtype=np.float32, order="F")
    d_W = cp.asarray(wTEMP, dtype=np.float32, order="F")
    d_iC = cp.asarray(iC, dtype=np.int32, order="F")
    d_dist = cp.asarray(dist, dtype=np.float32, order="F")
    d_v2 = cp.asarray(v2, dtype=np.float32, order="F")
    d_iC2 = cp.asarray(iC2, dtype=np.int32, order="F")
    d_dist2 = cp.asarray(dist2, dtype=np.float32, order="F")

    dimst = (NT, NchanUp)
    d_dout = cp.zeros(dimst, dtype=np.float32, order="F")
    d_kkmax = cp.zeros(dimst, dtype=np.int32, order="F")

    d_dfilt = cp.zeros((Nrank, NT, Nchan), dtype=np.float32, order="F")
    d_dmax = cp.zeros((NT, NchanUp), dtype=np.float32, order="F")
    d_st = cp.zeros((maxFR * 5), dtype=np.int32, order="F")
    d_cF = cp.zeros((maxFR * Nnearest,), dtype=np.float32, order="F")
    d_counter = cp.zeros(2, dtype=np.int32, order="F")

    # filter the data with the temporal templates
    Conv1D = cp.RawKernel(code, "Conv1D")
    Conv1D((Nchan,), (Nthreads,), (d_Params, d_data, d_W, d_dfilt))

    # sum each template across channels, square, take max
    sumChannels = cp.RawKernel(code, "sumChannels")
    tpP = (int(NT / Nthreads), NchanUp)
    sumChannels(
        tpP, (Nthreads,), (d_Params, d_dfilt, d_dout, d_kkmax, d_iC, d_dist, d_v2)
    )

    # get the max of the data
    max1D = cp.RawKernel(code, "max1D")
    max1D((NchanUp,), (Nthreads,), (d_Params, d_dout, d_dmax))

    # take max across nearby channels
    tpP = (int(NT / Nthreads), NchanUp)
    maxChannels = cp.RawKernel(code, "maxChannels")
    maxChannels(
        tpP,
        (Nthreads,),
        (
            d_Params,
            d_dout,
            d_dmax,
            d_iC,
            d_iC2,
            d_dist2,
            d_kkmax,
            d_dfilt,
            d_st,
            d_counter,
            d_cF,
        ),
    )
    counter = cp.asnumpy(d_counter)[0]

    minSize = min(maxFR, counter)
    d_sto = d_st[: 4 * minSize].reshape((4, minSize), order="F")
    d_cF2 = d_cF[: Nnearest * minSize].reshape((Nnearest, minSize), order="F")

    return d_dout.get(), d_kkmax.get(), d_sto.get(), d_cF2.get()


def kernelD(xp0, yp0, length):
    D = xp0.shape[0]
    N = xp0.shape[1] if len(xp0.shape) > 1 else 1
    M = yp0.shape[1] if len(yp0.shape) > 1 else 1

    K = np.zeros((N, M))
    cs = M

    for i in range(int(M * 1.0 / cs)):
        ii = np.arange(i * cs, min(M, (i + 1) * cs))
        mM = len(ii)

        xp = np.tile(xp0, (mM, 1)).T[np.newaxis, :, :]
        yp = np.tile(yp0[:, ii], (N, 1)).reshape((D, N, mM))
        a = (xp - yp) ** 2
        b = 1.0 / (length ** 2)
        Kn = np.exp(-np.sum((a * b) / 2, axis=0))

        K[:, ii] = Kn

    return K


def align_block2(F, ysamp, nblocks):

    # F is y bins by amp bins by batches
    # ysamp are the coordinates of the y bins in um

    Nbatches = F.shape[2]

    # look up and down this many y bins to find best alignment
    n = 15
    dc = np.zeros((2 * n + 1, Nbatches))
    dt = np.arange(-n, n + 1)

    # we do everything on the GPU for speed, but it's probably fast enough on
    # the CPU
    Fg = F

    # mean subtraction to compute covariance
    Fg = Fg - np.mean(Fg, axis=0)

    # initialize the target "frame" for alignment with a single sample
    F0 = Fg[:, :, min(299, np.floor(Fg.shape[2] / 2).astype("int")) - 1]
    F0 = F0[:, :, np.newaxis]

    # first we do rigid registration by integer shifts
    # everything is iteratively aligned until most of the shifts become 0.
    niter = 10
    dall = np.zeros((niter, Nbatches))
    for iteration in range(niter):
        for t in range(len(dt)):
            # for each NEW potential shift, estimate covariance
            Fs = np.roll(Fg, dt[t], axis=0)
            dc[t, :] = np.mean(Fs * F0, axis=(0,1))
        if iteration + 1 < niter:
            # up until the very last iteration, estimate the best shifts
            imax = np.argmax(dc, axis=0)
            # align the data by these integer shifts
            for t in range(len(dt)):
                ib = imax == t
                Fg[:, :, ib] = np.roll(Fg[:, :, ib], dt[t], axis=0)
                dall[iteration, ib] = dt[t]
            # new target frame based on our current best alignment
            F0 = np.mean(Fg, axis=2)[:, :, np.newaxis]

    # now we figure out how to split the probe into nblocks pieces
    # if nblocks = 1, then we're doing rigid registration
    nybins = F.shape[0]
    yl = np.floor(nybins / nblocks).astype("int") - 1
    # MATLAB rounds 0.5 to 1. Python uses "Bankers Rounding".
    # Numpy uses round to nearest even. Force the result to be like MATLAB
    # by adding a tiny constant.
    ifirst = np.round(np.linspace(0, nybins - yl - 1, 2 * nblocks - 1) + 1e-10).astype(
        "int"
    )
    ilast = ifirst + yl  # 287

    ##

    nblocks = len(ifirst)
    yblk = np.zeros(len(ifirst))

    # for each small block, we only look up and down this many samples to find
    # nonrigid shift
    n = 5
    dt = np.arange(-n, n + 1)

    # this part determines the up/down covariance for each block without
    # shifting anything
    dcs = np.zeros((2 * n + 1, Nbatches, nblocks))
    for j in range(nblocks):
        isub = np.arange(ifirst[j], ilast[j]+1)
        yblk[j] = np.mean(ysamp[isub])
        Fsub = Fg[isub, :, :]
        for t in range(len(dt)):
            Fs = np.roll(Fsub, dt[t], axis=0)
            dcs[t, :, j] = np.mean(Fs * F0[isub, :, :], axis=(0,1))

    # to find sub-integer shifts for each block ,
    # we now use upsampling, based on kriging interpolation
    dtup = np.linspace(-n, n, (2 * n * 10) + 1)
    K = kernelD(
        dt[np.newaxis, :], dtup[np.newaxis], 1
    )  # this kernel is fixed as a variance of 1
    dcs = cp.array(dcs)
    # dcs = my_conv2_cpu(dcs, .5, [0,1,2])
    for i in range(dcs.shape[0]):
        dcs[i, :, :] = my_conv2_cpu(
            dcs[i, :, :], 0.5, [0, 1]
        )  # some additional smoothing for robustness, across all dimensions
    for i in range(dcs.shape[2]):
        dcs[:, :, i] = my_conv2_cpu(
            dcs[:, :, i], 0.5, [0]
        )  # some additional smoothing for robustness, across all dimensions
        # dcs = my_conv2(cp.array(dcs), .5, [1, 2]) # some additional smoothing for robustness, across all dimensions
    dcs = dcs.get()

    # return K, dcs, dt, dtup
    imin = np.zeros((Nbatches, nblocks))
    for j in range(nblocks):
        # using the upsampling kernel K, get the upsampled cross-correlation
        # curves
        dcup = np.matmul(K.T, dcs[:, :, j])

        # find the max index of these curves
        imax = np.argmax(dcup, axis=0)

        # add the value of the shift to the last row of the matrix of shifts
        # (as if it was the last iteration of the main rigid loop )
        dall[niter - 1, :] = dtup[imax]

        # the sum of all the shifts equals the final shifts for this block
        imin[:, j] = np.sum(dall, axis=0)

    return imin, yblk, F0


def extended(ysamp, n, diff=None):
    if diff is None:
        diff = ysamp[1] - ysamp[0]
    pre = [ysamp[0] - i * diff for i in range(n, 0, -1)]
    post = [ysamp[-1] + i * diff for i in range(1, n)]
    return np.concatenate([pre, ysamp, post])


def zero_pad(shifts_in, n):
    pre = [0 for i in range(n, 0, -1)]
    post = [0 for i in range(1, n)]
    return np.concatenate([pre, shifts_in, post])


def kernel2D(xp, yp, sig):
    distx = np.abs(xp[:, 0] - yp[:, 0][np.newaxis, :].T).T
    disty = np.abs(xp[:, 1] - yp[:, 1][np.newaxis, :].T).T

    sigx = sig
    sigy = 1.5 * sig

    p = 1
    K = np.exp(-((distx / sigx) ** p) - (disty / sigy) ** p)

    return K


def shift_data(data, shift_matrix):
    """
    Applies the shift transformation to the data via matrix multiplication
    :param data: Data matrix to be shifted, numpy memmap array, (n_time, n_channels)
                dtype int16, f-contiguous
    :param shift_matrix: Tranformation matrix, numpy array, (n_channels, n_channels)
                dtype float64, c-contiguous
    :return: Shifted data, numpy array, (n_time, n_channels)
                dtype int16, f-contiguous
    """

    data_shifted = np.asfortranarray((data @ shift_matrix.T).astype("int16"))

    return data_shifted


def interpolate_1D(sample_shifts, sample_coords, probe_coords):
    """
    Interpolates the shifts found in one dimension to estimate the shifts for each channel
    :param sample_shifts: Detected shifts, numpy array
    :param sample_coords: Coordinates at which the detected shifts were found, numpy array
    :param probe_coords: Coordinates of the probe channels, numpy array
    :return: Upsampled shifts for each channel, numpy array
    """

    assert len(sample_coords) == len(sample_shifts)

    if len(sample_coords) == 1:
        return np.full(len(probe_coords), sample_shifts[0])

    else:
        # zero pad input so "extrapolation" tends to zero
        # MATLAB uses a "modified Akima" which is proprietry :(
        # _ysamp = extended(ysamp, 2, 10000)
        # _shifts_in = zero_pad(shifts_in, 2)
        # interpolation_function = Akima1DInterpolator(_ysamp, _shifts_in)
        interpolation_function = Akima1DInterpolator(sample_coords, sample_shifts)

        # interpolation_function = interp1d(ysamp, shifts_in, kind='cubic', fill_value=([0],[0])) #'extrapolate')
        shifts = interpolation_function(probe_coords, nu=0, extrapolate="True")

        return shifts


def get_kernel_matrix(probe, shifts, sig):
    """
    Calculate kernel prediction matrix for Gaussian Kriging interpolation
    :param probe: Bunch object with individual numpy arrays for the channel coordinates
    :param shifts: Numpy array of the estimated shift for each channel
    :param sig: Standard deviation used in Gaussian interpolation, float value
    :return: Prediction matrix, numpy array
    """

    # 2D coordinates of the original channel positions
    coords_old = np.vstack([probe.xc, probe.yc]).T

    # 2D coordinates of the new channel positions
    coords_new = np.copy(coords_old)
    coords_new[:, 1] = coords_new[:, 1] - shifts

    # 2D kernel of the original channel positions
    Kxx = kernel2D(coords_old, coords_old, sig)

    # 2D kernel of the new channel positions
    Kyx = kernel2D(coords_new, coords_old, sig)

    # kernel prediction matrix
    prediction_matrix = Kyx @ np.linalg.pinv(Kxx + 0.01 * np.eye(Kxx.shape[0]))

    return prediction_matrix


def apply_drift_transform(dat, shifts_in, ysamp, probe, sig):
    """
    Apply kriging interpolation on data batch
    :param dat: Data batch, (n_time, n_channels)
    :param shifts_in: Shifts per block (n_blocks)
    :param ysamp: Y-coords for block centres (n_blocks)
    :param probe: Bunch object with xc (xcoords) and yc (ycoords) attributes
    :param sig: Standard deviation for Gaussian in kriging interpolation
    :return: Shifted data batch via a kriging transformation
    """

    # upsample to get shifts for each channel
    shifts = interpolate_1D(shifts_in, ysamp, probe.yc)

    # kernel prediction matrix
    kernel_matrix = get_kernel_matrix(probe, shifts, sig)

    # apply shift transformation to the data
    data_shifted = shift_data(dat, kernel_matrix)

    return data_shifted


def shift_batch_on_disk2(
    ibatch,
    shifts_in,
    ysamp,
    sig,
    probe,
    data_loader,
):

    # load the batch
    dat = data_loader.load_batch(ibatch, rescale=False)

    # upsample the shift for each channel using interpolation
    shifts = interpolate_1D(shifts_in, ysamp, probe.yc)

    # kernel prediction matrix
    kernel_matrix = get_kernel_matrix(probe, shifts, sig)

    # apply shift transformation to the data
    data_shifted = shift_data(dat, kernel_matrix)

    # write the aligned data back to the same file
    data_loader.write_batch(ibatch, data_shifted)


def dartsort_detector(ctx, probe, params):

    # load preprocessed data via spikeinterface
    rec = si.read_binary(
        ctx.intermediate.proc_path, params.fs, params.data_dtype, probe.Nchan
    )
    geom = np.c_[probe.x, probe.y]

    rec.set_dummy_probe_from_locations(geom)
    rec = rec.astype("float32")
    ns = rec.get_total_samples()

    if params.do_unwhiten_before_drift:
        # un-whiten the data to be fed into the dartsort spike detector
        # ignore OOB channels prior to computing a pseudoinverse of Wrot
        Wrot = cp.asnumpy(ctx.intermediate.Wrot)
        W = np.eye(probe.Nchan)
        oob = probe.channel_labels != 3.
        idx = np.ix_(oob, oob)
        W[idx] = np.linalg.pinv(Wrot[idx])
        W /= params.scaleproc
    
        # transform the data to standard units via z-scoring, i.e. Z = (X-MU)/STDEV
        # estimate the STDEV and MU vectors from chunks throughout recording
        nchunk = 25
        t0s = np.linspace(10, ns / params.fs - 10, nchunk)
        stds = np.zeros((nchunk, probe.Nchan))
        mus = np.zeros((nchunk, probe.Nchan))
        for icc, t0 in enumerate(tqdm(t0s)):
            s0, s1 = int(t0 * params.fs), int((t0 + 0.4) * params.fs)
            # z-scoring is after unwhitening
            chunk = W @ rec.get_traces(0, s0, s1).T
            stds[icc, :] = np.std(chunk, axis=1)
            mus[icc, :] = np.mean(chunk, axis=1)
    
        std = np.median(stds, axis=0)
        mu = np.median(mus, axis=0)
    
        W[idx] /= std[oob]
    
        rec = si.whiten(rec, W=W, M=mu, apply_mean=True)

    # now run DARTsort thresholding to get spike times, amps, positions
    channel_index = torch.tensor(dartsort.make_channel_index(geom, 100.0))
    pipeline = dartsort.WaveformPipeline(
        (
            dartsort.transform.SingleChannelWaveformDenoiser(channel_index),
            dartsort.transform.Localization(channel_index, torch.tensor(geom)),
            dartsort.transform.MaxAmplitude(name_prefix="denoised"),
        )
    )

    peeler = dartsort.peel.ThresholdAndFeaturize(
                rec,
                detection_threshold=6,
                chunk_length_samples=4 * int(rec.sampling_frequency),
                max_spikes_per_chunk=4 * 10_000,
                channel_index=channel_index,
                featurization_pipeline=pipeline,
                n_chunks_fit=100,
                spatial_dedup_channel_index=torch.tensor(
                    dartsort.make_channel_index(geom, 75.0)
                ),
            )
    # at 40k spikes per chunk, should factor 2 Gb memory per job
    total_memory = torch.cuda.mem_get_info()[1] / 1024 ** 3
    njobs = int(np.minimum(np.floor(total_memory / (2 * 1.5)), 12))
    logger.info(f"Using {njobs} GPU jobs for spike detection, targeting memory usage of {njobs * 2} Gb over {total_memory} Gb")
    peeler.load_or_fit_and_save_models(
                ctx.context_path / "thresholding_models", n_jobs=njobs
            )
    peeler.peel(
            ctx.context_path / "thresholding.h5",
            n_jobs=njobs,
        )
    st = dartsort.DARTsortSorting.from_peeling_hdf5(ctx.context_path / "thresholding.h5")

    spikes = Bunch()

    margin = 20
    z = st.point_source_localizations[:, 2]
    valid = z == z.clip(geom[:, 1].min() - margin, geom[:, 1].max() + margin)

    spikes.depths = z[valid][:]
    spikes.amps = st.denoised_ptp_amplitudes[valid][:]
    spikes.times = st.times_seconds[valid][:]

    return spikes


def get_drift(spikes, probe, Nbatches, nblocks=5, genericSpkTh=10):
    """
    Estimates the drift using the spiking activity found in the first pass through the data
    :param spikes: Bunch object, contains the depths, amplitudes, times and batches of the spikes.
                    Each attribute is stored as a 1d numpy array
    :param probe: Bunch object, contains the x and y coordinates stored as 1D numpy arrays
    :param Nbatches: No batches in the dataset
    :param nblocks: No of blocks to divide the probe into when estimating drift
    :param genericSpkTh: Min amplitude of spiking activity found
    :return: dshift: 2D numpy array of drift estimates per batch and per sub-block in um
                    size (Nbatches, 2*nblocks-1)
            yblk: 1D numpy array containing average y position of each sub-block
    """

    # binning width across Y (um)
    dd = 5

    # min and max for the range of depths
    dmin = min(probe.yc) - 1

    dmax = int(1 + np.ceil((max(probe.yc) - dmin) / dd))

    # preallocate matrix of counts with 20 bins, spaced logarithmically
    F = np.zeros((dmax, 20, Nbatches))
    for t in range(Nbatches):
        # find spikes in this batch
        ix = np.where(spikes.batches == t)[0]

        # subtract offset
        spike_depths_batch = spikes.depths[ix] - dmin

        # amplitude bin relative to the minimum possible value
        spike_amps_batch = np.log10(np.clip(spikes.amps[ix], None, 99)) - np.log10(genericSpkTh)
        # normalization by maximum possible value
        spike_amps_batch = spike_amps_batch / (np.log10(100) - np.log10(genericSpkTh))

        # multiply by 20 to distribute a [0,1] variable into 20 bins
        # sparse is very useful here to do this binning quickly
        i, j, v, m, n = (
            np.ceil(1e-5 + spike_depths_batch / dd).astype("int"),
            np.minimum(np.ceil(1e-5 + spike_amps_batch * 20), 20).astype("int"),
            np.ones(len(ix)),
            dmax,
            20,
        )
        M = coo_matrix((v, (i-1, j-1)), shape=(m,n)).toarray()

        # the counts themselves are taken on a logarithmic scale (some neurons
        # fire too much!)
        F[:, :, t] = np.log2(1 + M)

    ysamp = dmin + dd * np.arange(1, dmax+1) - dd / 2
    imin, yblk, F0 = align_block2(F, ysamp, nblocks)

    # convert to um
    dshift = imin * dd

    return dshift, yblk


def get_dredge_drift(spikes, params):
    
    motion_est, _ = dredge_ap.register(
        spikes.amps,
        spikes.depths,
        spikes.times,
        bin_s=params.NT / params.fs,
        gaussian_smoothing_sigma_s=params.NT / params.fs,
        mincorr=0.5,
    )
    
    dshift = -motion_est.displacement.T
    yblk = motion_est.spatial_bin_centers_um
    
    return motion_est, dshift, yblk


def datashift2(ctx, qc_path=None):
    """
    Main function to re-register the preprocessed data
    """
    params = ctx.params
    probe = ctx.probe
    raw_data = ctx.raw_data
    ir = ctx.intermediate
    Nbatch = ir.Nbatch

    spikes = dartsort_detector(ctx, probe, params)

    # from brainbox.plot import driftmap
    # from viewephys.gui import viewephys
    #
    # driftmap(spikes['times'] / 30_000, spikes['depths'], plot_style='bincount', vmax=1)
    # t = 1400
    #
    # (-1, ir.data_loader.n_channels)
    # first, last = int(t * 30_000), int(((t + 1) * 30_000))
    # whiten = ir.data_loader.data[first * ir.data_loader.n_channels:last * ir.data_loader.n_channels].reshape(-1, ir.data_loader.n_channels)
    # viewephys(whiten.T, fs=30_000, title='whiten')
    # ir.data_loader.n_channels
    #
    # unwhite = (whiten @ np.linalg.pinv(cp.asnumpy(ir.Wrot) / params.scaleproc)).T
    # viewephys(unwhite, fs=30_000, title='unwhite')

    if params.save_drift_spike_detections:
        drift_path = ctx.context_path / 'drift'
        if not os.path.isdir(drift_path):
            os.mkdir(drift_path)
        np.save(drift_path / 'spike_times.npy', spikes.times)
        np.save(drift_path / 'spike_depths.npy', spikes.depths)
        np.save(drift_path / 'spike_amps.npy', spikes.amps)

    motion_est, dshift, yblk = get_dredge_drift(spikes, params)

    if qc_path is not None:
        iblsorter.qc.plot_motion_correction(motion_est, spikes, qc_path)

    # sort in case we still want to do "tracking"
    iorig = np.argsort(np.mean(dshift, axis=1))

    # register the data in-place batch by batch
    for ibatch in tqdm(range(Nbatch), desc='Shifting Data'):

        # load the batch from binary file
        dat = ir.data_loader.load_batch(ibatch, rescale=False)

        # align via kriging interpolation
        data_shifted = apply_drift_transform(dat, dshift[ibatch, :], yblk, probe, params.sig_datashift).astype(np.dtype(params.data_dtype))

        # write the aligned data back to the same file
        ir.data_loader.write_batch(ibatch, data_shifted)

    logger.info(f"Shifted up/down {Nbatch} batches")

    return Bunch(iorig=iorig, dshift=dshift, yblk=yblk)
