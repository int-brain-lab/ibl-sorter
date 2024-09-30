import logging
from math import ceil
from pathlib import Path

import numpy as np
from scipy.signal import butter
import scipy.stats
import cupy as cp
from tqdm.auto import tqdm

import iblsorter.qc
from .cptools import lfilter, median
from ibldsp.voltage import decompress_destripe_cbin, destripe, detect_bad_channels

logger = logging.getLogger(__name__)


def get_filter_params(fs, fshigh=None, fslow=None):
    if fslow and fslow < fs / 2:
        # butterworth filter with only 3 nodes (otherwise it's unstable for float32)
        return butter(3, (2 * fshigh / fs, 2 * fslow / fs), 'bandpass')
    else:
        # butterworth filter with only 3 nodes (otherwise it's unstable for float32)
        return butter(3, fshigh / fs * 2, 'high')


def gpufilter(buff, chanMap=None, fs=None, fslow=None, fshigh=None, car=True):
    # filter this batch of data after common average referencing with the
    # median
    # buff is timepoints by channels
    # chanMap are indices of the channels to be kep
    # params.fs and params.fshigh are sampling and high-pass frequencies respectively
    # if params.fslow is present, it is used as low-pass frequency (discouraged)

    dataRAW = buff  # .T  # NOTE: we no longer use Fortran order upstream
    assert dataRAW.flags.c_contiguous
    assert dataRAW.ndim == 2
    assert dataRAW.shape[0] > dataRAW.shape[1]
    if chanMap is not None and len(chanMap):
        dataRAW = dataRAW[:, chanMap]  # subsample only good channels
    assert dataRAW.ndim == 2

    # subtract the mean from each channel
    dataRAW = dataRAW - cp.mean(dataRAW, axis=0)  # subtract mean of each channel
    assert dataRAW.ndim == 2

    # CAR, common average referencing by median
    if car:
        # subtract median across channels
        dataRAW = dataRAW - median(dataRAW, axis=1)[:, np.newaxis]

    # set up the parameters of the filter
    filter_params = get_filter_params(fs, fshigh=fshigh, fslow=fslow)

    # next four lines should be equivalent to filtfilt (which cannot be
    # used because it requires float64)
    datr = lfilter(*filter_params, dataRAW, axis=0)  # causal forward filter
    datr = lfilter(*filter_params, datr, axis=0, reverse=True)  # backward
    return datr


# TODO: unclear - Do we really need these, can we not just pick a type for the config?
#               - We can move this complexity into a "config parsing" stage.
def _is_vect(x):
    return hasattr(x, '__len__') and len(x) > 1


def _make_vect(x):
    if not hasattr(x, '__len__'):
        x = np.array([x])
    return x


# TODO: design - can we abstract "running function" out so we don't duplicate most of the code in
#              - my_min and my_max.
def my_min(S1, sig, varargin=None):
    # returns a running minimum applied sequentially across a choice of dimensions and bin sizes
    # S1 is the matrix to be filtered
    # sig is either a scalar or a sequence of scalars, one for each axis to be filtered.
    #  it's the plus/minus bin length for the minimum filter
    # varargin can be the dimensions to do filtering, if len(sig) != x.shape
    # if sig is scalar and no axes are provided, the default axis is 2
    idims = 1
    if varargin is not None:
        idims = varargin
    idims = _make_vect(idims)
    if _is_vect(idims) and _is_vect(sig):
        sigall = sig
    else:
        sigall = np.tile(sig, len(idims))

    for sig, idim in zip(sigall, idims):
        Nd = S1.ndim
        S1 = cp.transpose(S1, [idim] + list(range(0, idim)) + list(range(idim + 1, Nd)))
        dsnew = S1.shape
        S1 = cp.reshape(S1, (S1.shape[0], -1), order='F' if S1.flags.f_contiguous else 'C')
        dsnew2 = S1.shape
        S1 = cp.concatenate(
            (cp.full((sig, dsnew2[1]), np.inf), S1, cp.full((sig, dsnew2[1]), np.inf)), axis=0)
        Smax = S1[:dsnew2[0], :]
        for j in range(1, 2 * sig + 1):
            Smax = cp.minimum(Smax, S1[j:j + dsnew2[0], :])
        S1 = cp.reshape(Smax, dsnew, order='F' if S1.flags.f_contiguous else 'C')
        S1 = cp.transpose(S1, list(range(1, idim + 1)) + [0] + list(range(idim + 1, Nd)))
    return S1


def my_sum(S1, sig, varargin=None):
    # returns a running sum applied sequentially across a choice of dimensions and bin sizes
    # S1 is the matrix to be filtered
    # sig is either a scalar or a sequence of scalars, one for each axis to be filtered.
    #  it's the plus/minus bin length for the summing filter
    # varargin can be the dimensions to do filtering, if len(sig) != x.shape
    # if sig is scalar and no axes are provided, the default axis is 2
    idims = 1
    if varargin is not None:
        idims = varargin
    idims = _make_vect(idims)
    if _is_vect(idims) and _is_vect(sig):
        sigall = sig
    else:
        sigall = np.tile(sig, len(idims))

    for sig, idim in zip(sigall, idims):
        Nd = S1.ndim
        S1 = cp.transpose(S1, [idim] + list(range(0, idim)) + list(range(idim + 1, Nd)))
        dsnew = S1.shape
        S1 = cp.reshape(S1, (S1.shape[0], -1), order='F')
        dsnew2 = S1.shape
        S1 = cp.concatenate(
            (cp.full((sig, dsnew2[1]), 0), S1, cp.full((sig, dsnew2[1]), 0)), axis=0)
        Smax = S1[:dsnew2[0], :]
        for j in range(1, 2 * sig + 1):
            Smax = Smax + S1[j:j + dsnew2[0], :]
        S1 = cp.reshape(Smax, dsnew, order='F')
        S1 = cp.transpose(S1, list(range(1, idim + 1)) + [0] + list(range(idim + 1, Nd)))
    return S1


def whiteningFromCovariance(CC, epsilon=1e-6):
    # function Wrot = whiteningFromCovariance(CC)
    # takes as input the matrix CC of channel pairwise correlations
    # outputs a symmetric rotation matrix (also Nchan by Nchan) that rotates
    # the data onto uncorrelated, unit-norm axes

    # covariance eigendecomposition (same as svd for positive-definite matrix)
    E, D, _ = cp.linalg.svd(CC)
    Di = cp.diag(1. / (D + epsilon) ** .5)
    Wrot = cp.dot(cp.dot(E, Di), E.T)  # this is the symmetric whitening matrix (ZCA transform)
    return Wrot


def whiteningLocal(CC, yc, xc, nRange, epsilon=1e-6):
    # function to perform local whitening of channels
    # CC is a matrix of Nchan by Nchan correlations
    # yc and xc are vector of Y and X positions of each channel
    # nRange is the number of nearest channels to consider
    Wrot = cp.zeros((CC.shape[0], CC.shape[0]))
    for j in range(CC.shape[0]):
        ds = (xc - xc[j]) ** 2 + (yc - yc[j]) ** 2
        ilocal = np.argsort(ds)
        # take the closest channels to the primary channel.
        # First channel in this list will always be the primary channel.
        ilocal = ilocal[:nRange]

        wrot0 = cp.asnumpy(whiteningFromCovariance(CC[np.ix_(ilocal, ilocal)], epsilon=epsilon))
        # the first column of wrot0 is the whitening filter for the primary channel
        Wrot[ilocal, j] = wrot0[:, 0]

    return Wrot


def get_data_covariance_matrix(raw_data, params, probe, nSkipCov=None):
    """
    Computes the data covariance matrix from a raw data file. Computes from a set of samples from the raw
    data
    :param raw_data:
    :param params:
    :param probe:
    :param nSkipCov: for the orginal kilosort2, number of batches to skip between samples
    :return:
    """
        # takes 25 samples of 500ms from 10 seconds to t -25s
    good_channels = probe.good_channels
    CCall = np.zeros((25, probe.Nchan, probe.Nchan))
    t0s = np.linspace(10, raw_data.shape[0] / params.fs - 10, 25)
    # on the second pass, apply destriping to the data
    for icc, t0 in enumerate(tqdm(t0s, desc="Computing covariance matrix")):
        s0 = slice(int(t0 * params.fs), int((t0 + 0.4) * params.fs))
        raw = raw_data[s0][:, :probe.Nchan].T.astype(np.float32) * probe['sample2volt']
        datr = destripe(raw, fs=params.fs, h=probe, channel_labels=probe.channel_labels) / probe['sample2volt']
        assert not (np.any(np.isnan(datr)) or np.any(np.isinf(datr))), "destriping unexpectedly produced NaNs"
        # attempt at fancy regularization, but this was very slow
        # from joblib import Parallel, delayed, cpu_count
        # import sklearn.covariance
        # cov = sklearn.covariance.GraphicalLassoCV(n_jobs=cpu_count(), assume_centered=True).fit(datr.T)
        # CCall[icc, :, :] = cov.covariance_
        CCall[icc, :, :] = np.dot(datr, datr.T) / datr.shape[1]
        # remove the bad channels from the covariance matrix, those get only divided by their rms
        CCall[icc, :, :] = np.dot(datr, datr.T) / datr.shape[1]
        CCall[icc, ~probe.good_channels, :] = 0
        CCall[icc, :, ~probe.good_channels] = 0
        # we stabilize the covariance matrix this is not trivial:
        # first remove all cross terms belonging to the bad channels
        median_std = np.median(np.diag(CCall[icc, :, :])[good_channels])
        # channels flagged as noisy (label=1) and dead (label=2) are interpolated
        replace_diag = np.interp(np.where(~good_channels)[0], np.where(good_channels)[0], np.diag(CCall[icc, :, :])[good_channels])
        CCall[icc, ~good_channels, ~good_channels] = replace_diag
        # the channels outside of the brain (label=3) are willingly excluded and stay with the high values above
        CCall[icc, probe.channel_labels == 3, probe.channel_labels == 3] = median_std * 1e6
    CC = cp.asarray(np.median(CCall, axis=0))
    return CC


def get_whitening_matrix(raw_data=None, probe=None, params=None, qc_path=None):
    """
    based on a subset of the data, compute a channel whitening matrix
    this requires temporal filtering first (gpufilter)
    :param raw_data:
    :param probe:
    :param params:
    :param kwargs: get_data_covariance_matrix kwargs
    """
    Nchan = probe.Nchan
    CC = get_data_covariance_matrix(raw_data, params, probe)
    logger.info(f"Data normalisation using {params.normalisation} method")
    if params.normalisation in ['whitening', 'original']:
        epsilon = np.mean(np.diag(CC)[probe.good_channels]) * 1e-3 if params.normalisation == 'whitening' else 1e-6
        if params.whiteningRange < np.inf:
            #  if there are too many channels, a finite whiteningRange is more robust to noise
            # in the estimation of the covariance
            whiteningRange = min(params.whiteningRange, Nchan)
            # this function performs the same matrix inversions as below, just on subsets of
            # channels around each channel
            Wrot = whiteningLocal(CC, probe.yc, probe.xc, whiteningRange, epsilon=epsilon)
        else:
            Wrot = whiteningFromCovariance(CC, epsilon=epsilon)
    elif params.normalisation == 'zscore':
        # Do individual channel z-scoring instead of whitening
        Wrot = cp.diag(cp.diag(CC) ** (-0.5))
    elif params.normalisation == 'global_zscore':
        Wrot = cp.eye(CC.shape[0]) * np.median(cp.diag(CC) ** (-0.5))  # same value for all channels
    if qc_path is not None:
        iblsorter.qc.plot_whitening_matrix(Wrot.get(), good_channels=probe.good_channels, out_path=qc_path)
        iblsorter.qc.plot_covariance_matrix(CC.get(), out_path=qc_path)

    Wrot = Wrot * params.scaleproc
    condition_number = np.linalg.cond(cp.asnumpy(Wrot)[probe.good_channels, :][:, probe.good_channels])
    logger.info(f"Computed the whitening matrix cond = {condition_number}.")
    if condition_number > 50:
        logger.warning("high conditioning of the whitening matrix can result in noisy and poor results")
    return Wrot


def get_good_channels(raw_data, params, probe, **kwargs):
    """
    Wrapper for two choices of channel detection: 'raw_correlations' and
    'kilosort'. Passes kwargs through to 
    individual methods.
    """
    method = params.channel_detection_method
    if method == "raw_correlations":
        return get_good_channels_raw_correlations(raw_data, params, probe, **kwargs)
    else:
        return get_good_channels_kilosort(raw_data, params, probe, **kwargs)


def get_good_channels_raw_correlations(raw_data, params, probe, t0s=None, return_labels=False):
    """
    Detect bad channels using the method described in IBL whitepaper
    :param raw_data:
    :param params:
    :param probe:
    :param t0s:
    :return:
    """
    if t0s is None:
        t0s = np.linspace(10, raw_data.shape[0] / params.fs - 10, 25)
    channel_labels = np.zeros((probe.Nchan, t0s.size))
    for icc, t0 in enumerate(tqdm(t0s, desc="Auto-detection of noisy channels")):
        s0 = slice(int(t0 * params.fs), int((t0 + 0.4) * params.fs))
        raw = raw_data[s0][:, :probe.Nchan].T.astype(np.float32) * probe['sample2volt']
        channel_labels[:, icc], _ = detect_bad_channels(raw, params.fs, **params.channel_detection_parameters.dict())
    channel_labels = scipy.stats.mode(channel_labels, axis=1)[0].squeeze()
    logger.info(f"Detected {np.sum(channel_labels == 1)} dead channels")
    logger.info(f"Detected {np.sum(channel_labels == 2)} noise channels")
    logger.info(f"Detected {np.sum(channel_labels == 3)} uppermost channels outside of the brain")
    if np.mean(channel_labels == 0) < 0.5:
        raise RuntimeError("More than half of channels are considered bad. Verify your raw data and eventually update"
                           "the channel_detection_parameters.")
    if return_labels:
        return channel_labels == 0, channel_labels
    else:
        return channel_labels == 0


def get_good_channels_kilosort(raw_data, params, probe, return_labels=False):
    """
    Of the channels indicated by the user as good (chanMap), 
    further subset those that have a mean firing rate above a certain value
    (default is ops.minfr_goodchannels = 0.1Hz)
    Needs the same filtering parameters in ops as usual
    also needs to know where to start processing batches (twind)
    and how many channels there are in total (NchanTOT).
    
    :param raw_data: Raw data loader from the main PyKS loop.
    :param params: Params object for this PyKS run.
    :param probe: Probe identified for this recording.
    :param return_labels: Whether to return channel labels 
        (0 for good, 1 for dead, etc.)
    """
    fs = params.fs
    fshigh = params.fshigh
    fslow = params.fslow
    Nbatch = get_Nbatch(raw_data, params)
    NT = params.NT
    spkTh = params.spkTh
    nt0 = params.nt0
    minfr_goodchannels = params.minfr_goodchannels

    chanMap = probe.chanMap
    NchanTOT = len(chanMap)

    ich = []
    k = 0
    ttime = 0

    # skip every 100 batches
    # TODO: move_to_config - every N batches
    for ibatch in tqdm(range(0, Nbatch, int(ceil(Nbatch / 100))), desc="Finding good channels"):
        i = NT * ibatch
        buff = raw_data[i:i + NT]
        # buff = _make_fortran(buff)
        # NOTE: using C order now
        assert buff.shape[0] > buff.shape[1]
        assert buff.flags.c_contiguous
        if buff.size == 0:
            break

        # Put on GPU.
        buff = cp.asarray(buff, dtype=np.float32)
        assert buff.flags.c_contiguous
        datr = gpufilter(buff, chanMap=chanMap, fs=fs, fshigh=fshigh, fslow=fslow)
        assert datr.shape[0] > datr.shape[1]

        # very basic threshold crossings calculation
        s = cp.std(datr, axis=0)
        datr = datr / s  # standardize each channel ( but don't whiten)
        # TODO: move_to_config (30 sample range)
        mdat = my_min(datr, 30, 0)  # get local minima as min value in +/- 30-sample range

        # take local minima that cross the negative threshold
        xi, xj = cp.nonzero((datr < mdat + 1e-3) & (datr < spkTh))

        # filtering may create transients at beginning or end. Remove those.
        xj = xj[(xi >= nt0) & (xi <= NT - nt0)]

        # collect the channel identities for the detected spikes
        ich.append(xj)
        k += xj.size

        # keep track of total time where we took spikes from
        ttime += datr.shape[0] / fs

    ich = cp.concatenate(ich)

    # count how many spikes each channel got
    nc, _ = cp.histogram(ich, cp.arange(NchanTOT + 1))

    # divide by total time to get firing rate
    nc = nc / ttime

    # keep only those channels above the preset mean firing rate
    igood = cp.asnumpy(nc >= minfr_goodchannels)

    if np.sum(igood) == 0:
        raise RuntimeError("No good channels found! Verify your raw data and parameters.")

    logger.info('Found %d threshold crossings in %2.2f seconds of data.' % (k, ttime))
    logger.info('Found %d/%d bad channels.' % (np.sum(~igood), len(igood)))

    if return_labels:
        labels = (~igood).astype(int) # mark channels as dead (code 1)
        return igood, labels
    else:
        return igood


def get_Nbatch(raw_data, params):
    n_samples = max(raw_data.shape)
    # we assume raw_data as been already virtually split with the requested trange
    return ceil(n_samples / params.NT)  # number of data batches


def destriping(ctx):
    """IBL destriping - multiprocessing CPU version for the time being, although leveraging the GPU
    for the many FFTs performed would probably be quite beneficial """
    probe = ctx.probe
    raw_data = ctx.raw_data
    ir = ctx.intermediate
    wrot = cp.asnumpy(ir.Wrot)
    # get the bad channels
    # detect_bad_channels_cbin
    kwargs = dict(output_file=ir.proc_path, wrot=wrot, nc_out=probe.Nchan, h=probe.h,
                  butter_kwargs={'N': 3, 'Wn': ctx.params.fshigh / ctx.params.fs * 2, 'btype': 'highpass'},
                  output_qc_path=ctx.output_qc_path)
    # _iblqc_ephysSaturation.samples.npy
    logger.info("Pre-processing: applying destriping option to the raw data")
    # there are inconsistencies between the mtscomp reader and the flat binary file reader
    # the flat bin reader as an attribute _paths that allows looping on each chunk
    if isinstance(raw_data.raw_data, list):
        for i, rd in enumerate(raw_data.raw_data):
            if i == (len(raw_data.raw_data) - 1):
                ns2add = ceil(raw_data.n_samples[-1] / ctx.params.NT) * ctx.params.NT - raw_data.n_samples[-1]
            else:
                ns2add = 0
            decompress_destripe_cbin(rd.name, ns2add=ns2add, append=i > 0, **kwargs)
    elif getattr(raw_data.raw_data, '_paths', None):
        nstot = 0
        for i, bin_file in enumerate(raw_data.raw_data._paths):
            ns, _ = raw_data.raw_data._mmaps[i].shape
            nstot += ns
            if i == (len(raw_data.raw_data._paths) - 1):
                ns2add = ceil(ns / ctx.params.NT) * ctx.params.NT - ns
            else:
                ns2add = 0
            decompress_destripe_cbin(bin_file, append=i > 0, ns2add=ns2add, **kwargs)
    else:  # in the case of cbin IBL files
        assert raw_data.raw_data.n_parts == 1
        bin_file = Path(raw_data.raw_data.name)
        ns2add = ceil(raw_data.n_samples / ctx.params.NT) * ctx.params.NT - raw_data.n_samples
        decompress_destripe_cbin(bin_file, ns2add=ns2add, **kwargs)
