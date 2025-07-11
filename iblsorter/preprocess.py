import logging
from math import ceil
from pathlib import Path

import numpy as np
import scipy.stats
import cupy as cp
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import iblsorter.qc
from ibldsp.voltage import decompress_destripe_cbin, destripe, detect_bad_channels
import ibldsp.plots

logger = logging.getLogger(__name__)

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
        raw = raw_data[s0][:, :probe.Nchan].T.astype(np.float32)
        datr = destripe(raw, fs=params.fs, h=probe, channel_labels=probe.channel_labels,
                        butter_kwargs=get_butter_kwargs(params))  / probe['sample2volt']
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


def get_good_channels(raw_data, params, probe, t0s=None, return_labels=False, qc_path=None):
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
        raw = raw_data[s0][:, :probe.Nchan].T.astype(np.float32)
        channel_labels[:, icc], xfeats = detect_bad_channels(raw, params.fs, **params.channel_detection_parameters.dict())
    channel_labels = scipy.stats.mode(channel_labels, axis=1)[0].squeeze()
    logger.info(f"Detected {np.sum(channel_labels == 1)} dead channels")
    logger.info(f"Detected {np.sum(channel_labels == 2)} noise channels")
    logger.info(f"Detected {np.sum(channel_labels == 3)} uppermost channels outside of the brain")
    if qc_path is not None:
        fig, ax = ibldsp.plots.show_channels_labels(raw, params.fs, channel_labels, xfeats, **params.channel_detection_parameters.dict())
        fig.savefig(Path(qc_path).joinpath('_iblqc_channel_detection.png'))
        plt.close(fig)
    if np.mean(channel_labels == 0) < 0.5:
        raise RuntimeError(f"More than half of channels are considered bad. Verify your raw data and eventually update"
                           f"the channel_detection_parameters. \n"
                           f"Check {Path(qc_path).joinpath('_iblqc_covariance_matrix.png')}"
                           f" and your data using `ibldsp.voltage.detect_bad_channels`")
    if return_labels:
        return channel_labels == 0, channel_labels
    else:
        return channel_labels == 0


def get_Nbatch(raw_data, params):
    n_samples = max(raw_data.shape)
    # we assume raw_data as been already virtually split with the requested trange
    return ceil(n_samples / params.NT)  # number of data batches


def get_butter_kwargs(params):
    fs, fshigh, fslow = params.fs, params.fshigh, params.fslow
    if fslow and fslow < fs / 2:
        assert fslow > fshigh
        # butterworth filter with only 3 nodes (otherwise it's unstable for float32)
        return dict(N=3, Wn=(fshigh, fslow), fs=fs, btype='bandpass')
    else:
        # butterworth filter with only 3 nodes (otherwise it's unstable for float32)
        return dict(N=3, Wn=fshigh, fs=fs, btype='highpass')


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
                  butter_kwargs=get_butter_kwargs(ctx.params),
                  output_qc_path=ctx.output_qc_path,
                  reject_channels=ctx.probe.channel_labels
                  )
    # _iblqc_ephysSaturation.samples.npy
    logger.info("Pre-processing: applying destriping option to the raw data")
    ns2add = ceil(raw_data.ns / ctx.params.NT) * ctx.params.NT - raw_data.ns
    decompress_destripe_cbin(raw_data.file_bin, ns2add=ns2add, **kwargs)
