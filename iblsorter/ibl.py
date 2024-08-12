from pathlib import Path
import datetime
import json
import logging
import shutil

import numpy as np

import spikeglx
import neuropixel
from ibllib.ephys import spikes
from one.alf.files import get_session_path
from one.remote import aws
from iblsorter import add_default_handler, run, Bunch, __version__
from iblsorter.params import KilosortParams, MotionEstimationParams


_logger = logging.getLogger(__name__)


def _get_multi_parts_records(bin_file):
    """ Looks for the multiple parts of the recording using sequence files from ibllib"""
    # if multiple files are already provided, do not look for sequence files
    if isinstance(bin_file, list) or isinstance(bin_file, tuple):
        for bf in bin_file:
            if not Path(bf).exists():
                raise FileNotFoundError(bf)
        return bin_file
    # if there is no sequence file attached to the binary file, return just the bin file
    bin_file = Path(bin_file)
    sequence_file = bin_file.parent.joinpath(f"{bin_file.stem.replace('.ap', '.sequence.json')}")
    if not sequence_file.exists():
        if not Path(bin_file).exists():
            raise FileNotFoundError(bin_file)
        else:
            return bin_file
    # if there is a sequence file, return all files if they're all present and this is the first index
    with sequence_file.open() as fid:
        seq = json.load(fid)
    if seq['index'] > 0:
        _logger.warning(f"Multi-part raw ephys: returns empty as this is not the first "
                        f"index in the sequence. Check: {sequence_file}")
        return
    # the common anchor path to look for other meta files is the subject path
    subject_folder = get_session_path(bin_file)
    subject_folder_seq = get_session_path(Path(seq['files'][0])).parents[1]
    # reconstruct path of each binary file, exit with None if one is not found
    cbin_files = []
    for f in seq['files']:
        meta_file = subject_folder.joinpath(Path(f).relative_to(subject_folder_seq))
        cbin_file = next(meta_file.parent.glob(meta_file.stem + '.*bin'), None)
        if cbin_file is None:
            _logger.error(f"Multi-part raw ephys error: missing bin file in folder {meta_file.parent}")
            return
        cbin_files.append(cbin_file)
    return cbin_files


def _sample2v(ap_file):
    md = spikeglx.read_meta_data(ap_file.with_suffix(".meta"))
    s2v = spikeglx._conversion_sample2v_from_meta(md)
    return s2v["ap"][0]


def run_spike_sorting_ibl(bin_file, scratch_dir=None, delete=True,
                          ks_output_dir=None, alf_path=None, log_level='INFO', stop_after=None, 
                          params=None, motion_params=None):
    """
    This runs the spike sorting and outputs the raw pykilosort without ALF conversion
    :param bin_file: binary file full path
    :param scratch_dir: working directory (home of the .kilosort folder) SSD drive preferred.
    :param delete: bool, optional, defaults to True: whether or not to delete the .kilosort temp folder
    :param ks_output_dir: string or Path: output directory defaults to None, in which case it will output in the
     scratch directory.
    :param alf_path: strint or Path, optional: if specified, performs ks to ALF conversion in the specified folder
    :param log_level: string, optional, defaults to 'INFO'
    :return:
    """
    START_TIME = datetime.datetime.now()
    # handles all the paths infrastructure
    assert scratch_dir is not None
    bin_file = _get_multi_parts_records(bin_file)
    scratch_dir.mkdir(exist_ok=True, parents=True)
    ks_output_dir = Path(ks_output_dir) if ks_output_dir is not None else scratch_dir.joinpath('output')
    ks_output_dir.mkdir(exist_ok=True, parents=True)
    log_file = scratch_dir.joinpath(f"_{START_TIME.isoformat()}_kilosort.log")
    add_default_handler(level=log_level)
    add_default_handler(level=log_level, filename=log_file)
    session_scratch_dir = scratch_dir.joinpath('.kilosort', bin_file.stem)
    # construct the probe geometry information
    if params is None:
        params = ibl_pykilosort_params(bin_file)
    if motion_params is None:
        # neuropixels 1/2 Dredge configs
        motion_params = MotionEstimationParams(
            bin_s=params.NT / params.fs,
            gaussian_smoothing_sigma_s=params.NT / params.fs,
            mincorr=0.5
        )
    try:
        _logger.info(f"Starting Pykilosort version {__version__}")
        _logger.info(f"Scratch dir {scratch_dir}")
        _logger.info(f"Output dir {ks_output_dir}")
        _logger.info(f"Data dir {bin_file.parent}")
        _logger.info(f"Log file {log_file}")
        _logger.info(f"Loaded probe geometry for NP{params['probe']['neuropixel_version']}")

        run(dat_path=bin_file, dir_path=scratch_dir, output_dir=ks_output_dir, stop_after=stop_after, **params)
    except Exception as e:
        _logger.exception("Error in the main loop")
        raise e
    [_logger.removeHandler(h) for h in _logger.handlers]
    # move the log file and all qcs to the output folder
    shutil.move(log_file, ks_output_dir.joinpath('spike_sorting_pykilosort.log'))

    # convert the pykilosort output to ALF IBL format
    if alf_path is not None:
        s2v = _sample2v(bin_file)
        alf_path.mkdir(exist_ok=True, parents=True)
        spikes.ks2_to_alf(ks_output_dir, bin_file, alf_path, ampfactor=s2v)
        # move all of the QC outputs to the alf folder as well
        for qc_file in scratch_dir.rglob('_iblqc_*.png'):
            shutil.move(qc_file, alf_path.joinpath(qc_file.name))
        for qc_file in scratch_dir.rglob('_iblqc_*.png'):
            shutil.move(qc_file, alf_path.joinpath(qc_file.name))
    # in production, we remove all of the temporary files after the run
    if delete:
        shutil.rmtree(scratch_dir.joinpath(".kilosort"), ignore_errors=True)


def ibl_pykilosort_params(bin_file):
    params = KilosortParams()
    params.channel_detection_method = 'raw_correlations'
    params.overlap_samples = 1024  # this needs to be a multiple of 1024
    params.probe = probe_geometry(bin_file)
    return dict(params)


def probe_geometry(bin_file):
    """
    Loads the geometry from the meta-data file of the spikeglx acquisition system
    sr: ibllib.io.spikeglx.Reader or integer with neuropixel version 1 or 2
    """
    if isinstance(bin_file, list):
        sr = spikeglx.Reader(bin_file[0])
        h, ver, s2v = (sr.geometry, sr.major_version, sr.sample2volts[0])
    elif isinstance(bin_file, str) or isinstance(bin_file, Path):
        sr = spikeglx.Reader(bin_file)
        h, ver, s2v = (sr.geometry, sr.major_version, sr.sample2volts[0])
    else:
        print(bin_file)
        assert(bin_file == 1 or bin_file == 2)
        h, ver, s2v = (neuropixel.trace_header(version=bin_file), bin_file, 2.34375e-06)
    nc = h['x'].size
    probe = Bunch()
    probe.NchanTOT = nc + 1
    probe.chanMap = np.arange(nc)
    probe.xc = h['x'] + h['shank'] * 200
    probe.yc = h['y']
    probe.x = h['x']
    probe.y = h['y']
    probe.shank = h['shank']
    probe.kcoords = np.zeros(nc)
    probe.channel_labels = np.zeros(nc, dtype=int)
    probe.sample_shift = h['sample_shift']
    probe.h, probe.neuropixel_version, probe.sample2volt = (h, ver, s2v)
    return probe


def download_test_data(local_folder):
    return aws.s3_download_folder('spikesorting/integration_tests', local_folder)


def download_benchmark_data(local_folder):
    return aws.s3_download_folder('spikesorting/benchmarks', local_folder)
