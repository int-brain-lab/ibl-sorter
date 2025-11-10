from pathlib import Path
import datetime
import importlib.metadata
import sys
import json
import logging
import shutil
import yaml

import numpy as np

import spikeglx
import neuropixel
from ibllib.ephys import spikes
from iblutil.util import log_to_file
import one.alf.io as alfio
from one.alf.path import get_session_path
from one.remote import aws

from iblsorter import run, Bunch, __version__
from iblsorter.params import KilosortParams, MotionEstimationParams

from ibldsp.waveform_extraction import extract_wfs_cbin

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
                          ks_output_dir=None, alf_path=None, stop_after=None,
                          params=None, motion_params=None, extract_waveforms=False):
    """
    Run spike sorting using iblsorter with IBL-specific configurations.

    This function performs spike sorting on electrophysiology data using the pykilosort 2.5 algorithm
    with International Brain Laboratory (IBL) pre-processing and configurations. It handles multi-part
    recordings, manages temporary files, and can optionally convert the output to ALF format
    and extract spike waveforms.

    Parameters
    ----------
    bin_file : str, Path, list, or tuple
        Path to the binary file containing electrophysiology data, or a list/tuple of paths
        for multi-part recordings.
    scratch_dir : Path or str
        Working directory for temporary files during processing. SSD drive preferred for
        performance. Required parameter.
    delete : bool, optional
        Whether to delete the temporary .kilosort folder after processing. Defaults to True.
    ks_output_dir : str or Path, optional
        Directory where the raw pykilosort output will be saved. If None (default),
        a subdirectory 'output' will be created in the scratch directory.
    alf_path : str or Path, optional
        If specified, the pykilosort output will be converted to ALF format and saved
        in this directory. If None (default), no ALF conversion is performed.
    stop_after : str, optional
        Name of processing stage after which to stop the pipeline. If None (default),
        the full pipeline is executed.
    params : KilosortParams, optional
        Custom parameters for the pykilosort algorithm. If None (default),
        default IBL parameters will be used.
    motion_params : MotionEstimationParams, optional
        Custom parameters for motion estimation. If None (default),
        default IBL motion parameters will be used.
    extract_waveforms : bool, optional
        Whether to extract spike waveforms after sorting. Only applies if alf_path
        is specified. Defaults to False.

    Returns
    -------
    None
    """
    assert scratch_dir is not None
    START_TIME = datetime.datetime.now()
    log_file = scratch_dir.joinpath(f"_{START_TIME.isoformat().replace(':', '')}_iblsorter.log")
    _logger = log_to_file('iblsorter', filename=log_file)
    # handles all the path
    bin_file = _get_multi_parts_records(bin_file)
    scratch_dir.mkdir(exist_ok=True, parents=True)
    ks_output_dir = Path(ks_output_dir) if ks_output_dir is not None else scratch_dir.joinpath('output')
    ks_output_dir.mkdir(exist_ok=True, parents=True)
    # construct the probe geometry information
    if params is None:
        params = ibl_pykilosort_params(bin_file)
    try:
        _logger.info(f"Starting iblsorter version {__version__}")
        _logger.info(f"Scratch dir {scratch_dir}")
        _logger.info(f"Output dir {ks_output_dir}")
        _logger.info(f"Data dir {bin_file.parent}")
        _logger.info(f"Log file {log_file}")
        _logger.info("Python executable: %s", sys.executable)
        _logger.info("Python version: %s", sys.version)
        packages = sorted([f"{dist.metadata['name']}=={dist.version}"
                           for dist in importlib.metadata.distributions()])
        _logger.info("Installed packages:\n---\n%s\n---", '\n'.join(packages))


        _logger.info(f"Loaded probe geometry for NP{params.probe.neuropixel_version}")


        run(dat_path=bin_file, dir_path=scratch_dir, output_dir=ks_output_dir,
            stop_after=stop_after, motion_params=motion_params, **dict(params))
    except Exception as e:
        _logger.exception("Error in the main loop")
        raise e
    # removes the file handler of the logger
    for h in _logger.handlers:
        if isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_file:
            _logger.removeHandler(h)
    # move the log file and all qcs to the output folder
    shutil.move(log_file, ks_output_dir.joinpath('_ibl_log.info_iblsorter.log'))
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
    # now extract the waveforms if required
    if extract_waveforms and alf_path is not None:
        extract_waveforms_after_sorting(bin_file, alf_path, scratch_dir)


def ibl_pykilosort_params(bin_file=None, params=None, npx_version=1):
    if bin_file and (param_file := bin_file.parent.joinpath('iblsorter_parameters.yaml')).exists():
        # read in the param_file with yaml
        with open(param_file, 'r') as fp:
            params = yaml.safe_load(fp) | (params or {})
    else:
        params = params or {}
    params = KilosortParams(**params)
    params.overlap_samples = 1024  # this needs to be a multiple of 1024
    params.probe = probe_geometry(bin_file, npx_version=npx_version)
    return params


def probe_geometry(bin_file=None, npx_version=1):
    """
    Loads the geometry from the meta-data file of the spikeglx acquisition system
    sr: ibllib.io.spikeglx.Reader or integer with neuropixel version 1 or 2
    """
    if bin_file is not None:
        sr = spikeglx.Reader(bin_file)
        h, ver, s2v = (sr.geometry, sr.major_version, sr.sample2volts[0])
    else:
        h, ver, s2v = (neuropixel.trace_header(version=npx_version), npx_version, 2.34375e-06)
    nc = h['x'].size
    probe = Bunch()
    probe.NchanTOT = nc + 1
    probe.chanMap = h['ind']
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


def extract_waveforms_after_sorting(bin_file, alf_dir, scratch_dir):
    """
    Extract spike waveforms from a binary file using spike sorting results.

    This function loads spike sorting results from ALF files, extracts waveforms
    from the binary recording file at the detected spike times, and saves the
    extracted waveforms in ALF format.

    Parameters
    ----------
    bin_file : str or Path
        Path to the binary recording file from which to extract waveforms.
    alf_dir : str or Path
        Directory containing ALF files with spike sorting results and where
        the extracted waveforms will be saved.
    scratch_dir : str or Path
        Directory for temporary files during waveform extraction.

    Returns
    -------
    list
        List of paths to the output waveform files that were created.
    """
    spikes = alfio.load_object(alf_dir, 'spikes', attribute=['samples', 'clusters'])
    clusters = alfio.load_object(alf_dir, 'clusters', attribute=['channels'])
    channels = alfio.load_object(alf_dir, 'channels')
    _output_waveform_files = extract_wfs_cbin(
        bin_file=bin_file,
        output_dir=alf_dir,
        spike_samples=spikes['samples'],
        spike_clusters=spikes['clusters'],
        spike_channels=clusters['channels'][spikes['clusters']],
        channel_labels=channels['labels'],
        max_wf=256,
        trough_offset=42,
        spike_length_samples=128,
        chunksize_samples=int(30_000),
        n_jobs=None,
        preprocess_steps=["phase_shift", "bad_channel_interpolation", "butterworth", "car"],
        scratch_dir=scratch_dir,
    )
    return _output_waveform_files


def download_test_data(local_folder):
    return aws.s3_download_folder('spikesorting/integration_tests/ibl', local_folder)


def download_benchmark_data(local_folder):
    return aws.s3_download_folder('spikesorting/benchmarks', local_folder)
