"""
This is an integration test on the IBL task
"""
import datetime
import logging
import shutil
from pathlib import Path

from iblutil.util import setup_logger
from ibllib.pipes.ephys_tasks import SpikeSorting

from viz import reports
import iblsorter
from iblsorter.params import load_integration_config

logger = setup_logger('iblsorter', level='DEBUG')
config = load_integration_config()
override_params = {}

if __name__ == "__main__":
    path_probe = config.integration_data_path.joinpath("testing_input", "ibl_spikesorting_task")
    output_dir = config.integration_data_path.joinpath(
        'testing_output', 'ibl_spikesorting_task', f"{iblsorter.__version__}")
    shutil.rmtree(output_dir, ignore_errors=True)
    session_path = output_dir.joinpath('Subjects', 'algernon', '2024-07-16', '001')
    raw_ephys_data_path = session_path.joinpath('raw_ephys_data', 'probe01')
    logger.info('copying raw_ephys_data to session path - removing existing folder if it exists')
    shutil.rmtree(raw_ephys_data_path, ignore_errors=True)
    raw_ephys_data_path.mkdir(parents=True, exist_ok=True)
    shutil.copytree(path_probe, raw_ephys_data_path, dirs_exist_ok=True)
    logger.info(f"iblsort run for probe01 in session {session_path}")
    ssjob = SpikeSorting(session_path, one=None, pname='probe01', device_collection='raw_ephys_data', location="local", scratch_folder=config.scratch_dir)
    ssjob.run()
    assert ssjob.status == 0
    ssjob.assert_expected_outputs()
    logger.info("Outputs are validated. Compute report")
    alf_path = session_path.joinpath('alf', 'probe01', 'iblsorter')
    reports.qc_plots_metrics(bin_file=next(raw_ephys_data_path.glob('*.ap.cbin')), pykilosort_path=alf_path, out_path=output_dir, raster_plot=True,
                             raw_plots=True, summary_stats=False, raster_start=0., raster_len=100., raw_start=50., raw_len=0.15,
                             vmax=0.05, d_bin=5, t_bin=0.001)
    logger.info("Remove raw data copy")
    shutil.rmtree(raw_ephys_data_path, ignore_errors=True)
    logger.info(f"Exiting now, test data results in {output_dir}")
