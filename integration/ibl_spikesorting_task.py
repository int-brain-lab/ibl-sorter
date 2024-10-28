"""
This is an integration test on the IBL task
"""
import datetime
import logging
import shutil
from pathlib import Path
from ibllib.pipes.ephys_tasks import SpikeSorting
from iblsorter.ibl import download_test_data

logger = logging.getLogger('iblsorter')

SCRATCH_FOLDER = Path('/home/olivier/scratch')
PATH_INTEGRATION = Path("/mnt/s1/spikesorting/integration_tests")


if __name__ == "__main__":

    PATH_INTEGRATION.joinpath(f"{datetime.datetime.now().isoformat().replace(':', '')}.temp").touch()
    download_test_data(PATH_INTEGRATION.joinpath('ibl'))
    path_probe = PATH_INTEGRATION.joinpath("ibl", "probe01")
    testing_path = PATH_INTEGRATION.joinpath("testing_output")
    shutil.rmtree(testing_path, ignore_errors=True)
    pname = path_probe.parts[-1]
    session_path = testing_path.joinpath('iblsort', 'Subjects', 'iblsort_subject', '2024-07-16', '001')
    raw_ephys_data_path = session_path.joinpath('raw_ephys_data', pname)
    logger.info('copying raw_ephys_data to session path')
    shutil.rmtree(raw_ephys_data_path, ignore_errors=True)
    raw_ephys_data_path.mkdir(parents=True, exist_ok=True)
    shutil.copytree(path_probe, raw_ephys_data_path, dirs_exist_ok=True)
    logger.info(f"iblsort run for probe {pname} in session {session_path}")
    ssjob = SpikeSorting(session_path, one=None, pname=pname, device_collection='raw_ephys_data', location="local", scratch_folder=SCRATCH_FOLDER)
    ssjob.run()
    assert ssjob.status == 0
    ssjob.assert_expected_outputs()
    logger.info("Test is complete and outputs validated - exiting now")
