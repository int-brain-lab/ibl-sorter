"""
This is an integration test on the IBL task
"""
from pathlib import Path
import shutil
from ibllib.pipes.ephys_tasks import SpikeSorting
from iblutil.util import setup_logger
import iblsorter

# /mnt/s1/spikesorting/integration_tests/
path_probe = Path("/datadisk/Data/neuropixel/integration_tests/inputs/probe01")
testing_path = Path("/datadisk/Data/neuropixel/integration_tests/testing_output")
logger = setup_logger(level="DEBUG")

if __name__ == "__main__":
    pname = path_probe.parts[-1]
    session_path = testing_path.joinpath('iblsort', 'subjects', '2024-07-16', '001')
    raw_ephys_data_path = session_path.joinpath('raw_ephys_data', pname)
    logger.info('copying raw_ephys_data to session path')
    raw_ephys_data_path.mkdir(parents=True, exist_ok=True)
    shutil.copytree(path_probe, raw_ephys_data_path, dirs_exist_ok=True)
    logger.info(f"iblsort run for probe {pname} in session {session_path}")
    ssjob = SpikeSorting(session_path, one=None, pname=pname, device_collection='raw_ephys_data', location="local")
    ssjob.run()
    assert ssjob.status == 0
