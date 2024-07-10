"""
This is an integration test on the IBL task
"""
import argparse
import datetime
from pathlib import Path
from ibllib.pipes.ephys_tasks import SpikeSorting

path_probe = Path("/mnt/s1/spikesorting/integration_tests/ibl/probe00")
testing_path = Path("/mnt/s1/spikesorting/integration_tests/ibl/subjects")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path_probe")
    args = parser.parse_args()
    path_probe = Path(args.path_probe)
    pname = path_probe.parts[-1]

    testing_path.joinpath('iblsort', )
    session_path = Path(path_probe).parents[1]
    print(f"iblsort run for probe {pname} in session {session_path}")
    ssjob = SpikeSorting(session_path, one=None, pname=pname, device_collection='raw_ephys_data', location="local")
    ssjob.run()
    assert ssjob.status == 0

