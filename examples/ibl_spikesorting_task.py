"""

python ./examples/ibl_spikesorting_task.py /mnt/s1/spikesorting/raw_data/mainenlab/Subjects/ZM_2240/2020-01-23/001/raw_ephys_data/probe00/
"""
import argparse
from pathlib import Path
from ibllib.pipes.ephys_tasks import SpikeSorting

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path_probe")
    args = parser.parse_args()
    pname = Path(args.path_probe).parts[-1]
    session_path = Path(args.path_probe).parents[1]
    print(f"pykilosort run for probe {pname} in session {session_path}")
    ssjob = SpikeSorting(session_path, one=None, pname=pname, device_collection='raw_ephys_data', location="local")
    ssjob.run()
    assert ssjob.status == 0
