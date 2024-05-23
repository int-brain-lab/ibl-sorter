"""
Usage from outside docker:

sudo docker compose up -d
docker compose exec spikesorter python /root/Documents/PYTHON/pykilosort/docker/run_pid.py aec2b14f-5dbc-400b-bf2e-dd13e711e2ff
sudo docker compose down

It is safer to stop / start the docker container for each run to flush the GPU memory on some machines
"""

import argparse
from ibllib.pipes.ephys_tasks import SpikeSorting

from pathlib import Path
from one.api import ONE
from brainbox.io.one import EphysSessionLoader, SpikeSortingLoader
from iblatlas.atlas import BrainRegions
br = BrainRegions()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pid")
    args = parser.parse_args()
    pid = args.pid
    one = ONE()  # cache_dir = "/mnt/s1/spikesorting/raw_data" or "/mnt/s0/spikesorting"
    ssl = SpikeSortingLoader(one=one, pid=pid)
    sr = ssl.raw_electrophysiology(band="ap", stream=False)
    ssl.samples2times(0)
    print(f"pykilosort run for pid {pid} probe {ssl.pname} in session {ssl.session_path}")
    ssjob = SpikeSorting(ssl.session_path, one=one, pname=ssl.pname, device_collection='raw_ephys_data', location="local")
    ssjob.run()
    if ssjob.status == 0:
        sr.file_bin.unlink()
        ss_path = ssl.session_path.joinpath('alf', ssl.pname, 'pykilosort')
        target_dir_sdsc = Path("/mnt/ibl/quarantine/tasks_external/SpikeSorting")
        command = f"rsync -av --progress --relative {one.cache_dir}/./{ss_path.relative_to(one.cache_dir)}/ sdsc:{target_dir_sdsc}/"
        with open(one.cache_dir.joinpath(f'rsync_{pid}.txt'), 'w+') as fid:
            fid.write(f"{command}\n")
            fid.write(f"#rm -fR {ssl.session_path.joinpath('spike_sorters', 'pykilosort', ssl.pname)}\n")
            fid.write(f"#rm -fR {ss_path}\n")
    else:
        ss_path = ssl.session_path.joinpath('alf', ssl.pname, 'pykilosort')
        with open(one.cache_dir.joinpath(f'error_{pid}.txt'), 'w+') as fid:
            fid.write(f"{ss_path}\n")
