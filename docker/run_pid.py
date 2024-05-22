"""

python ./examples/ibl_spikesorting_task.py /mnt/s1/spikesorting/raw_data/mainenlab/Subjects/ZM_2240/2020-01-23/001/raw_ephys_data/probe00/
"""
import argparse
from pathlib import Path
from ibllib.pipes.ephys_tasks import SpikeSorting

from pathlib import Path

import numpy as np
from one.api import ONE
from brainbox.io.one import EphysSessionLoader, SpikeSortingLoader
from iblatlas.atlas import BrainRegions
br = BrainRegions()

one = ONE()  # cache_dir = "/mnt/s1/spikesorting/raw_data"


# ssg4-a
pids = [
    'ca073754-be17-43b7-a38a-0c1e5563ff32',  # incomplete crashed after alf
    # 'cab81176-36cd-4c37-9353-841e85027d36',  # complete-rsynced
    # 'ccb501d1-a4fa-41c6-819e-54aaf74d439d',  # complete-rsynced
    'ce397420-3cd2-4a55-8fd1-5e28321981f4',
    '5d570bf6-a4c6-4bf1-a14b-2c878c84ef0e',  # error time out sdsc
    '63679ae2-6c87-495b-b888-a7273e545636',  # error time out sdsc
    '68f06c5f-8566-4a4f-a4b1-ab8398724913',  # error time out sdsc
]

# hausser
pids = [
    # '176b4fe3-f570-4d9f-9e25-a5d218f75c8b',  # error OOM cuda
    'ac839451-05bc-493e-b167-558b2b195baa',  # on disk
    '0c42a3d2-beb7-4b7b-a9b5-1eea6c51e1f4',  # on disk
]


pids = [  # errors to download and fix on parede
    '176b4fe3-f570-4d9f-9e25-a5d218f75c8b',  # error OOM cuda
    '4836a465-c691-4852-a0b1-dcd2b1ce38a1',  # error datashift
    'd0046384-16ea-4f69-bae9-165e8d0aeacf',  # error datashift
    '62a97aee-9f8b-40be-9ea7-f785ede30df8',  # error datashift
    'ee3345e6-540d-4cea-9e4a-7f1b2fb9a4e4',   # CUDA Runtime error
    '6a7544a8-d3d4-44a1-a9c6-7f4d460feaac',  # error datashift NP2
    '703f2cd2-377a-48ec-979a-135863169671',  # error datashift NP2
    'bf746764-78e0-4b9c-bdf6-65b6ff45745d',  # error datashift NP2
    'fb962a77-ed5a-40e0-bfdf-b3220427c55e',  # error datashift NP2
]

pids = [  # parede
    # '38124fca-a0ac-4b58-8e8b-84a2357850e6',  # complete rsynced
    '7c94d733-b913-4064-83f2-37422712204c',  # on disk
    'aec2b14f-5dbc-400b-bf2e-dd13e711e2ff',  # on disk
]

for pid in pids:
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
            # fid.write(f"rm -fR {session_path.joinpath('spike_sorters', 'pykilosort', pname)}\n")
            # fid.write(f"rm -fR {ss_path}\n")
    else:
        ss_path = ssl.session_path.joinpath('alf', ssl.pname, 'pykilosort')
        with open(one.cache_dir.joinpath(f'error_{pid}.txt'), 'w+') as fid:
            fid.write(f"{ss_path}\n")

