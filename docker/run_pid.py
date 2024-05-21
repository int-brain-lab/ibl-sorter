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

# hausser
pids = [
    '176b4fe3-f570-4d9f-9e25-a5d218f75c8b',  #  running
    'ac839451-05bc-493e-b167-558b2b195baa',  # download hausser
    '0c42a3d2-beb7-4b7b-a9b5-1eea6c51e1f4',  # download hausser
]

# parede
pids = [
    # '3fded122-619c-4e65-aadd-d5420978d167',  # complete
    '4836a465-c691-4852-a0b1-dcd2b1ce38a1',
    'd0046384-16ea-4f69-bae9-165e8d0aeacf',
    '38124fca-a0ac-4b58-8e8b-84a2357850e6',  # download
    '62a97aee-9f8b-40be-9ea7-f785ede30df8',  # download
    '7c94d733-b913-4064-83f2-37422712204c',  # download
    'aec2b14f-5dbc-400b-bf2e-dd13e711e2ff',  # download
]

# ssp2
pids = [
'ca073754-be17-43b7-a38a-0c1e5563ff32',
'cab81176-36cd-4c37-9353-841e85027d36',
'ccb501d1-a4fa-41c6-819e-54aaf74d439d',
'ce397420-3cd2-4a55-8fd1-5e28321981f4',
]


for pid in pids:
    ssl = SpikeSortingLoader(one=one, pid=pid)
    sr = ssl.raw_electrophysiology(band="ap", stream=False)
    ssl.samples2times(0)
    print(f"pykilosort run for pid {pid} probe {ssl.pname} in session {ssl.session_path}")
    ssjob = SpikeSorting(ssl.session_path, one=one, pname=ssl.pname, device_collection='raw_ephys_data', location="local")
    ssjob.run()
    assert ssjob.status == 0
    sr.file_bin.unlink()

