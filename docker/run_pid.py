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
from viewephys.gui import viewephys
from iblatlas.atlas import BrainRegions
br = BrainRegions()
one = ONE(base_url="https://alyx.internationalbrainlab.org", cache_dir="/mnt/s1/spikesorting/raw_data")


pid = '3fded122-619c-4e65-aadd-d5420978d167'

ssl = SpikeSortingLoader(one=one, pid=pid)
sr = ssl.raw_electrophysiology(band="ap", stream=False)
ssl.samples2times(0)
print(f"pykilosort run for pid {pid} probe {ssl.pname} in session {ssl.session_path}")
ssjob = SpikeSorting(ssl.session_path, one=one, pname=ssl.pname, device_collection='raw_ephys_data', location="local")
ssjob.run()
assert ssjob.status == 0
