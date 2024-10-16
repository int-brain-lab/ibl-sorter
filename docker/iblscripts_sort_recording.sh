#!/bin/bash
set -e

SCRATCH_DRIVE=/mnt/h0  # this is the path of the scratch SSD volume for intermediate KS2 results swapping
# --------------- DO NOT EDIT BELOW

source ~/Documents/PYTHON/SPIKE_SORTING/iblsort/.venv/bin/activate
which python  # checks the version for easier debugging
python ~/Documents/PYTHON/iblscripts/deploy/serverpc/iblsorter/run_iblsorter.py $1 $2
