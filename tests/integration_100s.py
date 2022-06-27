import shutil
from pathlib import Path
import numpy as np

import pykilosort
from pykilosort.ibl import run_spike_sorting_ibl, ibl_pykilosort_params

INTEGRATION_DATA_PATH = Path("/datadisk/Data/spike_sorting/pykilosort_tests")
SCRATCH_DIR = Path.home().joinpath("scratch", 'pykilosort')
shutil.rmtree(SCRATCH_DIR, ignore_errors=True)
SCRATCH_DIR.mkdir(exist_ok=True)
DELETE = False  # delete the intermediate run products, if False they'll be copied over
# bin_file = INTEGRATION_DATA_PATH.joinpath("imec_385_100s.ap.bin")
#
# label = "zscore"
# override_params = dict(do_whitening=False)
override_params = {}
label = ""
# params['preprocessing_function'] = 'kilosort2'
cluster_times_path = INTEGRATION_DATA_PATH.joinpath("cluster_times")

MULTIPARTS = False
if MULTIPARTS:
    bin_file = list(INTEGRATION_DATA_PATH.rglob("imec_385_100s.ap.cbin"))
    bin_file.sort()
    # _make_compressed_parts(bin_file)
    ks_output_dir = INTEGRATION_DATA_PATH.joinpath(
        f"{pykilosort.__version__}" + label, bin_file[0].name.split('.')[0] + 'multi_parts')
else:
    bin_file = INTEGRATION_DATA_PATH.joinpath("imec_385_100s.ap.cbin")
    ks_output_dir = INTEGRATION_DATA_PATH.joinpath(f"{pykilosort.__version__}" + label, bin_file.name.split('.')[0])


ks_output_dir.mkdir(parents=True, exist_ok=True)
alf_path = ks_output_dir.joinpath('alf')

params = ibl_pykilosort_params(bin_file)
for k in override_params:
    params[k] = override_params[k]

run_spike_sorting_ibl(bin_file, delete=DELETE, scratch_dir=SCRATCH_DIR, params=params,
                      ks_output_dir=ks_output_dir, alf_path=alf_path, log_level='DEBUG')

if DELETE == False:
    import shutil
    working_directory = SCRATCH_DIR.joinpath('.kilosort', bin_file.name)
    pre_proc_file = working_directory.joinpath('proc.dat')
    intermediate_directory = ks_output_dir.joinpath('intermediate')
    intermediate_directory.mkdir(exist_ok=True)
    shutil.copy(pre_proc_file, intermediate_directory)

# this is how to test against the current iblscript methods
# print(f'~/Documents/PYTHON/iblscripts/deploy/serverpc/kilosort2/run_pykilosort.py {INTEGRATION_DATA_PATH.joinpath("imec_385_100s.ap.cbin")} {SCRATCH_DIR}')
# ~/Documents/PYTHON/iblscripts/deploy/serverpc/kilosort2/run_pykilosort.py /datadisk/Data/spike_sorting/pykilosort_tests/imec_385_100s.ap.cbin /home/olivier/scratch/pykilosort