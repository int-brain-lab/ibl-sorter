import shutil
from pathlib import Path

import pykilosort
from pykilosort.ibl import run_spike_sorting_ibl, ibl_pykilosort_params, download_test_data

INTEGRATION_DATA_PATH = Path("/datadisk/Data/neuropixel/spike_sorting/integration_test")
# INTEGRATION_DATA_PATH = Path("/mnt/s1/spikesorting/integration_tests")

SCRATCH_DIR = Path.home().joinpath("scratch", 'pykilosort')
shutil.rmtree(SCRATCH_DIR, ignore_errors=True)
SCRATCH_DIR.mkdir(exist_ok=True)
DELETE = True  # delete the intermediate run products, if False they'll be copied over
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
    bin_file = list(INTEGRATION_DATA_PATH.rglob("imec_385_100s.ap.bin"))
    bin_file.sort()
    # _make_compressed_parts(bin_file)
    ks_output_dir = INTEGRATION_DATA_PATH.joinpath(
        f"{pykilosort.__version__}" + label, bin_file[0].name.split('.')[0] + 'multi_parts')
else:
    bin_file = INTEGRATION_DATA_PATH.joinpath("imec_385_100s.ap.bin")
    ks_output_dir = INTEGRATION_DATA_PATH.joinpath(f"{pykilosort.__version__}" + label, bin_file.name.split('.')[0])


ks_output_dir.mkdir(parents=True, exist_ok=True)
alf_path = ks_output_dir.joinpath('alf')

params = ibl_pykilosort_params(bin_file)
for k in override_params:
    params[k] = override_params[k]

run_spike_sorting_ibl(bin_file, delete=DELETE, scratch_dir=SCRATCH_DIR, params=params,
                      ks_output_dir=ks_output_dir, alf_path=alf_path, log_level='DEBUG')

if DELETE == False:
    working_directory = SCRATCH_DIR.joinpath('.kilosort', bin_file.stem)
    pre_proc_file = working_directory.joinpath('proc.dat')
    intermediate_directory = ks_output_dir.joinpath('intermediate')
    intermediate_directory.mkdir(exist_ok=True)
    shutil.copy(pre_proc_file, intermediate_directory)


from viz import reports
reports.qc_plots_metrics(bin_file=bin_file, pykilosort_path=alf_path, raster_plot=True, raw_plots=True, summary_stats=False,
                         raster_start=0., raster_len=100., raw_start=50., raw_len=0.15,
                         vmax=0.05, d_bin=5, t_bin=0.001)
# this is how to test against the current iblscript methods
# print(f'~/Documents/PYTHON/iblscripts/deploy/serverpc/kilosort2/run_pykilosort.py {INTEGRATION_DATA_PATH.joinpath("imec_385_100s.ap.cbin")} {SCRATCH_DIR}')
# ~/Documents/PYTHON/iblscripts/deploy/serverpc/kilosort2/run_pykilosort.py /datadisk/Data/spike_sorting/pykilosort_tests/imec_385_100s.ap.cbin /home/olivier/scratch/pykilosort
