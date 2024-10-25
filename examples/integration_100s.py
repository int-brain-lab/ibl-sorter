import shutil
from pathlib import Path

import iblsorter
from iblsorter.ibl import run_spike_sorting_ibl, ibl_pykilosort_params
from viz import reports

INTEGRATION_DATA_PATH = Path("/datadisk/Data/neuropixel/integration_tests/stand-alone")
SCRATCH_DIR = Path.home().joinpath("scratch", 'iblsort')
print(f"Using integration data path: {INTEGRATION_DATA_PATH}")

DELETE = True  # delete the intermediate run products, if False they'll be copied over

override_params = {}
label = ""


def run_integration_test(bin_file):
    """
    The folder architecture is as follows
    /mnt/s1/spikesorting/integration_tests1.6a.dev0/imec_385_100s  output_dir
    ---- pykilosort  ks_output_dir
    ---- alf_path  alf_path

    :param bin_file:
    """
    output_dir = INTEGRATION_DATA_PATH.joinpath(f"{iblsorter.__version__}" + label, bin_file.name.split('.')[0])
    ks_output_dir = output_dir.joinpath('pykilosort')
    alf_path = ks_output_dir.joinpath('alf')

    # this can't be outside of a function, otherwise each multiprocessing job will execute this code!
    shutil.rmtree(SCRATCH_DIR, ignore_errors=True)
    SCRATCH_DIR.mkdir(exist_ok=True)

    ks_output_dir.mkdir(parents=True, exist_ok=True)

    params = ibl_pykilosort_params(bin_file)
    for k in override_params:
        params[k] = override_params[k]

    run_spike_sorting_ibl(bin_file, delete=DELETE, scratch_dir=SCRATCH_DIR, params=params,
                          ks_output_dir=ks_output_dir, alf_path=alf_path, log_level='DEBUG')
    # we copy the temporary files to the output directory if we want to investigate them
    if not DELETE:
        working_directory = SCRATCH_DIR.joinpath('.kilosort', bin_file.stem)
        pre_proc_file = working_directory.joinpath('proc.dat')
        intermediate_directory = ks_output_dir.joinpath('intermediate')
        intermediate_directory.mkdir(exist_ok=True)
        shutil.copy(pre_proc_file, intermediate_directory)

    reports.qc_plots_metrics(bin_file=bin_file, pykilosort_path=alf_path, raster_plot=True, raw_plots=True, summary_stats=False,
                             raster_start=0., raster_len=100., raw_start=50., raw_len=0.15,
                             vmax=0.05, d_bin=5, t_bin=0.001)


if __name__ == "__main__":
    run_integration_test(INTEGRATION_DATA_PATH.joinpath("imec_385_100s.ap.bin"))
