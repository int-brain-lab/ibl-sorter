import shutil
from pathlib import Path

import iblsorter
from iblsorter.ibl import run_spike_sorting_ibl, ibl_pykilosort_params
from viz import reports

SCRATCH_DIR = Path('/home/olivier/scratch')
FILE_RECORDING = '/datadisk/Data/neuropixel/integration_tests/quarter_density/Subjects/KM_012/2024-03-05/002/raw_ephys_data/probe00/_spikeglx_ephysData_g0_t0.imec0.ap.bin'
OUTPUT_DIR = '/datadisk/Data/neuropixel/integration_tests/quarter_density'

override_params = {}  # here it is possible to set some parameters for the run


def spike_sort_recording(bin_file, output_dir):
    """
    The folder architecture is as follows
    ---- pykilosort  ks_output_dir
    ---- alf_path  alf_path

    :param bin_file:
    """
    bin_file = Path(bin_file)
    output_dir = Path(output_dir)

    ks_output_dir = output_dir.joinpath('pykilosort')
    alf_path = ks_output_dir.joinpath('alf')

    # this can't be outside of a function, otherwise each multiprocessing job will execute this code!
    shutil.rmtree(SCRATCH_DIR, ignore_errors=True)
    SCRATCH_DIR.mkdir(exist_ok=True)

    ks_output_dir.mkdir(parents=True, exist_ok=True)

    params = ibl_pykilosort_params(bin_file)
    for k in override_params:
        params[k] = override_params[k]

    run_spike_sorting_ibl(bin_file, scratch_dir=SCRATCH_DIR, params=params,
                          ks_output_dir=ks_output_dir, alf_path=alf_path, log_level='INFO')

    reports.qc_plots_metrics(bin_file=bin_file, pykilosort_path=alf_path, raster_plot=True, raw_plots=True, summary_stats=False,
                             raster_start=0., raster_len=100., raw_start=50., raw_len=0.15,
                             vmax=0.05, d_bin=5, t_bin=0.001)


if __name__ == "__main__":
    spike_sort_recording(bin_file=FILE_RECORDING, output_dir=OUTPUT_DIR)
