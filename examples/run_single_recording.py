import argparse
from pathlib import Path
import shutil

from iblsorter.ibl import run_spike_sorting_ibl, ibl_pykilosort_params
from viz import reports

SCRATCH_DIR = Path('/mnt/h0/iblsort')  # set this to a fast SSD drive with at least 500Gb free space
override_params = {}  # here it is possible to set some parameters for the run


def spike_sort_recording(bin_file, output_dir):
    """
    The folder architecture is as follows
    ---- iblsorter  ks_output_dir
    ---- alf_path  alf_path

    :param bin_file:
    """
    bin_file = Path(bin_file)
    output_dir = Path(output_dir)

    ks_output_dir = output_dir.joinpath('iblsorter')
    alf_path = output_dir.joinpath('alf')

    # this can't be outside of a function, otherwise each multiprocessing job will execute this code!
    shutil.rmtree(SCRATCH_DIR, ignore_errors=True)
    SCRATCH_DIR.mkdir(exist_ok=True)

    ks_output_dir.mkdir(parents=True, exist_ok=True)

    params = ibl_pykilosort_params(bin_file)
    for k in override_params:
        params[k] = override_params[k]

    run_spike_sorting_ibl(bin_file, scratch_dir=SCRATCH_DIR, params=params,
                          ks_output_dir=ks_output_dir, alf_path=alf_path)

    reports.qc_plots_metrics(bin_file=bin_file, pykilosort_path=alf_path, raster_plot=True, raw_plots=True, summary_stats=False,
                             raster_start=0., raster_len=100., raw_start=50., raw_len=0.15,
                             vmax=0.05, d_bin=5, t_bin=0.001)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run spike sorting on a session')
    parser.add_argument('recording_file', help='The full path file of the AP recording')
    parser.add_argument('output_directory', help='The full path to the output directory')
    args = parser.parse_args()
    spike_sort_recording(bin_file=args.recording_file, output_dir=args.output_directory)
