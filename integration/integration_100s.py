import shutil
from iblutil.util import setup_logger

import iblsorter
from iblsorter.ibl import run_spike_sorting_ibl, ibl_pykilosort_params
from iblsorter.params import load_integration_config
from viz import reports

config = load_integration_config()
setup_logger('iblsorter', level=config.log_level)
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
    output_dir = config.integration_data_path.joinpath(
        'testing_output', 'integration_100s', f"{iblsorter.__version__}" + label, bin_file.name.split('.')[0])
    ks_output_dir = output_dir.joinpath('iblsorter')
    alf_path = ks_output_dir.joinpath('alf')
    shutil.rmtree(config.scratch_dir, ignore_errors=True)
    config.scratch_dir.mkdir(exist_ok=True)
    ks_output_dir.mkdir(parents=True, exist_ok=True)

    params = ibl_pykilosort_params(bin_file)
    for k in override_params:
        params[k] = override_params[k]

    run_spike_sorting_ibl(bin_file, delete=config.delete, scratch_dir=config.scratch_dir, params=params,
                          ks_output_dir=ks_output_dir, alf_path=alf_path)
    # we copy the temporary files to the output directory if we want to investigate them
    if not config.delete:
        working_directory = config.scratch_dir.joinpath('.kilosort', bin_file.stem)
        pre_proc_file = working_directory.joinpath('proc.dat')
        intermediate_directory = ks_output_dir.joinpath('intermediate')
        intermediate_directory.mkdir(exist_ok=True)
        shutil.copy(pre_proc_file, intermediate_directory)

    reports.qc_plots_metrics(bin_file=bin_file, pykilosort_path=alf_path, out_path=output_dir, raster_plot=True,
                             raw_plots=True, summary_stats=False, raster_start=0., raster_len=100., raw_start=50., raw_len=0.15,
                             vmax=0.05, d_bin=5, t_bin=0.001)

if __name__ == "__main__":
    run_integration_test(config.integration_data_path.joinpath('stand-alone', 'imec_385_100s.ap.bin'))
