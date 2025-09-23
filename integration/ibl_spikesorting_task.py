
"""
This is an integration test on the IBL task

This script tests the spike sorting pipeline for IBL (International Brain Laboratory) data.
It copies test data to a temporary location, runs the spike sorting process,
validates the outputs, and generates quality control reports.
"""
import shutil

from iblutil.util import setup_logger
from ibllib.pipes.ephys_tasks import SpikeSorting

from viz import reports
import iblsorter
from iblsorter.params import load_integration_config

config = load_integration_config()
logger = setup_logger('iblsorter', level=config.log_level)
# Dictionary for any parameter overrides (empty in this case)
override_params = {}

if __name__ == "__main__":
    # Define path to the test input data
    path_probe = config.integration_data_path.joinpath('ibl', 'probe01')
    
    # Define output directory with version-specific subfolder
    output_dir = config.integration_data_path.joinpath(
        'testing_output', 'ibl_spikesorting_task', f"{iblsorter.__version__}")
    
    # Clean up any existing output directory
    shutil.rmtree(output_dir, ignore_errors=True)
    
    # Create a simulated session directory structure
    session_path = output_dir.joinpath('Subjects', 'algernon', '2024-07-16', '001')
    raw_ephys_data_path = session_path.joinpath('raw_ephys_data', 'probe01')
    
    # Log the data copying process
    logger.info('copying raw_ephys_data to session path - removing existing folder if it exists')
    
    # Remove any existing raw data directory and create a fresh one
    shutil.rmtree(raw_ephys_data_path, ignore_errors=True)
    raw_ephys_data_path.mkdir(parents=True, exist_ok=True)
    
    # Copy the test data to the session directory
    shutil.copytree(path_probe, raw_ephys_data_path, dirs_exist_ok=True)
    
    # Log the start of the spike sorting process
    logger.info(f"iblsort run for probe01 in session {session_path}")
    
    # Initialize the spike sorting job
    # Note: one=None means no database connection is used
    ssjob = SpikeSorting(
        session_path, 
        one=None, 
        pname='probe01', 
        device_collection='raw_ephys_data', 
        location="local", 
        scratch_folder=config.scratch_dir
    )
    
    # Execute the spike sorting pipeline
    ssjob.run()
    
    # Verify the job completed successfully (status 0 indicates success)
    assert ssjob.status == 0
    
    # Verify all expected output files were generated
    ssjob.assert_expected_outputs()
    
    # Log the start of report generation
    logger.info("Outputs are validated. Compute report")
    
    # Define path to the ALF output directory (ALF = ALyx Lightweight Format)
    alf_path = session_path.joinpath('alf', 'probe01', 'iblsorter')
    
    # Generate quality control plots and metrics
    reports.qc_plots_metrics(
        # Find the binary compressed file (.cbin) in the raw data directory
        bin_file=next(raw_ephys_data_path.glob('*.ap.cbin')), 
        pykilosort_path=alf_path, 
        out_path=output_dir, 
        raster_plot=True,           # Generate spike raster plots
        raw_plots=True,             # Generate raw data plots
        summary_stats=False,        # Skip summary statistics
        raster_start=0.,            # Start time for raster plot (seconds)
        raster_len=100.,            # Duration for raster plot (seconds)
        raw_start=50.,              # Start time for raw data plot (seconds)
        raw_len=0.15,               # Duration for raw data plot (seconds)
        vmax=0.05,                  # Maximum value for color scaling
        d_bin=5,                    # Depth bin size
        t_bin=0.001                 # Time bin size (1 ms)
    )
    
    # Clean up the raw data copy to save disk space
    logger.info("Remove raw data copy")
    shutil.rmtree(raw_ephys_data_path, ignore_errors=True)
    
    # Log completion message with output location
    logger.info(f"Exiting now, test data results in {output_dir}")
