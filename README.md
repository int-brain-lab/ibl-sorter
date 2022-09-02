# Python port of KiloSort2

This is a Python port of the original MATLAB version of [Kilosort 2.5](https://github.com/MouseLand/Kilosort), written by Marius Pachitariu, with Neuropixel specific improvements and software engineering enhancements.
The modifications are described in [this white paper](https://doi.org/10.6084/m9.figshare.19705522.v3).

## Installation 

### System Requirements

The code makes extensive use of the GPU via the CUDA framework. A high-end NVIDIA GPU with at least 8GB of memory is required.


### Doing the install using Anaconda (Linux)

Only on Linux, first install fftw by running the following 
    
    sudo apt-get install -y libfftw3-dev

Navigate to the desired location for the repository and clone it

    git clone https://github.com/int-brain-lab/pykilosort.git
    cd pykilosort

Create a conda environment

    conda env create -f pyks2.yml
    conda activate pyks2
    pip install -e .

## Usage

### Example

We provide a few datasets to explore parametrization and test on several brain regions.
The smallest dataset is a 100 seconds excerpt to test the installation. Here is the minimal working example:

```python
import shutil
from pathlib import Path

from pykilosort.ibl import run_spike_sorting_ibl, ibl_pykilosort_params, download_test_data

data_path = Path("/mnt/s0/spike_sorting/integration_tests")  # path on which the raw data will be downloaded
scratch_dir = Path.home().joinpath("scratch", 'pykilosort')  # temporary path on which intermediate raw data will be written, we highly recommend a SSD drive
ks_output_dir = Path("/mnt/s0/spike_sorting/outputs")  # path containing the kilosort output unprocessed
alf_path = ks_output_dir.joinpath('alf')  # this is the output standardized as per IBL standards (SI units, ALF convention)

# download the integration test data from amazon s3 bucket
bin_file, meta_file = download_test_data(data_path)

# prepare and mop up folder architecture for consecutive runs
DELETE = True  # delete the intermediate run products, if False they'll be copied over to the output directory for debugging
shutil.rmtree(scratch_dir, ignore_errors=True)
scratch_dir.mkdir(exist_ok=True)
ks_output_dir.mkdir(parents=True, exist_ok=True)

# loads parameters and run
params = ibl_pykilosort_params(bin_file)
params['Th'] = [6, 3]
run_spike_sorting_ibl(bin_file, delete=DELETE, scratch_dir=scratch_dir,
                      ks_output_dir=ks_output_dir, alf_path=alf_path, log_level='INFO', params=params)
```

## Troubleshooting
### Managing CUDA Errors

Errors with the CUDA installation can sometimes be fixed by downgrading
the version of cudatoolkit installed. Currently tested versions are 9.2,
10.0, 10.2, 11.0 and 11.5

To check the current version run the following:

    conda activate pyks2
    conda list cudatoolkit

To install version 10.0 for example run the following

    conda activate pyks2
    conda remove cupy, cudatoolkit
    conda install -c conda-forge cupy cudatoolkit=10.0