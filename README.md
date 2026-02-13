# IBL Spike Sorting

This is the implementation of the IBL spike sorting pipeline described on this white paper: (https://doi.org/10.6084/m9.figshare.19705522.v4).
The clustering part is based on the original MATLAB version of [Kilosort 2.5](https://github.com/MouseLand/Kilosort), written by Marius Pachitariu.

## Usage

We provide a few datasets to explore parametrization and test on several brain regions.
The smallest dataset is a 100 seconds excerpt to test the installation. You can see the download instructions [Here](./integration/README.md)

1) if you want to override default parameters, copy the [](./iblsorter/examples/iblsorter_parameters.yaml) in the same folder as your raw binary file and edit its values
2) make sure you have a scratch directory with > 200 Gb of disk space to write temporary data
3) run the command `iblsorter /path/to/raw_data_file.bin --scratch_directory /path/to/scratch`


```shell
SCRATCH_DIR=~/scratch  # SSD drive with > 200Gb space to write temporary data
INPUT_FILE=/datadisk/Data/spike-sorting/integration-tests/stand-alone/imec_385_100s.ap.bin
iblsorter $INPUT_FILE --scratch_directory $SCRATCH_DIR
```

Additionally, `iblsorter --help` will display additional command options.

## Installation 

### System Requirements

The code makes extensive use of the GPU via the CUDA framework. A high-end NVIDIA GPU with at least 8GB of memory is required.
The solution has been deployed and tested on Cuda 12+ and Python 3.10, 3.11 and 3.12


### Python environment

Only on Linux, first install fftw by running the following 
    
    sudo apt-get install -y libfftw3-dev

Navigate to the desired location for the repository and clone it

    git clone https://github.com/int-brain-lab/ibl-sorter.git
    cd ibl-sorter

Installation for cuda 11.x

    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    uv pip install cupy-cuda11x
    uv pip install -e .

Installation for cuda 12.x (as of October 2024, check installation instructions from pytorch for the latest)

    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
    uv pip install cupy-cuda12x
    uv pip install -e .

### Making sure the installation is successful and CUDA is available

Here we make sure that both `cupy` and `torch` are installed and that the CUDA framework is available.

```python
from iblsorter.utils import cuda_installation_test
cuda_installation_test()
```

Then we can run the integration test.
