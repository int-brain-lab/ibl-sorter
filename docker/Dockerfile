# DOCKER_BUILDKIT=0 docker build .
FROM  pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
#FROM cupy/cupy:latest
# RUN nvidia-smi  # uncomment to enforce GPU availability during the run

RUN apt-get update && apt-get install -y \
    git \
    libfftw3-dev \
    libfftw3-doc \
    && apt-get autoremove \
    && rm -rf /var/lib/apt/lists/*

RUN pip install cupy-cuda12x
## clone all of the repos
RUN mkdir PYTHON
WORKDIR PYTHON
RUN git clone -b ibl_prod_dev https://github.com/int-brain-lab/pykilosort.git
RUN git clone https://github.com/int-brain-lab/iblscripts.git
RUN git clone -b spikesorting_rerun https://github.com/int-brain-lab/ibllib.git

RUN pip install -r pykilosort/requirements.txt
## RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip uninstall -y ibllib
RUN pip install -e ./ibllib
#RUN pip install -e ./dredge
#RUN pip install -e ./dartsort
#RUN pip install -e ./iblscripts
RUN pip install ipython
ENTRYPOINT ["top", "-b"]