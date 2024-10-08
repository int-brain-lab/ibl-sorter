FROM  pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
# RUN nvidia-smi  # uncomment to enforce GPU availability during the run

RUN apt-get update && apt-get install -y \
    vim \
    gcc \
    git \
    libfftw3-dev \
    libfftw3-doc \
    rsync \
    && apt-get autoremove \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /root/Documents/PYTHON
WORKDIR /root/Documents/PYTHON

RUN git clone -b ibl_prod_dev https://github.com/int-brain-lab/pykilosort.git
RUN git clone -b spikesorting_rerun https://github.com/int-brain-lab/ibllib.git
RUN git clone https://github.com/int-brain-lab/iblscripts.git
RUN git clone https://github.com/cwindolf/dartsort.git
RUN git clone https://github.com/evarol/dredge.git

RUN pip install cupy-cuda12x
RUN pip install -r pykilosort/requirements.txt
RUN pip install -r dartsort/requirements-full.txt
RUN pip install -r dredge/requirements.txt
RUN pip uninstall -y ibllib
RUN pip install -e ./ibllib
RUN pip install -e ./pykilosort
RUN pip install -e ./dartsort
RUN pip install -e ./dredge
RUN pip install h5py
RUN pip install hdbscan
RUN pip install ipython

COPY run_pykilosort.sh /root/Documents/PYTHON/iblscripts/deploy/serverpc/kilosort2/run_pykilosort.sh
RUN git -C ./pykilosort pull
ENTRYPOINT ["top", "-b"]
