# Containerization of ibl-sorter

## Usage

The command to run a PID is the following:
```shell
BINFILE=/mnt/s1/spikesorting/integration_tests/testing_input/integration_100s/imec_385_100s.ap.bin
cd /home/olivier/PycharmProjects/pykilosort/ibl-sorter/docker

docker run --rm --gpus 1 -it internationalbrainlab/iblsorter:latest 
# from cupy_backends.cuda.libs import cusolver
docker run \
  --rm \
  --name spikesorter \
  --gpus 1 \
  -v /mnt/s1:/mnt/s1 \
  -v /home/$USER/.one:/root/.one \
  -v /mnt/h1:/scratch \
  internationalbrainlab/iblsorter:latest \
  python /root/Documents/PYTHON/ibl-sorter/examples/run_single_recording.py $BINFILE  /mnt/h1/iblsorter_integration --scratch_directory /scratch

```

 
For IBL users, the command to run spike sorting for a registered PID is the following:
```shell
sudo docker compose exec spikesorter python /root/Documents/PYTHON/ibl-sorter/examples/run_ibl_recording.py eid probe00 --cache_dir /mnt/s0/ONE
```

This is the command to get access to a shell inside of the container: 
```shell
sudo docker compose exec spikesorter /bin/bash
```

## Installation of the container

The container Dockerfile and building procedure is described a the following repository: https://github.com/int-brain-lab/iblsre/tree/main/servers
