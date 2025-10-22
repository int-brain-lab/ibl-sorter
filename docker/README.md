# Containerization of ibl-sorter

## Usage

The command to run a PID is the following:
TODO: env_file + docker compose
TODO: iblsorter --version
TODO: set the scratch dir inside the docker to home so that the argument is not necessary

```shell

SCRATCH_DIR=/home/rlab/scratch
DATA_PATH=/mnt/disk2/Naz/Ephys/test5_g0_imec0
DATA_FILE=test5_g0_t0.imec0.ap.bin 

docker run \
  -it \
  --rm \
  --gpus=all \
  --name spikesorter \
  -v $DATA_PATH:/mnt/s0 \
  -v $SCRATCH_DIR:/scratch \
  internationalbrainlab/iblsorter:1.12 \
  iblsorter /mnt/s0/$DATA_FILE --scratch_directory /scratch
```
 
For IBL users, the command to run spike sorting for a registered PID is the following:
```shell
sudo docker compose exec spikesorter python /root/Documents/PYTHON/ibl-sorter/examples/run_ibl_recording.py eid probe00 --cache_dir /mnt/s0/ONE
```

This is the command to get access to a shell inside of the container: 
```shell
docker run --rm --gpus 1 -it internationalbrainlab/iblsorter:latest
```

## Installation of the container

The container Dockerfile and building procedure is described a the following repository: https://github.com/int-brain-lab/iblsre/tree/main/servers
