# Containerization of ibl-sorter

## Usage

The command to run a PID is the following:
```shell
tmux
sudo docker compose exec spikesorter python /root/Documents/PYTHON/ibl-sorter/docker/run_ibl_pid.py 39e6c9a9-2241-4781-9ed6-db45979207e7
```

 
For IBL users, the command to run spike sorting for a registered PID is the following:
```shell
tmux
sudo docker compose exec spikesorter python /root/Documents/PYTHON/ibl-sorter/docker/run_ibl_pid.py 39e6c9a9-2241-4781-9ed6-db45979207e7
```

## Installation of the container

Pre-requisites:
- nvidia driver
- Docker
- Nvidia Container toolkit

```
sudo ./setup_nvidia_container_toolkit.sh
```

### Building the image, and creating the container

From the `./docker` folder, run the following command. This will take 5 to 10 mins
```shell
sudo docker buildx build . --platform linux/amd64 --tag int-brain-lab/iblsorter:latest
```
Once the image is built, create and run the container in the background
```shell
sudo docker compose up -d 
```

And this is the command to access a shell in the container:
```shell
cd Documents/PYTHON/ibl-sorter/docker/
sudo docker compose exec spikesorter /bin/bash
``` 


