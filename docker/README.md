# Containerization of pykilosort


## Installation of the container

Pre-requisites:
- Docker
- Nvidia Container toolkit

```
git clone -b ibl_prod_dev https://github.com/int-brain-lab/pykilosort.git
cd pykilosort/docker
sudo ./setup_nvidia_container_toolkit.sh
```

## Building the container

From the `./docker` folder, run the following command:
```shell
docker buildx build . --platform linux/amd64 --tag int-brain-lab/pykilosort:latest
```

Running the container in interactive mode: 
`docker compose up -d`

And then connect to it by running a terminal:
`sudo docker compose exec spikesorter /bin/bash`
