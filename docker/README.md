# Containerization of pykilosort

## Usage
 
 ```shell
sudo docker compose run spikesorter 5d570bf6-a4c6-4bf1-a14b-2c878c84ef0e
 ```

## Installation of the container

Pre-requisites:
- nvidia driver
- Docker
- Nvidia Container toolkit

```
sudo ./setup_nvidia_container_toolkit.sh
```

### Building the container

From the `./docker` folder, run the following command:
```shell
docker buildx build . --platform linux/amd64 --tag int-brain-lab/pykilosort:latest
```

Running the container in interactive mode: 
`sudo docker compose up -d`

And then connect to it by running a terminal:
`sudo docker compose exec spikesorter /bin/bash`


## Installing from a blank EC2

1. Install the nvidia drivers
```
sudo apt install -y ubuntu-drivers-common
sudo ubuntu-drivers install
sudo reboot
```

2. Install Docker using the bootstrap script provided
```
mkdir -p ~/Documents/PYTHON
cd ~/Documents/PYTHON
git clone -b ibl_prod_dev https://github.com/int-brain-lab/pykilosort.git
cd pykilosort/docker
sudo ./setup_nvidia_container_toolkit.sh
```

3. Setup ONE from inside the container, make sure the base url for alyx is `/mnt/s0/spikesorting`

4. If you want to send the data to flatiron you'll have to setup the `~/.ssh/config` file so as to reflect the `sdsc` SSH configuration.
