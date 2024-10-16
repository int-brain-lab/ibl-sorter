# Containerization of pykilosort

## Usage
 
The command to run a PID is the following
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

Select an EC2

1. Install the nvidia drivers (~3mins) - maybe this can be skipped using a EC2 pytorch image
```
sudo apt-get update
sudo apt install -y ubuntu-drivers-common
sudo ubuntu-drivers install
sudo reboot
```
It will take a couple of minutes before the instance is available to login again

2. Install Docker using the bootstrap script provided (~2mins)

[//]: # todo change the docker branch to main once released()
```
mkdir -p ~/Documents/PYTHON
cd ~/Documents/PYTHON
git clone -b aws https://github.com/int-brain-lab/ibl-sorter.git
cd ibl-sorter/docker
sudo ./setup_nvidia_container_toolkit.sh
```

3. Setup ONE from inside the container, make sure the cache directory is `/mnt/s0/spikesorting`, configure the base URL according to your needs,
for internal re-runs it should be set to https://alyx.internationalbrainlab.org


4. If you want to send the data to flatiron you'll have to setup the `~/.ssh/config` file so as to reflect the `sdsc` SSH configuration.
