# Containerization of pykilosort

## Usage
 
The command to run a PID is the following:
 ```shell
tmux
sudo docker compose exec spikesorter python /root/Documents/PYTHON/ibl-sorter/docker/run_pid.py 39e6c9a9-2241-4781-9ed6-db45979207e7
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


## Installing from a blank EC2

Select an EC2 instance type with Ubuntu on it (P2 or GP3).
Needs to have one system volume of 16 Gb (for the docker build) and an another volume of 800 Gb


1. Install the nvidia drivers (~3mins) - maybe this can be skipped using a EC2 pytorch image
```shell
sudo apt-get update
sudo apt install -y ubuntu-drivers-common
sudo ubuntu-drivers install
sudo reboot
```
It will take a couple of minutes before the instance is available to login again

2. Install Docker with GPU support using the bootstrap script provided (~2mins)

```shell
mkdir -p ~/Documents/PYTHON
cd ~/Documents/PYTHON
git clone -b aws https://github.com/int-brain-lab/ibl-sorter.git
cd ibl-sorter/docker
sudo ./setup_nvidia_container_toolkit.sh
```

3. Format and mount the attached volume, here you want to check that the volume is indeed `/dev/xvdb` using `df -h` (few secs):
```shell
sudo mkfs -t xfs /dev/xvdb
sudo mkdir -p /mnt/s0
sudo mount /dev/xvdb /mnt/s0
sudo chown ubuntu:ubuntu -fR /mnt/s0
df -h
```

4. Build the docker image referring to the instructions above and start the container

5. Setup ONE from inside the container, make sure the cache directory is `/mnt/s0/spikesorting`, configure the base URL according to your needs,
for internal re-runs it should be set to https://alyx.internationalbrainlab.org
Here is the command to enter the container shell: 
```shell
sudo docker compose exec spikesorter /bin/bash
```

6. If you want to send the data to flatiron you'll have to setup the `~/.ssh/config` file so as to reflect the `sdsc` SSH configuration.
