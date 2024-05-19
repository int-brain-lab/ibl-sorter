# Containerization of pykilosort

Pre-requisites:
- Docker
- Nvidia Container toolkit


## Building the container

From the `./docker` folder, run the following command:
```shell
docker buildx build . \
    --platform linux/amd64 \
    --tag int-brain-lab/pykilosort:latest
```

