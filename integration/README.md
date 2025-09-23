# Integration tests

The spike sorting integration test datasets are available here:

https://ibl-brain-wide-map-public.s3.amazonaws.com/index.html#spikesorting/integration_tests/


## Setup: download data and set local paths

### Download the dataset
The test data is in the [stand-alone](https://ibl-brain-wide-map-public.s3.amazonaws.com/index.html#spikesorting/integration_tests) folder. Here is the command to download this using a shell.


```shell
LOCAL_FOLDER="/datadisk/Data/spike-sorting/integration-tests"

aws s3 sync s3://ibl-brain-wide-map-public/spikesorting/integration_tests $LOCAL_FOLDER --no-sign-request
```
The content of your local folder should be:

```shell
stand-alone/
├── _iblqc_ephysSaturation.samples.npy
├── _iblqc_ephysTimeRmsAP.rms.npy
├── _iblqc_ephysTimeRmsAP.timestamps.npy
├── imec_385_100s.ap.bin
└── imec_385_100s.ap.meta
ibl
└── probe01
    ├── 749cb2b7.ap.cbin
    ├── 749cb2b7.ap.ch
    ├── 749cb2b7.ap.meta
    └── 749cb2b7.sync.npy
```

## Modify the configuration
Copy the [integration\config_template.yaml](integration\config_template.yaml) file to `integration\config.yaml` and set the temporary scratch directory for spike sorting and the integration tests data path. Additionally you can setup the log level

## Running the tests

### Integration 100s

This is the generic test, with an uncompressed binary spikeglx file of 100 seconds, acquired with a NP1 probe.

Run the `integration/integration_100s.py` script.


## IBL - full task integration

This is an IBL session, and the test will replicate the IBL folder architecture, and additionally output quality control datasets and re-extract the waveforms from the raw data.

Run the `integration/ibl-spikesorting_task.py` script.

