# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Added / Removed / Changed / Fixed

### [1.11.0] 2025-09-18

### Added
- Documentation and instructions for integration tests and contribution guidelines.

### Changed
- The saved channel_map now contains the index of sorted channels into the original raw data

### Fixed
- writing npy headers was using deprecated numpy functions 

## [1.10.0] 2025-06-27

### Added
- documentation and examples for EC2 and Docker integration
- in IBL processing function, option to extract the waveforms
### Fixed
- bad channels labels were not passed down to decompress_destripe_cbin pre-processing function


## [1.9.0]

### Changed
- order by Neuropixel channels byshank / row / col: swapped the readers for spikeglx.Readers 
- output channel detection QC figure with thresholds 
- removed support for original channel detection
- drop support for file list. This would have to be implemented outside with a reader object


## 1.8
### 1.8.0
- Add the passingSpikes.table.pqt output
- expose dredge and channel detection parameters

### 1.8.1
- raise Runtime error if more than half of the channels are detected as bad
- datashift patch when the number of spikes is insufficient to register the electrode signal

## 1.7
### 1.7.3
- Update installation instructions for using `cupy` alongside torch
- Remove some PyDantic and NumPy deprecations
- Reworked integration test

### 1.7.1
- bugfix geometry of probes in registration was hard coded to 384 channels
- bugfix dredge detection memory management sets a number of jobs according to GPU size

## 1.6
### 1.6a.dev5
-   reworks datashift2 to use DARTsort spike detection, fed into DREDge motion estimation
### 1.6a.dev4
-   uses ibl-neuropixel 1.0.0 that contains the saturation mitigation in the pre-processing
### 1.6a.dev3
-   spike hole branch merge with overlap parameter set to 1024
### 1.6a.dev2
-   spike hole branch merge with overlap parameter set to 0
### 1.6a.dev1
-   adds APR RMS output to pre-processing
### 1.6a.dev0
-   outputs `probe.channels.npy` via phylib forced update

## 1.5
### 1.5.1 bugfix
- `probes.channel_labels` was referenced in plural in some parts of the code
### 1.5.0
- set minFR to 0.02 Hz to avoid too many near empty clusters
## 1.4
### 1.4.5
- make pykilosort compatible with spikeglx 2023-04
### 1.4.4
- remove annoying prints (PR#16)
### 1.4.1
- bugfix: normalisation factor if `whitening` is used in pre-processing
### 1.4.0
- stabilisation of the covariance matrix prior whitening computation
- add QC outputs of the preprocessing phase

## 1.3
### 1.3.3
-   change integration tests infrastructure
### 1.3.2
-   support for NP2 geometries: label x coordinates of shanks at 200um, outputs shanks dataset
### 1.3.1
-   copy the `_iblqc_` files to the original probe directory
### 1.3.0
-   Support for Neuropixel 2
-   read geometry from the spikeglx metadata file
-   GPU support for destriping

## 1.2
### 1.2.1
-   IBL pre-proc: add the channel removal code to the computing of the whitening matrix 

### 1.2.0 alpha02
-   fix bug for the last template (Kush)

### 1.2.0 alpha01 
-   ibllib > 2.5.0 pre-processing (destriping)
    -   channel rejection and interpolation before destriping
    -   uses the pykilosort parameters for the high-pass filter
    -   multi-processing version of the destriping
-   QC:
    -   destriping outputs the RMS of each batch after pre-processing
    -   outputs

## 1.1
-   add pre-processing within pykilosort
-   whitening is optional and set as a parameter

## 1.0
### 1.0.1 2021-08-08
-   output the drift matrix in Alf format
### 1.0.2 2021-08-31
-   attempt to fix bugs introduced by chronic recordings that reduce amount of detected spikes
