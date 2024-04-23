# 1.6
### 1.6a.dev0
-   outputs `probe.channels.npy` via phylib forced update
### 1.6a.dev1
-   merge the spike holes fix branch for testing
- 
# 1.5
### 1.5.1 bugfix
- `probes.channel_labels` was referenced in plural in some parts of the code
### 1.5.0
- set minFR to 0.02 Hz to avoid too many near empty clusters
# 1.4
### 1.4.5
- make pykilosort compatible with spikeglx 2023-04
### 1.4.4
- remove annoying prints (PR#16)
### 1.4.1
- bugfix: normalisation factor if `whitening` is used in pre-processing
### 1.4.0
- stabilisation of the covariance matrix prior whitening computation
- add QC outputs of the preprocessing phase

# 1.3
## 1.3.3
-   change integration tests infrastructure
## 1.3.2
-   support for NP2 geometries: label x coordinates of shanks at 200um, outputs shanks dataset
## 1.3.1
-   copy the `_iblqc_` files to the original probe directory
## 1.3.0
-   Support for Neuropixel 2
-   read geometry from the spikeglx metadata file
-   GPU support for destriping

# 1.2
## 1.2.1
-   IBL pre-proc: add the channel removal code to the computing of the whitening matrix 

## 1.2.0 alpha02
-   fix bug for the last template (Kush)

## 1.2.0 alpha01 
-   ibllib > 2.5.0 pre-processing (destriping)
    -   channel rejection and interpolation before destriping
    -   uses the pykilosort parameters for the high-pass filter
    -   multi-processing version of the destriping
-   QC:
    -   destriping outputs the RMS of each batch after pre-processing
    -   outputs

# 1.1
-   add pre-processing within pykilosort
-   whitening is optional and set as a parameter

# 1.0
## 1.0.1 2021-08-08
-   output the drift matrix in Alf format
## 1.0.2 2021-08-31
-   attempt to fix bugs introduced by chronic recordings that reduce amount of detected spikes
