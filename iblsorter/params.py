from math import ceil
from pathlib import Path
from typing import List, Optional, Tuple, Literal
import yaml

import numpy as np
from pydantic import BaseModel, Field, field_validator, DirectoryPath

import iblsorter
from .utils import Bunch


class IntegrationConfig(BaseModel):
    integration_data_path: DirectoryPath
    scratch_dir: DirectoryPath
    delete: bool


def load_integration_config(yaml_path=None) -> IntegrationConfig:
    if yaml_path is None:
        yaml_path = Path(iblsorter.__file__).parents[1].joinpath('integration', 'config.yaml')
    with open(yaml_path, 'r') as fid:
        config = yaml.safe_load(fid)
    return IntegrationConfig.model_validate(config)


class Probe(BaseModel):
    NchanTOT: int
    Nchan: Optional[int] = Field(
        None, description="Nchan < NchanTOT if some channels should not be used."
    )
    shank: Optional[np.ndarray] = Field(
        None, description="Channel shanks labels"
    )
    chanMap: np.ndarray  # TODO: add constraints
    kcoords: np.ndarray  # TODO: add constraints
    xc: np.ndarray
    yc: np.ndarray

    model_config = {
        "arbitrary_types_allowed": True
    }

    @field_validator("yc")
    def coords_same_length(cls, v, values):
        assert len(values["xc"]) == len(v)
        return v

    @classmethod
    def load_from_npy(cls, rootZ, **kwargs):
        return cls(
            chanMap=np.load(f"{rootZ}/channel_map.npy").flatten().astype(int),
            xc=np.load(rootZ + "/channel_positions.npy")[:, 0],
            yc=np.load(rootZ + "/channel_positions.npy")[:, 1],
            **kwargs,
        )


class DatashiftParams(BaseModel):
    sig: float = Field(20.0, description="sigma for the Gaussian process smoothing")
    nblocks: int = Field(
        5, description="blocks for registration. 1 does rigid registration."
    )
    output_filename: Optional[str] = Field(
        None, description="optionally save registered data to a new binary file"
    )
    overwrite: bool = Field(True, description="overwrite proc file with shifted data")

    @field_validator("nblocks")
    def validate_nblocks(v):
        if v < 1:
            raise ValueError(
                "datashift.nblocks must be >= 1, or datashift should be None"
            )
        return v


class MotionEstimationParams(BaseModel):
    """
    See https://github.com/evarol/dredge/blob/main/python/dredge/dredge_ap.py
    """

    bin_um: float = Field(1.0)
    bin_s: float = Field(1.0)
    
    max_dt_s: float = Field(1000.)
    mincorr: float = Field(0.1)

    win_shape: str = Field("gaussian")
    win_step_um: float = Field(400.)
    win_scale_um: float = Field(450.)

    # default depends on other parameters
    max_disp_um: float | None = Field(None, description="", validate_default=True) 
    @field_validator("max_disp_um")
    def set_max_disp_um(cls, v, values):
        return v or values.data["win_scale_um"] / 4.

    # default depends on other parameters
    win_margin_um: float | None = Field(None, description="", validate_default=True)  ## -win_scale_um / 2
    @field_validator("win_margin_um")
    def set_win_margin_um(cls, v, values):
        return v or -values.data["win_scale_um"] / 2.

    # weights parameters
    do_window_weights: bool = Field(True)
    weights_threshold_low: float = Field(0.2)
    weights_threshold_high: float = Field(0.2)
    mincorr_percentile: float | None = Field(None)
    mincorr_percentile_nneighbs: float | None = Field(None)

    # raster parameters
    # amp_scale_fn=None,
    # post_transform=np.log1p,
    gaussian_smoothing_sigma_um: float = Field(1)
    gaussian_smoothing_sigma_s: float = Field(1)
    avg_in_bin: bool = Field(False)
    count_masked_correlation: bool = Field(False)
    count_bins: int = Field(401)
    count_bin_min: int = Field(2)


class ChannelDetectionParams(BaseModel):
    psd_hf_threshold: float = Field(.02, description="Threshold power spectral density above which the PSD at 0.8 Nyquist is considered noisy (units: uV ** 2 / Hz ")
    similarity_threshold: Optional[Tuple[float]] = Field((-.5, 1), description="similarity threshold for channel detection")


class KilosortParams(BaseModel):
    AUCsplit: float = Field(0.9, description="""splitting a cluster at the end requires at least this much isolation for each sub-cluster (max=1)""")
    Nfilt: Optional[int] = None  # This should be a computed property once we add the probe to the config
    Th: List[float] = Field([6, 3], description="""threshold on projections (like in Kilosort1, can be different for last pass like [10 4])""",)
    ThPre: float = Field(8, description="threshold crossings for pre-clustering (in PCA projection space)",)
    channel_detection_parameters: Optional[ChannelDetectionParams] = Field(
        ChannelDetectionParams(), description='parameters for raw correlation channel detection option')
    data_dtype: str = Field('int16', description='data type of raw data')
    datashift: Optional[DatashiftParams] = Field(None, description="parameters for 'datashift' drift correction. not required")
    deterministic_mode: bool = Field(True, description="make output deterministic by sorting spikes before applying kernels")
    fs: float = Field(30000.0, description="sample rate")
    fshigh: float = Field(300.0, description="high pass filter frequency")
    fslow: Optional[float] = Field(None, description="low pass filter frequency")
    gain: int = 1
    genericSpkTh: float = Field(8.0, description="threshold for crossings with generic templates")
    lam: float = Field(10,description="""how important is the amplitude penalty (like in Kilosort1, 0 means not used,10 is average, 50 is a lot)""")
    loc_range: List[int] = [5, 4]
    long_range: List[int] = [30, 6]
    low_memory: bool = Field(False, description='low memory setting for running chronic recordings')
    minFR: float = Field(1.0 / 50,description="""minimum spike rate (Hz), if a cluster falls below this for too long it gets removed""",)
    minfr_goodchannels: float = Field(0, description="minimum firing rate on a 'good' channel (0 to skip)")
    momentum: List[float] = Field([20, 400],description="""number of samples to average over (annealed from first to second value)""")
    normalisation: Literal['whitening', 'zscore', 'global zscore'] = Field('whitening', description='Normalisation strategy. Choices are: '
    '"whitening": uses the inverse of the covariance matrix,'
    '"zscore": uses individual channel normalisation,'
    '"global zscore" for median channel normalisation')
    nPCs: int = Field(3, description="how many PCs to project the spikes into")
    nSkipCov: int = Field(25, description="compute whitening matrix from every N-th batch")
    n_channels: int = Field(385, description='number of channels in the data recording')
    nblocks: int = Field(5, description="number of blocks used to segment the probe when tracking drift, 0 == don't track, 1 == rigid, > 1 == non-rigid")
    nfilt_factor: int = Field(4, description="max number of clusters per good channel (even temporary ones)")
    nskip: int = Field(5, description="how many batches to skip for determining spike PCs")
    nt0: int = 61
    ntbuff: int = Field(64,description="""samples of symmetrical buffer for whitening and spike detection Must be multiple of 32 + ntbuff. This is the batch size (try decreasing if out of memory).""",)
    nup: int = 10
    output_filename: Optional[str] = Field(None, description="optionally save registered data to a new binary file")
    overlap_samples: int = Field(0, description='number of overlap time samples to load at the beginning and end of each batch in the main template matching algorithm')
    overwrite: bool = Field(True, description="overwrite proc file with shifted data")
    perform_drift_registration: bool = Field(True, description='Estimate electrode drift and apply registration')
    probe: Optional[Probe] = Field(None, description="recording probe metadata")
    read_only: bool = Field(False, description='Read only option for raw data') # TODO: Make this true by default
    reorder: bool = Field(True, description="whether to reorder batches for drift correction.")
    save_drift_estimates: bool = Field(False, description='save estimated probe drift')
    save_drift_spike_detections: bool = Field(False, description='save detected spikes in drift correction')
    save_temp_files: bool = Field(        True, description="keep temporary files created while running")
    scaleproc: int = Field(200, description="int16 scaling of whitened data")
    seed: Optional[int] = Field(42, description="seed for deterministic output")
    sig: int = 1
    sig_datashift: float = Field(20.0, description="sigma for the Gaussian process smoothing")
    sigmaMask: float = Field(30, description="""spatial constant in um for computing residual variance of spike""")
    skip_preprocessing: bool = Field(False, description='skip IBL destriping if the bin file is already preprocessed')
    spkTh: float = Field(-6, description="spike threshold in standard deviations")
    stable_mode: bool = Field(True, description="make output more stable")
    templateScaling: float = 20.0
    unwhiten_before_drift: bool = Field(True, description='perform unwhitening operation prior to estimating drift')
    whiteningRange: int = Field(32, description="number of channels to use for whitening each channel")

    # Computed properties
    @property
    def NT(self) -> int:
        return 64 * 1024 + self.ntbuff

    @property
    def NTbuff(self) -> int:
        return self.NT + 3 * self.ntbuff

    @property
    def nt0min(self) -> int:
        return int(ceil(20 * self.nt0 / 61))

    @property
    def ephys_reader_args(self):
        "Key word arguments passed to ephys reader"
        args = {
            'n_channels': self.n_channels,
            'dtype': self.data_dtype,
            'sample_rate': self.fs,
        }
        if self.read_only:
            args['mode'] = 'r'
        return args
