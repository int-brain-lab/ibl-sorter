import numpy as np
import pytest
from iblsorter.params import MotionEstimationParams, KilosortParams
from iblsorter.ibl import ibl_dredge_params

def test_motion_params():
    # test modify params
    p = MotionEstimationParams(max_dt_s=1200.)
    assert p.max_dt_s == 1200.
    p = MotionEstimationParams()
    p.max_dt_s = 1500.
    assert p.max_dt_s == 1500.
    # test validator default, should set to -win_scale_um/2
    p = MotionEstimationParams(win_margin_um=None, win_scale_um=400.)
    assert p.win_margin_um == -200.

def test_kilosort_params():
    # test modify params
    p = KilosortParams(skip_preprocessing=True)
    assert p.skip_preprocessing
    p = KilosortParams()
    p.skip_preprocessing = True
    assert p.skip_preprocessing

    # document property behavior
    p = KilosortParams(n_channels=40, data_dtype="float64", fs=2500)
    test_args = {"n_channels": 40, "dtype": "float64", "sample_rate": 2500}
    assert p.ephys_reader_args == test_args

    # can't set this directly
    p = KilosortParams()
    with pytest.raises(AttributeError):
        p.ephys_reader_args = test_args

    # setting it in constructor does not work either
    p = KilosortParams(n_channels=40, data_dtype="float64", fs=2500, 
                       ephys_reader_args={
                           "n_channels": 384,
                           "dtype": "float32",
                           "sample_rate": 2500
                       }
                    )
    assert p.ephys_reader_args == test_args

def test_get_motion_params_from_kilosort_params():
    k = KilosortParams()
    p = ibl_dredge_params(k)
    assert p.bin_s == p.gaussian_smoothing_sigma_s == k.NT / k.fs
    assert p.mincorr == 0.5
    
    