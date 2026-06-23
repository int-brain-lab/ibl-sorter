import numpy as np
import pytest
from iblsorter.params import DestripeParams, MotionEstimationParams, KilosortParams
import yaml
from iblsorter.ibl import ibl_pykilosort_params, probe_geometry


def test_override_parameters_with_local_yaml_file(tmp_path, monkeypatch):
    temp_dir = tmp_path.joinpath('iblsorter_test')
    temp_dir.mkdir()
    # Create a YAML file with custom parameters
    config_path = temp_dir / "iblsorter_parameters.yaml"

    custom_params = {
        "channel_detection_parameters": {
            "psd_hf_threshold": .2,
        },
        "motion_estimation_parameters": {
            "win_scale_um": 500,
        }
    }
    with open(config_path, 'w') as f:
        yaml.dump(custom_params, f)

    # we get the default values
    params = ibl_pykilosort_params()
    assert params.motion_estimation_parameters.win_scale_um == 450
    assert params.channel_detection_parameters.psd_hf_threshold == .02
    # when a yaml file is present in the folder it will be read and overlad params
    mock_geometry = probe_geometry(npx_version=1)
    monkeypatch.setattr('iblsorter.ibl.probe_geometry', lambda *args, **kwargs: mock_geometry)
    params = ibl_pykilosort_params(bin_file=temp_dir.joinpath('toto.bin'))
    assert params.motion_estimation_parameters.win_scale_um == 500
    assert params.channel_detection_parameters.psd_hf_threshold == .2
    # the arguments take precedence over the yaml file
    params = ibl_pykilosort_params(bin_file=temp_dir.joinpath('toto.bin'), params=
    {"channel_detection_parameters": {"psd_hf_threshold": .3}})
    assert params.motion_estimation_parameters.win_scale_um == 500
    assert params.channel_detection_parameters.psd_hf_threshold == .3



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
    params = KilosortParams(NT=10_000, fs=25_000)
    assert params.motion_estimation_parameters.bin_s == params.motion_estimation_parameters.gaussian_smoothing_sigma_s == params.NT / params.fs
    assert params.motion_estimation_parameters.mincorr == 0.5


def test_destripe_params_defaults():
    p = DestripeParams()
    assert p.k_filter is True
    assert p.k_kwargs is None
    assert p.butter_kwargs is None


def test_destripe_params_k_filter_variants():
    """k_filter accepts True (k-filter), False (CAR), or None (skip)."""
    assert DestripeParams(k_filter=True).k_filter is True
    assert DestripeParams(k_filter=False).k_filter is False
    assert DestripeParams(k_filter=None).k_filter is None


def test_destripe_params_custom_dicts():
    # fs is not stored in butter_kwargs — it is injected at runtime from params.fs
    p = DestripeParams(
        butter_kwargs={'N': 3, 'Wn': 300.0, 'btype': 'highpass'},
        k_kwargs={'ntr_pad': 30, 'lagc': 1500},
    )
    assert p.butter_kwargs['Wn'] == 300.0
    assert 'fs' not in p.butter_kwargs
    assert p.k_kwargs['ntr_pad'] == 30


def test_destripe_params_nested_in_kilosort_params():
    kp = KilosortParams(destripe_parameters={'k_filter': False, 'k_kwargs': {'ntr_pad': 30}})
    assert kp.destripe_parameters.k_filter is False
    assert kp.destripe_parameters.k_kwargs == {'ntr_pad': 30}


def test_destripe_params_yaml_roundtrip(tmp_path):
    """destripe_parameters round-trips through a YAML file."""
    # fs is absent from butter_kwargs in the config — injected at runtime from params.fs
    custom = {
        'destripe_parameters': {
            'butter_kwargs': {'N': 3, 'Wn': 500.0, 'btype': 'highpass'},
            'k_filter': False,
            'k_kwargs': {'ntr_pad': 30, 'lagc': 1500},
        }
    }
    yaml_path = tmp_path / 'params.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(custom, f)
    with open(yaml_path) as f:
        loaded = yaml.safe_load(f)
    p = KilosortParams.model_validate(loaded)
    assert p.destripe_parameters.k_filter is False
    assert p.destripe_parameters.butter_kwargs['Wn'] == 500.0
    assert 'fs' not in p.destripe_parameters.butter_kwargs
    assert p.destripe_parameters.k_kwargs['ntr_pad'] == 30
