import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from optimagic.benchmarking.scaling import (
    LinspaceBenchmarkScaler,
    pre_process_scaling,
)


def test_linspace_scaler_scale():
    scaler = LinspaceBenchmarkScaler(min_scale=1, max_scale=3)
    aaae(scaler.scale(np.array([1.0, 1.0, 1.0])), np.array([1.0, 2.0, 3.0]))


def test_linspace_scaler_unscale_inverts_scale():
    scaler = LinspaceBenchmarkScaler()
    params = np.array([-1.0, 0.5, 2.0, 7.0])
    aaae(scaler.unscale(scaler.scale(params)), params)


def test_linspace_scaler_keeps_infinite_values():
    scaler = LinspaceBenchmarkScaler()
    bounds = np.array([-np.inf, np.inf, 1.0])
    scaled = scaler.scale(bounds)
    assert scaled[0] == -np.inf
    assert scaled[1] == np.inf


@pytest.mark.parametrize(
    "kwargs", [{"min_scale": 0}, {"max_scale": -1.0}, {"min_scale": -1, "max_scale": 0}]
)
def test_linspace_scaler_with_non_positive_scale_raises(kwargs):
    with pytest.raises(ValueError, match="min_scale and max_scale must be positive"):
        LinspaceBenchmarkScaler(**kwargs)


def test_pre_process_scaling():
    scaler = LinspaceBenchmarkScaler(min_scale=2, max_scale=4)
    assert pre_process_scaling(scaler) is scaler
    assert pre_process_scaling(False) is None
    assert pre_process_scaling(True) == LinspaceBenchmarkScaler()


def test_pre_process_scaling_with_invalid_type_raises():
    with pytest.raises(TypeError, match="scaling must be a bool or an instance"):
        pre_process_scaling({"min_scale": 1})  # ty: ignore[invalid-argument-type]
