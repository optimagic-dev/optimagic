import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from estimagic.optimization.scaling import calculate_scaling_factor_and_offset


@pytest.fixture
def inputs():
    params = pd.DataFrame()
    params["value"] = np.arange(5)
    params["lower_bound"] = [-1, 0, 0, 0, 0]
    params["upper_bound"] = np.full(5, 10)

    constraints = [{"loc": [3, 4], "type": "fixed"}]

    def criterion(params):
        return (params["value"] ** 2).sum()

    return params, constraints, criterion


def test_scaling_with_start_values(inputs):
    calc_factor, calc_offest = calculate_scaling_factor_and_offset(
        *inputs, method="start_values"
    )
    aaae(calc_factor, np.array([0.1, 1, 2]))
    assert calc_offest is None


def test_scaling_with_start_values_with_magnitude(inputs):
    calc_factor, calc_offest = calculate_scaling_factor_and_offset(
        *inputs, method="start_values", magnitude=4
    )
    aaae(calc_factor, np.array([0.025, 0.25, 0.5]))
    assert calc_offest is None


def test_scaling_with_bounds(inputs):
    calc_factor, calc_offset = calculate_scaling_factor_and_offset(
        *inputs, method="bounds"
    )

    aaae(calc_factor, np.array([11, 10, 10]))
    aaae(calc_offset, np.array([-1, 0, 0]))


def test_scaling_with_gradient(inputs):
    calc_factor, calc_offset = calculate_scaling_factor_and_offset(
        *inputs, method="gradient", clipping_value=0.2
    )

    aaae(calc_factor, np.array([0.2, 2, 4]))
    assert calc_offset is None
