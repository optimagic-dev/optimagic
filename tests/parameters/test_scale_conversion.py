import numpy as np
import pytest
from estimagic import first_derivative
from estimagic.parameters.scale_conversion import get_scale_converter
from estimagic.parameters.tree_conversion import FlatParams
from numpy.testing import assert_array_almost_equal as aaae


def func_scalar(x):
    return (x**2).sum()


def func_vector(x):
    return x.reshape(2, -1) ** 2


TEST_CASES = {
    "start_values": FlatParams(
        values=np.array([0, 1, 1, 1, 1, 1]),
        lower_bounds=np.array([-2, 0, 0.5, 2 / 3, 3 / 4, 4 / 5]),
        upper_bounds=np.array([2, 2, 1.5, 4 / 3, 5 / 4, 6 / 5]),
        names=None,
    ),
    "bounds": FlatParams(
        values=np.full(6, 0.5),
        lower_bounds=np.zeros(6),
        upper_bounds=np.ones(6),
        names=None,
    ),
    "gradient": FlatParams(
        values=np.array([0, 0.5, 0.5, 0.5, 0.5, 0.5]),
        lower_bounds=np.array([-2, 0, 1 / 4, 2 / 6, 3 / 8, 4 / 10]),
        upper_bounds=np.array([2, 1, 3 / 4, 4 / 6, 5 / 8, 6 / 10]),
        names=None,
    ),
}

IDS = list(TEST_CASES)
PARAMETRIZATION = list(TEST_CASES.items())


@pytest.mark.parametrize("method, expected", PARAMETRIZATION, ids=IDS)
def test_get_scale_converter_scalar_start_values(method, expected):
    params = FlatParams(
        values=np.arange(6),
        lower_bounds=np.arange(6) - 1,
        upper_bounds=np.arange(6) + 1,
        names=list("abcdef"),
    )

    scaling_options = {
        "method": method,
        "clipping_value": 0.5,
    }

    converter, scaled = get_scale_converter(
        flat_params=params, func=func_scalar, scaling_options=scaling_options
    )

    aaae(scaled.values, expected.values)
    aaae(scaled.lower_bounds, expected.lower_bounds)
    aaae(scaled.upper_bounds, expected.upper_bounds)

    aaae(converter.params_to_internal(params.values), expected.values)
    aaae(converter.params_from_internal(expected.values), params.values)

    calculated_jacobian = converter.derivative_to_internal(np.eye(len(params.values)))

    numerical_jacobian = first_derivative(
        converter.params_from_internal, expected.values
    )["derivative"]

    aaae(calculated_jacobian, numerical_jacobian)
