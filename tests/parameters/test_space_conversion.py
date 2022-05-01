import numpy as np
import pytest
from estimagic import first_derivative
from estimagic.parameters.space_conversion import get_space_converter
from estimagic.parameters.tree_conversion import FlatParams
from numpy.testing import assert_array_almost_equal as aaae
from numpy.testing import assert_array_equal as aae


def _get_test_case_no_constraint():
    n_params = 10
    fp = FlatParams(
        values=np.arange(n_params),
        lower_bounds=np.full(n_params, -1),
        upper_bounds=np.full(n_params, 11),
        names=list("abcdefghij"),
    )

    constraints = []
    return constraints, fp, fp


def _get_test_case_fixed(with_value):
    fp = FlatParams(
        values=np.arange(5),
        lower_bounds=np.full(5, -np.inf),
        upper_bounds=np.full(5, np.inf),
        names=list("abcde"),
    )
    if with_value:
        constraints = [{"index": [0, 2, 4], "type": "fixed", "value": [0, 2, 4]}]
    else:
        constraints = [{"index": [0, 2, 4], "type": "fixed"}]

    internal = FlatParams(
        values=np.array([1, 3]),
        lower_bounds=np.full(2, -np.inf),
        upper_bounds=np.full(2, np.inf),
        names=None,
    )

    return constraints, fp, internal


TEST_CASES = {
    "no_constraints": _get_test_case_no_constraint(),
    "fixed_at_start": _get_test_case_fixed(with_value=False),
    "fixed_at_value": _get_test_case_fixed(with_value=True),
}


PARAMETRIZATION = list(TEST_CASES.values())
IDS = list(TEST_CASES)


@pytest.mark.parametrize(
    "constraints, params, expected_internal", PARAMETRIZATION, ids=IDS
)
def test_space_converter_with_params(constraints, params, expected_internal):
    converter, internal = get_space_converter(
        flat_params=params,
        flat_constraints=constraints,
    )

    aae(internal.values, expected_internal.values)
    aae(internal.lower_bounds, expected_internal.lower_bounds)
    aae(internal.upper_bounds, expected_internal.upper_bounds)

    aae(converter.params_to_internal(params.values), expected_internal.values)
    aae(converter.params_from_internal(expected_internal.values), params.values)

    numerical_jacobian = first_derivative(
        converter.params_from_internal, expected_internal.values
    )["derivative"]

    calculated_jacobian = converter.derivative_to_internal(
        external_derivative=np.eye(len(params.values)),
        internal_values=expected_internal.values,
    )

    aaae(calculated_jacobian, numerical_jacobian)
