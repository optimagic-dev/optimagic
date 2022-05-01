import numpy as np
import pytest
from estimagic.parameters.space_conversion import get_space_converter
from estimagic.parameters.tree_conversion import FlatParams
from numpy.testing import assert_array_equal as aae


@pytest.fixture
def flat_params():
    n_params = 10
    fp = FlatParams(
        values=np.arange(n_params),
        lower_bounds=np.full(n_params, -1),
        upper_bounds=np.full(n_params, 11),
        names=list("abcdefghij"),
    )
    return fp


def test_space_converter_no_constraints(flat_params):

    converter, internal_params = get_space_converter(
        flat_params=flat_params,
        flat_constraints=[],
    )

    aae(converter.params_to_internal(flat_params.values), flat_params.values)
    aae(converter.params_from_internal(flat_params.values), flat_params.values)
    aae(
        converter.derivative_to_internal(flat_params.values, flat_params.values),
        flat_params.values,
    )

    aae(internal_params.values, flat_params.values)
    aae(internal_params.lower_bounds, flat_params.lower_bounds)
    aae(internal_params.upper_bounds, flat_params.upper_bounds)
