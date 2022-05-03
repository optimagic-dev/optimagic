import numpy as np
from estimagic.parameters.conversion import get_converter
from numpy.testing import assert_array_almost_equal as aaae


def test_get_converter_fast_case():

    converter, internal = get_converter(
        func=lambda x: (x**2).sum(),
        params=np.arange(3),
        constraints=None,
        lower_bounds=None,
        upper_bounds=None,
        func_eval=3,
        derivative_eval=2 * np.arange(3),
        primary_key="value",
        scaling=False,
        scaling_options=None,
    )

    aaae(internal.values, np.arange(3))
    aaae(internal.lower_bounds, np.full(3, -np.inf))
    aaae(internal.upper_bounds, np.full(3, np.inf))

    aaae(converter.params_to_internal(np.arange(3)), np.arange(3))
    aaae(converter.params_from_internal(np.arange(3)), np.arange(3))
    aaae(
        converter.derivative_to_internal(2 * np.arange(3), np.arange(3)),
        2 * np.arange(3),
    )
    aaae(converter.func_to_internal(3), 3)


#
