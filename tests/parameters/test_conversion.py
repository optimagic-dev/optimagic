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


def test_get_converter_with_constraints_and_bounds():
    converter, internal = get_converter(
        func=lambda x: (x**2).sum(),
        params=np.arange(3),
        constraints=[{"loc": 2, "type": "fixed"}],
        lower_bounds=np.array([-1, -np.inf, -np.inf]),
        upper_bounds=np.array([np.inf, 10, np.inf]),
        func_eval=3,
        derivative_eval=2 * np.arange(3),
        primary_key="value",
        scaling=False,
        scaling_options=None,
    )

    aaae(internal.values, np.arange(2))
    aaae(internal.lower_bounds, np.array([-1, -np.inf]))
    aaae(internal.upper_bounds, np.array([np.inf, 10]))

    aaae(converter.params_to_internal(np.arange(3)), np.arange(2))
    aaae(converter.params_from_internal(np.arange(2)), np.arange(3))
    aaae(
        converter.derivative_to_internal(2 * np.arange(3), np.arange(2)),
        2 * np.arange(2),
    )
    aaae(converter.func_to_internal(3), 3)


def test_get_converter_with_scaling():

    converter, internal = get_converter(
        func=lambda x: (x**2).sum(),
        params=np.arange(3),
        constraints=None,
        lower_bounds=np.arange(3) - 1,
        upper_bounds=np.arange(3) + 1,
        func_eval=3,
        derivative_eval=2 * np.arange(3),
        primary_key="value",
        scaling=True,
        scaling_options={"method": "start_values", "clipping_value": 0.5},
    )

    aaae(internal.values, np.array([0, 1, 1]))
    aaae(internal.lower_bounds, np.array([-2, 0, 0.5]))
    aaae(internal.upper_bounds, np.array([2, 2, 1.5]))

    aaae(converter.params_to_internal(np.arange(3)), np.array([0, 1, 1]))
    aaae(converter.params_from_internal(np.array([0, 1, 1])), np.arange(3))
    aaae(
        converter.derivative_to_internal(2 * np.arange(3), np.arange(3)),
        np.array([0, 2, 8]),
    )
    aaae(converter.func_to_internal(3), 3)


def test_get_converter_with_trees():
    params = {"a": 0, "b": 1, "c": 2}
    converter, internal = get_converter(
        func=lambda x: (x**2).sum(),
        params=params,
        constraints=None,
        lower_bounds=None,
        upper_bounds=None,
        func_eval={"contributions": {"d": 1, "e": 2}},
        derivative_eval={"a": 0, "b": 2, "c": 4},
        primary_key="value",
        scaling=False,
        scaling_options=None,
    )

    aaae(internal.values, np.arange(3))
    aaae(internal.lower_bounds, np.full(3, -np.inf))
    aaae(internal.upper_bounds, np.full(3, np.inf))

    aaae(converter.params_to_internal(params), np.arange(3))
    assert converter.params_from_internal(np.arange(3)) == params
    aaae(
        converter.derivative_to_internal(params, np.arange(3)),
        np.arange(3),
    )
    aaae(converter.func_to_internal({"contributions": {"d": 1, "e": 2}}), 3)
