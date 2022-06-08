import itertools
import warnings

import numpy as np
import pytest
from estimagic import maximize
from estimagic.optimization import AVAILABLE_ALGORITHMS
from numpy.testing import assert_array_almost_equal as aaae


NLC_ALGORITHMS = [
    name
    for name, algo in AVAILABLE_ALGORITHMS.items()
    if "nonlinear_constraints" in algo._algorithm_info.arguments
]


@pytest.fixture()
def nlc_2d_example():
    """Non-linear constraints: 2-dimensional example.

    See the example section in https://en.wikipedia.org/wiki/Nonlinear_programming.

    """

    def criterion(x):
        return np.sum(x)

    def derivative(x):
        return np.ones_like(x)

    def constraint_func(x):
        value = np.dot(x, x)
        return np.array([value - 1, 2 - value])

    def constraint_jac(x):
        return 2 * np.row_stack((x.reshape(1, -1), -x.reshape(1, -1)))

    constraints_long = [
        {
            "type": "nonlinear",
            "func": constraint_func,
            "derivative": constraint_jac,
            "lower_bounds": np.zeros(2),
            "tol": 1e-8,
        }
    ]

    constraints_flat = [
        {
            "type": "nonlinear",
            "func": lambda x: np.dot(x, x),
            "derivative": lambda x: 2 * x,
            "lower_bounds": 1,
            "upper_bounds": 2,
            "tol": 1e-8,
        }
    ]

    constraints_equality = [
        {
            "type": "nonlinear",
            "func": lambda x: np.dot(x, x),
            "derivative": lambda x: 2 * x,
            "value": 2,
        }
    ]

    constraints_equality_and_inequality = [
        {
            "type": "nonlinear",
            "func": lambda x: np.dot(x, x),
            "derivative": lambda x: 2 * x,
            "lower_bounds": 1,
        },
        {
            "type": "nonlinear",
            "func": lambda x: np.dot(x, x),
            "derivative": lambda x: 2 * x,
            "value": 2,
        },
    ]

    _kwargs = {
        "criterion": criterion,
        "params": np.array([0, np.sqrt(2)]),
        "derivative": derivative,
        "lower_bounds": np.zeros(2),
        "upper_bounds": 2 * np.ones(2),
    }

    kwargs = {
        "flat": {**_kwargs, **{"constraints": constraints_flat}},
        "long": {**_kwargs, **{"constraints": constraints_long}},
        "equality": {**_kwargs, **{"constraints": constraints_equality}},
        "equality_and_inequality": {
            **_kwargs,
            **{"constraints": constraints_equality_and_inequality},
        },
    }

    solution_x = np.ones(2)

    return solution_x, kwargs


TEST_CASES = list(
    itertools.product(
        NLC_ALGORITHMS, ["flat", "long", "equality", "equality_and_inequality"]
    )
)


@pytest.mark.parametrize("algorithm, constr_type", TEST_CASES)
def test_nonlinear_optimization(nlc_2d_example, algorithm, constr_type):
    """Test that available nonlinear optimizers solve a nonlinear constraints problem.

    We test for the cases of "equality", "inequality" and "equality_and_inequality"
    constraints.

    """
    if "equality" in constr_type and algorithm == "nlopt_mma":
        pytest.skip(msg="Very slow and low accuracy.")

    solution_x, kwargs = nlc_2d_example
    if algorithm == "scipy_cobyla":
        del kwargs[constr_type]["lower_bounds"]
        del kwargs[constr_type]["upper_bounds"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = maximize(algorithm=algorithm, **kwargs[constr_type])

    if AVAILABLE_ALGORITHMS[algorithm]._algorithm_info.is_global:
        decimal = 0
    else:
        decimal = 4

    aaae(result.params, solution_x, decimal=decimal)
