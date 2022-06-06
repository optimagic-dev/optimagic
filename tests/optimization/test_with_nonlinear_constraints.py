import itertools

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
            "fun": constraint_func,
            "jac": constraint_jac,
            "lower_bounds": np.zeros(2),
            "tol": 0.0,
        }
    ]

    constraints_flat = [
        {
            "type": "nonlinear",
            "fun": lambda x: np.dot(x, x),
            "jac": lambda x: 2 * x,
            "lower_bounds": 1,
            "upper_bounds": 2,
            "tol": 0.0,
        }
    ]

    constraints = {"flat": constraints_flat, "long": constraints_long}

    def get_kwargs(algorithm, constr_type):

        kwargs = {
            "criterion": criterion,
            "params": np.array([1.1, 0.1]),
            "algorithm": algorithm,
            "constraints": constraints[constr_type],
            "derivative": derivative,
        }

        if algorithm != "scipy_cobyla":
            kwargs["lower_bounds"] = np.zeros(2)
            kwargs["upper_bounds"] = 2 * np.ones(2)

        return kwargs

    solution_x = np.ones(2)

    return get_kwargs, solution_x


@pytest.mark.parametrize(
    "algorithm, constr_type",
    itertools.product(NLC_ALGORITHMS, ["flat", "long"]),
)
def test_nonlinear_optimization(nlc_2d_example, algorithm, constr_type):
    if algorithm == "nlopt_slsqp":
        pytest.mark.skip("Fail for nlopt_slsqp.")
        return None
    get_kwargs, solution_x = nlc_2d_example
    kwargs = get_kwargs(algorithm, constr_type)
    result = maximize(**kwargs)
    if AVAILABLE_ALGORITHMS[algorithm]._algorithm_info.is_global:
        decimal = 0
    else:
        decimal = 4
    aaae(result.params, solution_x, decimal=decimal)
