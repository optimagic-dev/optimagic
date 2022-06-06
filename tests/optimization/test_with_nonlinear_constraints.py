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
        }
    ]

    constraints_flat = [
        {
            "type": "nonlinear",
            "fun": lambda x: np.dot(x, x),
            "jac": lambda x: 2 * x,
            "lower_bounds": 1,
            "upper_bounds": 2,
            "tol": 1e-7,
        }
    ]

    constraints = {"flat": constraints_flat, "long": constraints_long}

    def get_kwargs(algorithm, constr_type):

        kwargs = {
            "criterion": criterion,
            "params": np.array([0.1, 1.1]),
            "algorithm": algorithm,
            "constraints": constraints[constr_type],
            "derivative": derivative,
        }

        if algorithm != "scipy_cobyla":
            kwargs["lower_bounds"] = np.zeros(2)
            kwargs["upper_bounds"] = 10 * np.ones(2)

        return kwargs

    solution_x = np.ones(2)

    return get_kwargs, solution_x


@pytest.mark.parametrize(
    "algorithm, constr_type",
    itertools.product(NLC_ALGORITHMS, ["flat", "long"]),
)
def test_nonlinear_optimization(nlc_2d_example, algorithm, constr_type):
    get_kwargs, solution_x = nlc_2d_example
    kwargs = get_kwargs(algorithm, constr_type)
    result = maximize(**kwargs)
    if algorithm == "nlopt_slsqp":
        decimal = 1
    elif "nlopt" in algorithm:
        decimal = 0
    else:
        decimal = 5
    aaae(result.params, solution_x, decimal=decimal)
