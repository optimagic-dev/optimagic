import itertools

import numpy as np
import pytest
from estimagic import maximize
from estimagic.config import IS_CYIPOPT_INSTALLED
from numpy.testing import assert_array_almost_equal as aaae


@pytest.fixture()
def nlc_2d_example():
    """Non-linear constraints: 2-dimensional example."""

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
        }
    ]

    constraints = {"flat": constraints_flat, "long": constraints_long}

    def get_kwargs(algorithm, constr_type):

        kwargs = {
            "criterion": criterion,
            "params": np.array([0.1, 1.1]),  # start params
            "constraints": constraints[constr_type],
            "derivative": derivative,
            "algorithm": algorithm,
        }

        if algorithm != "scipy_cobyla":
            kwargs["lower_bounds"] = np.zeros(2)

        return kwargs

    solution_x = np.ones(2)

    return get_kwargs, solution_x


@pytest.mark.parametrize(
    "algorithm, constr_type",
    itertools.product(
        ["scipy_slsqp", "scipy_cobyla", "scipy_trust_constr", "ipopt"], ["flat", "long"]
    ),
)
def test_nonlinear_optimization(nlc_2d_example, algorithm, constr_type):
    if algorithm == "ipopt" and not IS_CYIPOPT_INSTALLED:
        pytest.skip(msg="cyipopt not installed.")
    get_kwargs, solution_x = nlc_2d_example
    kwargs = get_kwargs(algorithm, constr_type)
    result = maximize(**kwargs)
    aaae(result.params, solution_x, decimal=5)
