import numpy as np
import pytest
from estimagic.optimization.cyipopt_optimizers import ipopt
from estimagic.optimization.scipy_optimizers import scipy_cobyla
from estimagic.optimization.scipy_optimizers import scipy_slsqp
from numpy.testing import assert_array_almost_equal as aaae


@pytest.fixture()
def nlc_2d_example():
    def criterion(x):
        return -np.sum(x)

    def derivative(x):
        return -np.ones_like(x)

    def constraint_func(x):
        value = np.dot(x, x)
        return np.array([value - 1, 2 - value])

    def constraint_jac(x):
        return 2 * np.row_stack((x.reshape(1, -1), -x.reshape(1, -1)))

    x = np.array([0, 1.1])

    constraints = [
        {
            "type": "ineq",
            "fun": constraint_func,
            "jac": constraint_jac,
            "args": (),
        }
    ]

    kwargs = {"criterion": criterion, "x": x, "constraints": constraints}

    full_kwargs = {
        **kwargs,
        **{
            "derivative": derivative,
            "lower_bounds": np.zeros(2),
            "upper_bounds": np.array([np.inf, np.inf]),
        },
    }

    # the expected / correct values
    solution_x = np.ones(2)

    return {
        "full_kwargs": full_kwargs,
        "kwargs": kwargs,
        "solution_x": solution_x,
    }


def test_scipy_slsqp(nlc_2d_example):

    full_kwargs = nlc_2d_example["full_kwargs"]
    kwargs = nlc_2d_example["kwargs"]
    solution_x = nlc_2d_example["solution_x"]

    res_ipopt = ipopt(**full_kwargs)
    res_slsqp = scipy_slsqp(**full_kwargs)
    res_cobyla = scipy_cobyla(**kwargs)

    aaae(res_ipopt["solution_x"], solution_x)
    aaae(res_slsqp["solution_x"], solution_x)
    aaae(res_cobyla["solution_x"], solution_x, decimal=5)
