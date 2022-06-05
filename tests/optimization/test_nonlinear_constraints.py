import numpy as np
import pytest
from estimagic import minimize
from numpy.testing import assert_array_almost_equal as aaae


@pytest.fixture()
def nlc_2d_example():
    """Non-linear constraints: 2-dimensional example."""

    def criterion(x):
        return -np.sum(x)

    def derivative(x):
        return -np.ones_like(x)

    def constraint_func(x):
        value = np.dot(x, x)
        return np.array([value - 1, 2 - value])

    def constraint_jac(x):
        return 2 * np.row_stack((x.reshape(1, -1), -x.reshape(1, -1)))

    def constraint_hess(x, v):
        batch_hess = 2 * np.stack((np.eye(len(x)), -np.eye(len(x))))
        return np.dot(batch_hess.T, v)

    x = np.array([0, 1.1])

    constraints = [
        {
            "type": "ineq",
            "fun": constraint_func,
            "jac": constraint_jac,
            "hess": constraint_hess,
            "n_constr": 2,
        }
    ]

    full_kwargs = {
        "criterion": criterion,
        "params": x,
        "algo_options": {"constraints": constraints},
        "derivative": derivative,
        "lower_bounds": np.zeros(2),
        "upper_bounds": np.array([np.inf, np.inf]),
    }

    solution_x = np.ones(2)

    return {
        "full_kwargs": full_kwargs,
        "solution_x": solution_x,
    }


def test_ipopt(nlc_2d_example):
    kwargs = nlc_2d_example["full_kwargs"]
    kwargs["algorithm"] = "ipopt"
    del kwargs["algo_options"]["constraints"][0]["hess"]
    solution_x = nlc_2d_example["solution_x"]
    result = minimize(**kwargs)
    aaae(result.params, solution_x)


def test_scipy_slsqp(nlc_2d_example):
    kwargs = nlc_2d_example["full_kwargs"]
    kwargs["algorithm"] = "scipy_slsqp"
    solution_x = nlc_2d_example["solution_x"]
    result = minimize(**kwargs)
    aaae(result.params, solution_x)


def test_scipy_cobyla(nlc_2d_example):
    kwargs = nlc_2d_example["full_kwargs"]
    kwargs["algorithm"] = "scipy_cobyla"
    del kwargs["lower_bounds"]
    del kwargs["upper_bounds"]
    solution_x = nlc_2d_example["solution_x"]
    result = minimize(**kwargs)
    aaae(result.params, solution_x, decimal=5)


def test_scipy_trust_constr(nlc_2d_example):
    kwargs = nlc_2d_example["full_kwargs"]
    kwargs["algorithm"] = "scipy_trust_constr"
    solution_x = nlc_2d_example["solution_x"]
    result = minimize(**kwargs)
    aaae(result.params, solution_x)
