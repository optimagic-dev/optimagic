import numpy as np
import pytest
from estimagic.optimization.cyipopt_optimizers import ipopt
from estimagic.optimization.scipy_optimizers import scipy_cobyla
from estimagic.optimization.scipy_optimizers import scipy_slsqp
from estimagic.optimization.scipy_optimizers import scipy_trust_constr
from numpy.testing import assert_array_almost_equal as aaae


@pytest.fixture()
def nlc_2d_example():
    """Non-linear constraints: 2-dimensional example."""

    def criterion(x):
        return -np.sum(x)

    def derivative(x):
        return -np.ones_like(x)

    def criterion_and_derivative(x):
        return criterion(x), derivative(x)

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
        "x": x,
        "constraints": constraints,
        "criterion_and_derivative": criterion_and_derivative,
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
    del kwargs["criterion_and_derivative"]
    del kwargs["constraints"][0]["hess"]  # not implemneted or what?
    solution_x = nlc_2d_example["solution_x"]
    result = ipopt(**kwargs)
    aaae(result["solution_x"], solution_x)


def test_scipy_slsqp(nlc_2d_example):
    kwargs = nlc_2d_example["full_kwargs"]
    del kwargs["criterion_and_derivative"]
    solution_x = nlc_2d_example["solution_x"]
    result = scipy_slsqp(**kwargs)
    aaae(result["solution_x"], solution_x)


def test_scipy_cobyla(nlc_2d_example):
    kwargs = nlc_2d_example["full_kwargs"]
    for delete in [
        "criterion_and_derivative",
        "derivative",
        "lower_bounds",
        "upper_bounds",
    ]:
        del kwargs[delete]
    solution_x = nlc_2d_example["solution_x"]
    result = scipy_cobyla(**kwargs)
    aaae(result["solution_x"], solution_x, decimal=5)


def test_scipy_trust_constr(nlc_2d_example):
    kwargs = nlc_2d_example["full_kwargs"]
    del kwargs["criterion"]
    del kwargs["derivative"]
    solution_x = nlc_2d_example["solution_x"]
    res_trust_constr = scipy_trust_constr(**kwargs)
    aaae(res_trust_constr["solution_x"], solution_x)
