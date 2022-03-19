import numpy as np
import pytest
from estimagic import first_derivative
from estimagic import second_derivative
from estimagic.optimization.tranquillo.surrogate_models import fit_ols
from numpy.testing import assert_array_almost_equal


def aaae(x, y):
    assert_array_almost_equal(x, y, decimal=3)


@pytest.fixture
def quadratic_case():
    """Test scenario with true quadtratic function.

    We return true function, and function evaluations and data on random points.

    """

    def func(x):
        coef_linear = 1 + np.arange(len(x))
        coef_square = -1 - np.arange(len(x))
        interaction = (
            20 * x[0] * x[1]
            + 25 * x[1] * x[2]
            + 30 * x[2] * x[3]
            + 35 * x[0] * x[2]
            + 40 * x[1] * x[3]
            + 45 * x[0] * x[3]
        )
        y = -10 + coef_linear @ x + coef_square @ (x**2) + interaction
        return y

    # data
    x0 = np.ones(4)

    # theoretical terms
    linear_terms = 1 + np.arange(4)
    square_terms = np.array(
        [[-1, 0, 0, 0], [20, -2, 0, 0], [35, 25, -3, 0], [45, 40, 30, -4]]
    )

    # random data
    x = np.array([x0 + np.random.uniform(-0.01 * x0, 0.01 * x0) for _ in range(10_000)])
    y = np.array([func(u) for u in list(x)]).reshape(-1, 1)

    out = {
        "func": func,
        "x0": x0,
        "x": x,
        "y": y,
        "linear_terms_expected": linear_terms,
        "square_terms_expected": square_terms,
    }
    return out


def test_fit_ols_against_truth(quadratic_case):
    got = fit_ols(quadratic_case["x"], quadratic_case["y"])

    aaae(got["linear_terms"].squeeze(), quadratic_case["linear_terms_expected"])
    aaae(got["square_terms"].squeeze(), quadratic_case["square_terms_expected"])


def test_fit_ols_against_gradient(quadratic_case):
    got = fit_ols(quadratic_case["x"], quadratic_case["y"])

    a = got["linear_terms"].squeeze()
    b = (got["square_terms"] + got["square_terms"].transpose(1, 0, 2)).squeeze()
    grad = a + b @ quadratic_case["x0"]

    gradient = first_derivative(quadratic_case["func"], quadratic_case["x0"])
    aaae(gradient["derivative"], grad)


def test_fit_ols_against_hessian(quadratic_case):
    got = fit_ols(quadratic_case["x"], quadratic_case["y"])

    hessian = second_derivative(quadratic_case["func"], quadratic_case["x0"])

    tril = got["square_terms"]
    hess = tril + tril.transpose(1, 0, 2)
    aaae(hessian["derivative"], hess.squeeze())
