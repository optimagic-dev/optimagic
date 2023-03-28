import numpy as np
import pytest
from estimagic import first_derivative, second_derivative
from estimagic.optimization.tranquilo.fit_models import _quadratic_features, get_fitter
from estimagic.optimization.tranquilo.region import Region
from numpy.testing import assert_array_almost_equal, assert_array_equal


def aaae(x, y, decimal=None, case=None):
    tolerance = {
        "hessian": 2,
        "gradient": 3,
    }
    decimal = decimal or tolerance.get(case, None)
    assert_array_almost_equal(x, y, decimal=decimal)


# ======================================================================================
# Fixtures
# ======================================================================================


@pytest.fixture()
def quadratic_case():
    """Test scenario with true quadratic function.

    We return true function, and function evaluations and data on random points.

    """
    n_params = 4
    n_samples = 15

    # theoretical terms
    linear_terms = 1 + np.arange(n_params)
    square_terms = np.arange(n_params**2).reshape(n_params, n_params)
    square_terms = square_terms + square_terms.T

    def func(x):
        y = -10 + linear_terms @ x + 0.5 * x.T @ square_terms @ x
        return y

    x0 = np.ones(n_params)

    # random data
    rng = np.random.default_rng(56789)
    x = np.array([x0 + rng.uniform(-0.01 * x0, 0.01 * x0) for _ in range(n_samples)])
    y = np.array([func(_x) for _x in list(x)]).reshape(-1, 1)

    out = {
        "func": func,
        "x0": x0,
        "x": x,
        "y": y,
        "linear_terms_expected": linear_terms,
        "square_terms_expected": square_terms,
    }
    return out


# ======================================================================================
# Tests
# ======================================================================================


@pytest.mark.parametrize("fitter", ["ols", "ridge", "powell", "tranquilo"])
def test_fit_against_truth_quadratic(fitter, quadratic_case):
    options = {"l2_penalty_square": 0}
    fit_pounders = get_fitter(
        fitter,
        options,
        model_type="quadratic",
        residualize=False,
        infinity_handling="relative",
    )
    got = fit_pounders(
        quadratic_case["x"],
        quadratic_case["y"],
        region=Region(center=np.zeros(4), radius=1.0),
        old_model=None,
    )
    decimal = 3 if fitter != "ridge" else 2
    aaae(
        got.linear_terms.flatten(),
        quadratic_case["linear_terms_expected"],
        decimal=decimal,
    )
    aaae(
        got.square_terms.reshape((4, 4)),
        quadratic_case["square_terms_expected"],
        decimal=decimal,
    )


@pytest.mark.parametrize("model", ["ols", "ridge", "tranquilo"])
def test_fit_ols_against_gradient(model, quadratic_case):
    options = {"l2_penalty_square": 0}
    fit_ols = get_fitter(
        model,
        options,
        model_type="quadratic",
        residualize=False,
        infinity_handling="relative",
    )
    got = fit_ols(
        quadratic_case["x"],
        quadratic_case["y"],
        region=Region(center=np.zeros(4), radius=1.0),
        old_model=None,
    )

    a = got.linear_terms.flatten()
    hess = got.square_terms.reshape((4, 4))
    grad = a + hess @ quadratic_case["x0"]

    gradient = first_derivative(quadratic_case["func"], quadratic_case["x0"])
    aaae(gradient["derivative"], grad, case="gradient")


@pytest.mark.parametrize("model", ("ols", "ridge", "tranquilo", "powell"))
def test_fit_ols_against_hessian(model, quadratic_case):
    options = {"l2_penalty_square": 0}
    fit_ols = get_fitter(
        model,
        options,
        model_type="quadratic",
        residualize=False,
        infinity_handling="relative",
    )
    got = fit_ols(
        quadratic_case["x"],
        quadratic_case["y"],
        region=Region(center=np.zeros(4), radius=1.0),
        old_model=None,
    )
    hessian = second_derivative(quadratic_case["func"], quadratic_case["x0"])
    hess = got.square_terms.reshape((4, 4))
    aaae(hessian["derivative"], hess, case="hessian")


def test_quadratic_features():
    x = np.array([[0, 1, 2], [3, 4, 5]])

    expected = np.array([[0, 1, 2, 0, 0, 0, 1, 2, 4], [3, 4, 5, 9, 12, 15, 16, 20, 25]])
    got = _quadratic_features(x)
    assert_array_equal(got, expected)
