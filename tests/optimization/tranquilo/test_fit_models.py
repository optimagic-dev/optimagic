import numpy as np
import pytest
from estimagic import first_derivative
from estimagic import second_derivative
from estimagic.optimization.tranquilo.fit_models import _polynomial_features
from estimagic.optimization.tranquilo.fit_models import get_fitter
from estimagic.optimization.tranquilo.models import ModelInfo
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_array_equal


def aaae(x, y, case=None):
    tolerance = {
        None: 3,
        "hessian": 2,
        "gradient": 3,
    }
    assert_array_almost_equal(x, y, decimal=tolerance[case])


@pytest.fixture
def quadratic_case():
    """Test scenario with true quadratic function.

    We return true function, and function evaluations and data on random points.
    """
    n_params = 4
    n_samples = 2_000

    # theoretical terms
    linear_terms = 1 + np.arange(n_params)
    square_terms = np.arange(n_params**2).reshape(n_params, n_params)
    square_terms = square_terms + square_terms.T

    def func(x):
        y = -10 + linear_terms @ x + 0.5 * x.T @ square_terms @ x
        return y

    x0 = np.ones(n_params)

    # random data
    x = np.array(
        [x0 + np.random.uniform(-0.01 * x0, 0.01 * x0) for _ in range(n_samples)]
    )
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


@pytest.fixture
def just_identified_case():
    """Test scenario with true quadratic function and n + 1 points.

    We return true function, and function evaluations and data on random points.
    """
    n_params = 4
    n_samples = n_params + 1

    # theoretical terms
    linear_terms = 1 + np.arange(n_params)
    square_terms = np.zeros((n_params, n_params))

    def func(x):
        y = -10 + linear_terms @ x + 0.5 * x.T @ square_terms @ x
        return y

    x0 = np.ones(n_params)

    # random data
    x = np.array(
        [x0 + np.random.uniform(-0.01 * x0, 0.01 * x0) for _ in range(n_samples)]
    )
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


def test_fit_ols_against_truth(quadratic_case):
    fit_ols = get_fitter("ols")
    got = fit_ols(quadratic_case["x"], quadratic_case["y"])

    aaae(got.linear_terms.squeeze(), quadratic_case["linear_terms_expected"])
    aaae(got.square_terms.squeeze(), quadratic_case["square_terms_expected"])


@pytest.mark.parametrize("scenario", ["just_identified_case", "quadratic_case"])
def test_fit_pounders_against_truth(scenario, request):
    test_case = request.getfixturevalue(scenario)

    model_info = ModelInfo(has_intercepts=True, has_squares=True, has_interactions=True)
    fit_pounders = get_fitter("pounders", model_info=model_info)
    got = fit_pounders(test_case["x"], test_case["y"])

    aaae(got.linear_terms.squeeze(), test_case["linear_terms_expected"])
    aaae(got.square_terms.squeeze(), test_case["square_terms_expected"])


@pytest.mark.parametrize("scenario", ["just_identified_case", "quadratic_case"])
def test_fit_original_pounders_against_truth(scenario, request):
    test_case = request.getfixturevalue(scenario)

    model_info = ModelInfo(has_intercepts=True, has_squares=True, has_interactions=True)
    fit_pounders = get_fitter("pounders_original", model_info=model_info)
    got = fit_pounders(test_case["x"], test_case["y"])

    aaae(got.linear_terms.squeeze(), test_case["linear_terms_expected"])
    aaae(got.square_terms.squeeze(), test_case["square_terms_expected"])


# =====================================================================================


@pytest.mark.parametrize("model", ["ols", "ridge"])
def test_fit_ols_against_gradient(model, quadratic_case):
    if model == "ridge":
        options = {"l2_penalty_square": 0}
    else:
        options = None

    fit_ols = get_fitter(model, options)
    got = fit_ols(quadratic_case["x"], quadratic_case["y"])

    a = got.linear_terms.squeeze()
    hess = got.square_terms.squeeze()
    grad = a + hess @ quadratic_case["x0"]

    gradient = first_derivative(quadratic_case["func"], quadratic_case["x0"])
    aaae(gradient["derivative"], grad, case="gradient")


@pytest.mark.parametrize(
    "model, options",
    [("ols", None), ("ridge", {"l2_penalty_linear": 0, "l2_penalty_square": 0})],
)
def test_fit_ols_against_hessian(model, options, quadratic_case):
    fit_ols = get_fitter(model, options)
    got = fit_ols(quadratic_case["x"], quadratic_case["y"])
    hessian = second_derivative(quadratic_case["func"], quadratic_case["x0"])
    hess = got.square_terms.squeeze()
    aaae(hessian["derivative"], hess, case="hessian")


@pytest.mark.parametrize("has_intercepts, has_squares", [(True, False), (True, False)])
def test_polynomial_features(has_intercepts, has_squares):

    x = np.array([[0, 1, 2], [3, 4, 5]])

    expected = {
        # (has_intercepts, has_squares): expected value,
        (True, True): np.array(
            [[1, 0, 1, 2, 0, 0, 0, 1, 2, 4], [1, 3, 4, 5, 9, 12, 15, 16, 20, 25]]
        ),
        (True, False): np.array([[1, 0, 1, 2, 0, 0, 2], [1, 3, 4, 5, 12, 15, 20]]),
        (False, True): np.array(
            [[0, 1, 2, 0, 0, 0, 1, 2, 4], [3, 4, 5, 9, 12, 15, 16, 20, 25]]
        ),
        (False, False): np.array([[0, 1, 2, 0, 0, 2], [3, 4, 5, 12, 15, 20]]),
    }

    got = _polynomial_features(
        x, has_intercepts=has_intercepts, has_squares=has_squares
    )

    assert_array_equal(got, expected[(has_intercepts, has_squares)])
