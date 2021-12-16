from functools import partial

import numpy as np
import pandas as pd
import pytest
from estimagic.config import TEST_FIXTURES_DIR
from estimagic.optimization.history import LeastSquaresHistory
from estimagic.optimization.pounders_auxiliary import add_more_points
from estimagic.optimization.pounders_auxiliary import find_affine_points
from estimagic.optimization.pounders_auxiliary import get_approximation_error
from estimagic.optimization.pounders_auxiliary import get_coefficients_residual_model
from estimagic.optimization.pounders_auxiliary import improve_main_model
from estimagic.optimization.pounders_auxiliary import update_main_from_residual_model
from estimagic.optimization.pounders_auxiliary import update_residual_model
from numpy.testing import assert_array_almost_equal as aaae


@pytest.fixture
def criterion():
    data = pd.read_csv(TEST_FIXTURES_DIR / "pounders_example_data.csv")
    endog = np.asarray(data["y"])
    exog = np.asarray(data["t"])

    def func(x: np.ndarray, exog: np.ndarray, endog: np.ndarray) -> np.ndarray:
        """User provided residual function."""
        return endog - np.exp(-x[0] * exog) / (x[1] + x[2] * exog)

    return partial(func, exog=exog, endog=endog)


@pytest.fixture(
    params=[
        "model_improving_points_zero_i",
        "model_improving_points_zero_ii",
        "model_improving_points_zero_iii",
        "model_improving_points_zero_iv",
        "model_improving_points_nonzero_i",
        "model_improving_points_nonzero_ii",
        "model_improving_points_nonzero_iii",
    ]
)
def dict_find_affine_points(request):
    return pd.read_pickle(TEST_FIXTURES_DIR / f"find_affine_points_{request.param}.pkl")


@pytest.fixture(params=["4", "7"])
def dict_get_approximation_error(request):
    return pd.read_pickle(
        TEST_FIXTURES_DIR / f"get_approximation_error_iter_{request.param}.pkl"
    )


@pytest.fixture(params=["i", "ii"])
def dict_improve_model(request):
    return pd.read_pickle(TEST_FIXTURES_DIR / f"improve_model_{request.param}.pkl")


@pytest.fixture
def dict_add_more_points():
    return pd.read_pickle(TEST_FIXTURES_DIR / "add_more_points.pkl")


@pytest.fixture
def dict_get_coefficients_residual_model():
    return pd.read_pickle(TEST_FIXTURES_DIR / "get_params_quadratic_model.pkl")


@pytest.fixture
def dict_update_residual_model():
    return pd.read_pickle(TEST_FIXTURES_DIR / "update_gradient_and_hessian.pkl")


@pytest.fixture
def dict_update_main_from_residual_model():
    return pd.read_pickle(TEST_FIXTURES_DIR / "calc_first_and_second_derivative.pkl")


@pytest.mark.skip(reason="refactoring")
def test_find_affine_points(dict_find_affine_points):
    (
        model_improving_points_out,
        model_indices_out,
        n_modelpoints_out,
        project_x_onto_null_out,
    ) = find_affine_points(
        history_x=dict_find_affine_points["history_x"],
        min_x=dict_find_affine_points["min_x"],
        model_improving_points=dict_find_affine_points["model_improving_points"],
        project_x_onto_null=dict_find_affine_points["project_x_onto_null"],
        delta=dict_find_affine_points["delta"],
        c=dict_find_affine_points["c"],
        model_indices=dict_find_affine_points["model_indices"],
        n_modelpoints=dict_find_affine_points["n_modelpoints"],
        n_history=dict_find_affine_points["n_history"],
        theta1=1e-5,
        n=3,
    )

    aaae(
        model_improving_points_out,
        dict_find_affine_points["model_improving_points_expected"],
    )
    aaae(model_indices_out, dict_find_affine_points["model_indices_expected"])
    assert np.allclose(
        n_modelpoints_out, dict_find_affine_points["n_modelpoints_expected"]
    )
    assert np.allclose(project_x_onto_null_out, True)


@pytest.mark.skip(reason="refactoring")
def test_improve_model(dict_improve_model, criterion):
    history = LeastSquaresHistory()
    (
        _,
        history_x_out,
        history_criterion_out,
        _,
        model_indices_out,
        n_modelpoints_out,
        n_history_out,
    ) = improve_main_model(
        history,
        history_x=dict_improve_model["history_x"],
        history_criterion=dict_improve_model["history_criterion"],
        history_criterion_norm=dict_improve_model["history_criterion_norm"],
        first_derivative=dict_improve_model["first_derivative"],
        second_derivative=dict_improve_model["second_derivative"],
        model_improving_points=dict_improve_model["model_improving_points"],
        model_indices=dict_improve_model["model_indices"],
        index_min_x=dict_improve_model["index_min_x"],
        n_modelpoints=dict_improve_model["n_modelpoints"],
        n_history=dict_improve_model["n_history"],
        delta=dict_improve_model["delta"],
        lower_bounds=None,
        upper_bounds=None,
        add_all_points=1,
        n=3,
        criterion=criterion,
    )

    aaae(history_x_out, dict_improve_model["history_x_expected"])
    aaae(history_criterion_out, dict_improve_model["history_criterion_expected"])
    aaae(model_indices_out, dict_improve_model["model_indices_expected"])
    assert np.allclose(n_modelpoints_out, dict_improve_model["n_modelpoints_expected"])
    assert np.allclose(n_history_out, dict_improve_model["n_history_expected"])


@pytest.mark.skip(reason="refactoring")
def test_add_more_points(dict_add_more_points):
    n = 3
    n_maxinterp = 2 * n + 1

    (
        lower_triangular,
        basis_null_space,
        monomial_basis,
        interpolation_set,
        n_modelpoints,
    ) = add_more_points(
        history_x=dict_add_more_points["history_x"],
        min_x=dict_add_more_points["min_x"],
        model_indices=dict_add_more_points["model_indices"],
        index_min_x=dict_add_more_points["index_min_x"],
        delta=dict_add_more_points["delta"],
        n_modelpoints=dict_add_more_points["n_modelpoints"],
        n_history=dict_add_more_points["n_history"],
        c2=10,
        theta2=1e-4,
        n=n,
        n_maxinterp=n_maxinterp,
    )

    aaae(lower_triangular, dict_add_more_points["lower_triangular_expected"])
    aaae(basis_null_space, dict_add_more_points["basis_null_space_expected"])
    aaae(monomial_basis, dict_add_more_points["monomial_basis_expected"])
    aaae(interpolation_set, dict_add_more_points["interpolation_set_expected"])
    assert np.allclose(n_modelpoints, dict_add_more_points["n_modelpoints_expected"])


@pytest.mark.skip(reason="refactoring")
def test_get_approximation_error(dict_get_approximation_error):
    history_x = dict_get_approximation_error["history_x"]
    min_x = dict_get_approximation_error["min_x"]
    model_indices = dict_get_approximation_error["model_indices"]
    n_modelpoints = dict_get_approximation_error["n_modelpoints"]
    delta_old = dict_get_approximation_error["delta_old"]

    n = 3
    n_obs = 214
    n_maxinterp = 2 * n + 1

    xk = (history_x[model_indices[:n_modelpoints]] - min_x) / delta_old

    approximiation_error = get_approximation_error(
        xk=xk,
        hessian=dict_get_approximation_error["hessian"],
        history_criterion=dict_get_approximation_error["history_criterion"],
        min_criterion=dict_get_approximation_error["min_criterion"],
        gradient=dict_get_approximation_error["gradient"],
        model_indices=model_indices,
        n_modelpoints=n_modelpoints,
        n_obs=n_obs,
        n_maxinterp=n_maxinterp,
    )

    aaae(xk, dict_get_approximation_error["xk"])
    aaae(
        approximiation_error,
        dict_get_approximation_error["approximation_error_expected"],
    )


def test_get_coefficients_residual_model(dict_get_coefficients_residual_model):
    coefficients_to_add = get_coefficients_residual_model(
        lower_triangular=dict_get_coefficients_residual_model["lower_triangular"],
        basis_null_space=dict_get_coefficients_residual_model["basis_null_space"],
        monomial_basis=dict_get_coefficients_residual_model["monomial_basis"],
        interpolation_set=dict_get_coefficients_residual_model["interpolation_set"],
        approximation_error=dict_get_coefficients_residual_model["approximation_error"],
        n_modelpoints=dict_get_coefficients_residual_model["n_modelpoints"],
        n=3,
        n_obs=214,
    )

    aaae(
        coefficients_to_add["linear_terms"],
        dict_get_coefficients_residual_model["params_gradient_expected"].T,
    )
    aaae(
        coefficients_to_add["square_terms"],
        dict_get_coefficients_residual_model["params_hessian_expected"],
    )


def test_update_residual_model(dict_update_residual_model):
    residual_model = {
        "linear_terms": dict_update_residual_model["gradient"],
        "square_terms": dict_update_residual_model["hessian"],
    }
    coefficients_to_add = {
        "linear_terms": dict_update_residual_model["params_gradient"].T,
        "square_terms": dict_update_residual_model["params_hessian"],
    }

    residual_model = update_residual_model(
        residual_model=residual_model,
        coefficients_to_add=coefficients_to_add,
        delta=dict_update_residual_model["delta"],
        delta_old=dict_update_residual_model["delta_old"],
    )

    aaae(
        residual_model["linear_terms"],
        dict_update_residual_model["gradient_expected"],
    )
    aaae(
        residual_model["square_terms"],
        dict_update_residual_model["hessian_expected"],
    )


def test_update_main_from_residual_model(dict_update_main_from_residual_model):
    residual_model = {
        "intercepts": dict_update_main_from_residual_model["min_criterion"],
        "linear_terms": dict_update_main_from_residual_model["gradient"],
        "square_terms": dict_update_main_from_residual_model["hessian"],
    }
    main_model = update_main_from_residual_model(residual_model, first_evaluation=False)

    aaae(
        main_model["linear_terms"],
        dict_update_main_from_residual_model["first_derivative_expected"],
    )
    aaae(
        main_model["square_terms"],
        dict_update_main_from_residual_model["second_derivative_expected"],
        decimal=3,
    )
