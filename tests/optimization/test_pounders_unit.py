from functools import partial

import numpy as np
import pandas as pd
import pytest
from estimagic.config import TEST_FIXTURES_DIR
from estimagic.optimization.pounders_auxiliary import add_more_points
from estimagic.optimization.pounders_auxiliary import calc_first_and_second_derivative
from estimagic.optimization.pounders_auxiliary import find_affine_points
from estimagic.optimization.pounders_auxiliary import get_approximation_error
from estimagic.optimization.pounders_auxiliary import get_params_quadratic_model
from estimagic.optimization.pounders_auxiliary import improve_model
from estimagic.optimization.pounders_auxiliary import update_center
from estimagic.optimization.pounders_auxiliary import update_gradient_and_hessian
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
def dict_update_center():
    return pd.read_pickle(TEST_FIXTURES_DIR / "update_center.pkl")


@pytest.fixture
def dict_add_more_points():
    return pd.read_pickle(TEST_FIXTURES_DIR / "add_more_points.pkl")


@pytest.fixture
def dict_get_params_quadratic_model():
    return pd.read_pickle(TEST_FIXTURES_DIR / "get_params_quadratic_model.pkl")


@pytest.fixture
def dict_update_gradient_and_hessian():
    return pd.read_pickle(TEST_FIXTURES_DIR / "update_gradient_and_hessian.pkl")


@pytest.fixture
def dict_calc_first_and_second_derivative():
    return pd.read_pickle(TEST_FIXTURES_DIR / "calc_first_and_second_derivative.pkl")


def test_update_center(dict_update_center):
    (
        min_x_out,
        min_criterion_out,
        gradient_out,
        min_criterion_norm_out,
        first_derivative_out,
        index_min_x_out,
    ) = update_center(
        xplus=dict_update_center["xplus"],
        min_x=dict_update_center["min_x"],
        history_x=dict_update_center["history_x"],
        delta=dict_update_center["delta"],
        min_criterion=dict_update_center["min_criterion"],
        gradient=dict_update_center["gradient"],
        history_criterion_norm=dict_update_center["history_criterion_norm"],
        hessian=dict_update_center["hessian"],
        first_derivative=dict_update_center["first_derivative"],
        second_derivative=dict_update_center["second_derivative"],
        n_history=dict_update_center["n_history"],
    )
    aaae(min_x_out, dict_update_center["min_x_expected"])
    aaae(min_criterion_out, dict_update_center["min_criterion_expected"])
    aaae(gradient_out, dict_update_center["gradient_expected"])
    aaae(min_criterion_norm_out, dict_update_center["min_criterion_norm_expected"])
    aaae(
        first_derivative_out, dict_update_center["first_derivative_expected"], decimal=5
    )
    assert np.allclose(index_min_x_out, dict_update_center["index_min_x_expected"])


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


def test_improve_model(dict_improve_model, criterion):
    (
        history_x_out,
        history_criterion_out,
        _,
        model_indices_out,
        n_modelpoints_out,
        n_history_out,
    ) = improve_model(
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


def test_get_params_quadratic_model(dict_get_params_quadratic_model):
    params_gradient, params_hessian = get_params_quadratic_model(
        lower_triangular=dict_get_params_quadratic_model["lower_triangular"],
        basis_null_space=dict_get_params_quadratic_model["basis_null_space"],
        monomial_basis=dict_get_params_quadratic_model["monomial_basis"],
        interpolation_set=dict_get_params_quadratic_model["interpolation_set"],
        approximation_error=dict_get_params_quadratic_model["approximation_error"],
        n_modelpoints=dict_get_params_quadratic_model["n_modelpoints"],
        n=3,
        n_obs=214,
    )

    aaae(params_gradient, dict_get_params_quadratic_model["params_gradient_expected"])
    aaae(params_hessian, dict_get_params_quadratic_model["params_hessian_expected"])


def test_update_gradient_and_hessian(dict_update_gradient_and_hessian):
    gradient_out, hessian_out = update_gradient_and_hessian(
        gradient=dict_update_gradient_and_hessian["gradient"],
        hessian=dict_update_gradient_and_hessian["hessian"],
        params_gradient=dict_update_gradient_and_hessian["params_gradient"],
        params_hessian=dict_update_gradient_and_hessian["params_hessian"],
        delta=dict_update_gradient_and_hessian["delta"],
        delta_old=dict_update_gradient_and_hessian["delta_old"],
    )

    aaae(gradient_out, dict_update_gradient_and_hessian["gradient_expected"])
    aaae(hessian_out, dict_update_gradient_and_hessian["hessian_expected"])


def test_calc_first_and_second_derivative(dict_calc_first_and_second_derivative):
    first_derivative, second_derivative = calc_first_and_second_derivative(
        gradient=dict_calc_first_and_second_derivative["gradient"],
        min_criterion=dict_calc_first_and_second_derivative["min_criterion"],
        hessian=dict_calc_first_and_second_derivative["hessian"],
    )

    aaae(
        first_derivative,
        dict_calc_first_and_second_derivative["first_derivative_expected"],
    )
    aaae(
        second_derivative,
        dict_calc_first_and_second_derivative["second_derivative_expected"],
        decimal=3,
    )
