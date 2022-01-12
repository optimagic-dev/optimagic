from functools import partial

import numpy as np
import pandas as pd
import pytest
from estimagic.config import TEST_FIXTURES_DIR
from estimagic.optimization.history import LeastSquaresHistory
from estimagic.optimization.pounders_auxiliary import add_more_points
from estimagic.optimization.pounders_auxiliary import find_affine_points
from estimagic.optimization.pounders_auxiliary import get_coefficients_residual_model
from estimagic.optimization.pounders_auxiliary import improve_main_model
from estimagic.optimization.pounders_auxiliary import interpolate_f
from estimagic.optimization.pounders_auxiliary import update_initial_residual_model
from estimagic.optimization.pounders_auxiliary import update_main_from_residual_model
from estimagic.optimization.pounders_auxiliary import (
    update_main_model_with_new_accepted_x,
)
from estimagic.optimization.pounders_auxiliary import update_residual_model
from estimagic.optimization.pounders_auxiliary import (
    update_residual_model_with_new_accepted_x,
)
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


@pytest.fixture(params=["4", "7"])
def dicts_interpolate_f(request):
    dict_ = pd.read_pickle(
        TEST_FIXTURES_DIR / f"get_approximation_error_iter_{request.param}.pkl"
    )

    history = LeastSquaresHistory()
    history.add_entries(
        dict_["history_x"],
        dict_["history_criterion"],
    )

    n = 3
    n_obs = 214
    n_maxinterp = 2 * n + 1
    x_accepted = dict_["min_x"]
    model_indices = dict_["model_indices"]
    n_modelpoints = dict_["n_modelpoints"]
    delta_old = dict_["delta_old"]

    center_info = {"x": x_accepted, "radius": delta_old}
    interpolation_set = history.get_centered_xs(
        center_info, index=model_indices[:n_modelpoints]
    )

    residual_model = {
        "intercepts": dict_["min_criterion"],
        "linear_terms": dict_["gradient"],
        "square_terms": dict_["hessian"],
    }

    inputs_dict = {
        "history": history,
        "interpolation_set": interpolation_set,
        "residual_model": residual_model,
        "model_indices": model_indices,
        "n_modelpoints": n_modelpoints,
        "n_obs": n_obs,
        "n_maxinterp": n_maxinterp,
    }

    expected_dict = {
        "interpolation_set_expected": dict_["xk"],
        "f_interpolated_expected": dict_["approximation_error_expected"],
    }

    return inputs_dict, expected_dict


@pytest.fixture
def dicts_add_more_points():
    dict_ = pd.read_pickle(TEST_FIXTURES_DIR / "add_more_points.pkl")

    history = LeastSquaresHistory()
    history.add_entries(
        dict_["history_x"],
        np.zeros(dict_["history_x"].shape),
    )

    n = 3
    n_maxinterp = 2 * n + 1
    x_accepted = dict_["min_x"]
    model_indices = dict_["model_indices"]
    delta = dict_["delta"]
    n_modelpoints = dict_["n_modelpoints"]
    c2 = 10
    theta2 = 1e-4

    inputs_dict = {
        "history": history,
        "x_accepted": x_accepted,
        "model_indices": model_indices,
        "delta": delta,
        "c2": c2,
        "theta2": theta2,
        "n": n,
        "n_maxinterp": n_maxinterp,
        "n_modelpoints": n_modelpoints,
    }

    expected_dict = {
        "lower_triangular_expected": dict_["lower_triangular_expected"],
        "basis_null_space_expected": dict_["basis_null_space_expected"],
        "monomial_basis_expected": dict_["monomial_basis_expected"],
        "x_sample_monomial_basis_expected": dict_["interpolation_set_expected"],
        "n_modelpoints_expected": dict_["n_modelpoints_expected"],
    }

    return inputs_dict, expected_dict


@pytest.fixture
def dict_get_coefficients_residual_model():
    return pd.read_pickle(TEST_FIXTURES_DIR / "get_params_quadratic_model.pkl")


@pytest.fixture
def dict_update_residual_model():
    return pd.read_pickle(TEST_FIXTURES_DIR / "update_gradient_and_hessian.pkl")


@pytest.fixture
def dict_update_main_from_residual_model():
    return pd.read_pickle(TEST_FIXTURES_DIR / "calc_first_and_second_derivative.pkl")


@pytest.fixture(params=["i", "ii"])
def dicts_improve_model(request, criterion):
    history = LeastSquaresHistory()
    dict_ = pd.read_pickle(TEST_FIXTURES_DIR / f"improve_model_{request.param}.pkl")

    n = 3
    n_modelpoints = dict_["n_modelpoints"]
    history.add_entries(
        dict_["history_x"][: -(n - n_modelpoints)],
        dict_["history_criterion"][: -(n - n_modelpoints)],
    )

    index_min_x = dict_["index_min_x"]
    x_accepted = dict_["history_x"][index_min_x]
    delta = dict_["delta"]
    main_model = {
        "linear_terms": dict_["first_derivative"],
        "square_terms": dict_["second_derivative"],
    }
    model_improving_points = dict_["model_improving_points"]
    model_indices = dict_["model_indices"]

    inputs_dict = {
        "history": history,
        "main_model": main_model,
        "model_improving_points": model_improving_points,
        "model_indices": model_indices,
        "x_accepted": x_accepted,
        "n_modelpoints": n_modelpoints,
        "n": n,
        "delta": delta,
        "criterion": criterion,
        "lower_bounds": None,
        "upper_bounds": None,
    }

    expected_dict = {
        "model_indices_expected": dict_["model_indices_expected"],
        "history_x_expected": dict_["history_x_expected"],
    }

    return inputs_dict, expected_dict


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
def dicts_find_affine_points(request):
    history = LeastSquaresHistory()
    dict_ = pd.read_pickle(
        TEST_FIXTURES_DIR / f"find_affine_points_{request.param}.pkl"
    )

    history.add_entries(
        dict_["history_x"],
        np.zeros(dict_["history_x"].shape),
    )
    x_accepted = dict_["min_x"]
    model_improving_points = dict_["model_improving_points"]
    project_x_onto_null = dict_["project_x_onto_null"]
    delta = dict_["delta"]
    c = dict_["c"]
    model_indices = dict_["model_indices"]
    n_modelpoints = dict_["n_modelpoints"]
    theta1 = 1e-5
    n = 3

    inputs_dict = {
        "history": history,
        "x_accepted": x_accepted,
        "model_improving_points": model_improving_points,
        "project_x_onto_null": project_x_onto_null,
        "delta": delta,
        "theta1": theta1,
        "c": c,
        "model_indices": model_indices,
        "n": n,
        "n_modelpoints": n_modelpoints,
    }

    expected_dict = {
        "model_improving_points_expected": dict_["model_improving_points_expected"],
        "model_indices_expected": dict_["model_indices_expected"],
        "n_modelpoints_expected": dict_["n_modelpoints_expected"],
    }

    return inputs_dict, expected_dict


@pytest.fixture
def dicts_update_initial_residual_model():
    dict_ = pd.read_pickle(TEST_FIXTURES_DIR / "update_initial_residual_model.pkl")

    inputs_dict = {}

    inputs_dict["initial_residual_model"] = dict_["initial_residual_model"]
    inputs_dict["x_candidate"] = dict_["x_candidate"]
    inputs_dict["residuals_candidate"] = dict_["residuals_candidate"]

    expected_dict = dict_["residual_model_expected"]

    return inputs_dict, expected_dict


@pytest.fixture
def dicts_update_main_model_with_new_accepted_x():
    dict_ = pd.read_pickle(TEST_FIXTURES_DIR / "update_center.pkl")
    inputs_dict = {}
    expected_dict = {}

    main_model = {
        "linear_terms": dict_["first_derivative"],
        "square_terms": dict_["second_derivative"],
    }

    inputs_dict["main_model"] = main_model
    inputs_dict["x_candidate"] = (dict_["xplus"] - dict_["min_x"]) / dict_["delta"]

    expected_dict["linear_terms"] = dict_["first_derivative_expected"]  # fdiff

    return inputs_dict, expected_dict


@pytest.fixture
def dicts_update_residual_model_with_new_accepted_x():
    dict_ = pd.read_pickle(TEST_FIXTURES_DIR / "update_center.pkl")
    inputs_dict = {}
    expected_dict = {}

    residual_model = {
        "intercepts": dict_["min_criterion"],
        "linear_terms": dict_["gradient"],
        "square_terms": dict_["hessian"],
    }

    inputs_dict["residual_model"] = residual_model
    inputs_dict["x_candidate"] = (dict_["xplus"] - dict_["min_x"]) / dict_["delta"]

    expected_dict["intercepts"] = dict_["min_criterion_expected"]
    expected_dict["linear_terms"] = dict_["gradient_expected"]

    return inputs_dict, expected_dict


def test_update_initial_residual_model(dicts_update_initial_residual_model):
    inputs_dict, residual_model_expected = dicts_update_initial_residual_model

    residual_model_out = update_initial_residual_model(**inputs_dict)

    aaae(residual_model_out["intercepts"], residual_model_expected["intercepts"])
    aaae(residual_model_out["linear_terms"], residual_model_expected["linear_terms"])


def test_update_residual_model_with_new_accepted_x(
    dicts_update_residual_model_with_new_accepted_x,
):
    (
        inputs_dict,
        residual_model_expected,
    ) = dicts_update_residual_model_with_new_accepted_x

    residual_model_out = update_residual_model_with_new_accepted_x(**inputs_dict)

    aaae(residual_model_out["intercepts"], residual_model_expected["intercepts"])
    aaae(residual_model_out["linear_terms"], residual_model_expected["linear_terms"])
    aaae(
        residual_model_out["square_terms"],
        inputs_dict["residual_model"]["square_terms"],
    )


@pytest.mark.xfail(
    reason="Known differences in rounding of numbers that are virtually zero."
)
def test_update_main_model_with_new_accepted_x(
    dicts_update_main_model_with_new_accepted_x,
):
    (
        inputs_dict,
        main_model_expected,
    ) = dicts_update_main_model_with_new_accepted_x

    main_model_out = update_main_model_with_new_accepted_x(**inputs_dict)

    aaae(main_model_out["linear_terms"], main_model_expected["linear_terms"])


def test_find_affine_points(dicts_find_affine_points):
    inputs_dict, expected_dict = dicts_find_affine_points

    (
        model_improving_points_out,
        model_indices_out,
        n_modelpoints_out,
        project_x_onto_null_out,
    ) = find_affine_points(**inputs_dict)

    aaae(
        model_improving_points_out,
        expected_dict["model_improving_points_expected"],
    )
    aaae(model_indices_out, expected_dict["model_indices_expected"])
    assert np.allclose(n_modelpoints_out, expected_dict["n_modelpoints_expected"])
    assert np.allclose(project_x_onto_null_out, True)


def test_improve_model(dicts_improve_model):
    inputs_dict, expected_dict = dicts_improve_model

    history_out, model_indices_out = improve_main_model(
        **inputs_dict,
    )

    aaae(model_indices_out, expected_dict["model_indices_expected"])
    for index_added in range(inputs_dict["n"] - inputs_dict["n_modelpoints"], 0, -1):
        aaae(
            history_out.get_xs(index=-index_added),
            expected_dict["history_x_expected"][-index_added],
        )


def test_add_more_points(dicts_add_more_points):
    inputs_dict, expected_dict = dicts_add_more_points
    (
        lower_triangular,
        basis_null_space,
        monomial_basis,
        x_sample_monomial_basis,
        n_modelpoints,
    ) = add_more_points(**inputs_dict)

    aaae(lower_triangular, expected_dict["lower_triangular_expected"])
    aaae(basis_null_space, expected_dict["basis_null_space_expected"])
    aaae(monomial_basis, expected_dict["monomial_basis_expected"])
    aaae(x_sample_monomial_basis, expected_dict["x_sample_monomial_basis_expected"])
    assert np.allclose(n_modelpoints, expected_dict["n_modelpoints_expected"])


def test_interpolate_f(dicts_interpolate_f):
    inputs_dict, expected_dict = dicts_interpolate_f
    f_interpolated = interpolate_f(**inputs_dict)

    aaae(f_interpolated, expected_dict["f_interpolated_expected"])


def test_get_coefficients_residual_model(dict_get_coefficients_residual_model):
    coefficients_to_add = get_coefficients_residual_model(
        lower_triangular=dict_get_coefficients_residual_model["lower_triangular"],
        basis_null_space=dict_get_coefficients_residual_model["basis_null_space"],
        monomial_basis=dict_get_coefficients_residual_model["monomial_basis"],
        x_sample_monomial_basis=dict_get_coefficients_residual_model[
            "interpolation_set"
        ],
        f_interpolated=dict_get_coefficients_residual_model["approximation_error"],
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

    residual_model_out = update_residual_model(
        residual_model=residual_model,
        coefficients_to_add=coefficients_to_add,
        delta=dict_update_residual_model["delta"],
        delta_old=dict_update_residual_model["delta_old"],
    )

    aaae(
        residual_model_out["linear_terms"],
        dict_update_residual_model["gradient_expected"],
    )
    aaae(
        residual_model_out["square_terms"],
        dict_update_residual_model["hessian_expected"],
    )


def test_update_main_from_residual_model(dict_update_main_from_residual_model):
    residual_model = {
        "intercepts": dict_update_main_from_residual_model["min_criterion"],
        "linear_terms": dict_update_main_from_residual_model["gradient"],
        "square_terms": dict_update_main_from_residual_model["hessian"],
    }
    main_model = update_main_from_residual_model(
        residual_model, multiply_square_terms_with_residuals=True
    )

    aaae(
        main_model["linear_terms"],
        dict_update_main_from_residual_model["first_derivative_expected"],
    )
    aaae(
        main_model["square_terms"],
        dict_update_main_from_residual_model["second_derivative_expected"],
        decimal=3,
    )
