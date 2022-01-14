from functools import partial

import numpy as np
import pandas as pd
import pytest
from estimagic.batch_evaluators import joblib_batch_evaluator
from estimagic.config import TEST_FIXTURES_DIR
from estimagic.optimization.history import LeastSquaresHistory
from estimagic.optimization.pounders_auxiliary import (
    add_points_to_make_main_model_fully_linear,
)
from estimagic.optimization.pounders_auxiliary import find_affine_points
from estimagic.optimization.pounders_auxiliary import get_coefficients_residual_model
from estimagic.optimization.pounders_auxiliary import (
    get_interpolation_matrices_residual_model,
)
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

# ======================================================================================
# Fixtures
# ======================================================================================


@pytest.fixture
def criterion():
    data = pd.read_csv(TEST_FIXTURES_DIR / "pounders_example_data.csv")
    endog = np.asarray(data["y"])
    exog = np.asarray(data["t"])

    def func(x: np.ndarray, exog: np.ndarray, endog: np.ndarray) -> np.ndarray:
        """User provided residual function."""
        return endog - np.exp(-x[0] * exog) / (x[1] + x[2] * exog)

    return partial(func, exog=exog, endog=endog)


@pytest.fixture
def data_update_initial_residual_model():
    test_data = pd.read_pickle(TEST_FIXTURES_DIR / "update_initial_residual_model.pkl")
    inputs_dict = {}

    inputs_dict["initial_residual_model"] = test_data["initial_residual_model"]
    inputs_dict["x_candidate"] = test_data["x_candidate"]
    inputs_dict["residuals_candidate"] = test_data["residuals_candidate"]

    expected_dict = test_data["residual_model_expected"]

    return inputs_dict, expected_dict


@pytest.fixture
def data_update_residual_model():
    test_data = pd.read_pickle(TEST_FIXTURES_DIR / "update_residual_model.pkl")

    residual_model = {
        "linear_terms": test_data["linear_terms"],
        "square_terms": test_data["square_terms"],
    }
    coefficients_to_add = {
        "linear_terms": test_data["coefficients_linear_terms"].T,
        "square_terms": test_data["coefficients_square_terms"],
    }

    inputs_dict = {
        "residual_model": residual_model,
        "coefficients_to_add": coefficients_to_add,
        "delta": test_data["delta"],
        "delta_old": test_data["delta_old"],
    }

    expected_dict = {
        "linear_terms": test_data["linear_terms_expected"],
        "square_terms": test_data["square_terms_expected"],
    }

    return inputs_dict, expected_dict


@pytest.fixture
def data_update_main_from_residual_model():
    test_data = pd.read_pickle(
        TEST_FIXTURES_DIR / "update_main_from_residual_model.pkl"
    )

    residual_model = {
        "intercepts": test_data["residuals"],
        "linear_terms": test_data["linear_terms_residual_model"],
        "square_terms": test_data["square_terms_residual_model"],
    }

    main_model_expected = {
        "linear_terms": test_data["linear_terms_main_model_expected"],
        "square_terms": test_data["square_terms_main_model_expected"],
    }

    return residual_model, main_model_expected


@pytest.fixture
def data_update_residual_model_with_new_accepted_x():
    test_data = pd.read_pickle(
        TEST_FIXTURES_DIR / "update_residual_model_with_new_accepted_x.pkl"
    )
    inputs_dict = {}
    expected_dict = {}

    residual_model = {
        "intercepts": test_data["residuals"],
        "linear_terms": test_data["linear_terms"],
        "square_terms": test_data["square_terms"],
    }

    inputs_dict["residual_model"] = residual_model
    inputs_dict["x_candidate"] = (
        test_data["x_candidate_uncentered"] - test_data["best_xs"]
    ) / test_data["delta"]

    expected_dict["intercepts"] = test_data["residuals_expected"]
    expected_dict["linear_terms"] = test_data["linear_terms_expected"]

    return inputs_dict, expected_dict


@pytest.fixture
def data_update_main_model_with_new_accepted_x():
    test_data = pd.read_pickle(
        TEST_FIXTURES_DIR / "update_main_model_with_new_accepted_x.pkl"
    )
    inputs_dict = {}
    expected_dict = {}

    main_model = {
        "linear_terms": test_data["linear_terms"],
        "square_terms": test_data["square_terms"],
    }

    inputs_dict["main_model"] = main_model
    inputs_dict["x_candidate"] = (
        test_data["x_candidate_uncentered"] - test_data["best_xs"]
    ) / test_data["delta"]

    expected_dict["linear_terms"] = test_data["linear_terms_expected"]

    return inputs_dict, expected_dict


@pytest.fixture(
    params=[
        "zero_i",
        "zero_ii",
        "zero_iii",
        "zero_iv",
        "nonzero_i",
        "nonzero_ii",
        "nonzero_iii",
    ]
)
def data_find_affine_points(request):
    test_data = pd.read_pickle(
        TEST_FIXTURES_DIR / f"find_affine_points_{request.param}.pkl"
    )

    history = LeastSquaresHistory()
    history.add_entries(
        test_data["history_x"],
        np.zeros(test_data["history_x"].shape),
    )

    inputs_dict = {
        "history": history,
        "x_accepted": test_data["x_accepted"],
        "model_improving_points": test_data["model_improving_points"],
        "project_x_onto_null": test_data["project_x_onto_null"],
        "delta": test_data["delta"],
        "theta1": test_data["theta1"],
        "c": test_data["c"],
        "model_indices": test_data["model_indices"],
        "n_modelpoints": test_data["n_modelpoints"],
    }

    expected_dict = {
        "model_improving_points_expected": test_data["model_improving_points_expected"],
        "model_indices_expected": test_data["model_indices_expected"],
        "n_modelpoints_expected": test_data["n_modelpoints_expected"],
    }

    return inputs_dict, expected_dict


@pytest.fixture(params=["i", "ii"])
def data_add_points_until_main_model_fully_linear(request, criterion):
    test_data = pd.read_pickle(
        TEST_FIXTURES_DIR
        / f"add_points_until_main_model_fully_linear_{request.param}.pkl"
    )

    history = LeastSquaresHistory()
    n = 3
    n_modelpoints = test_data["n_modelpoints"]
    history.add_entries(
        test_data["history_x"][: -(n - n_modelpoints)],
        test_data["history_criterion"][: -(n - n_modelpoints)],
    )

    main_model = {
        "linear_terms": test_data["linear_terms"],
        "square_terms": test_data["square_terms"],
    }

    index_best_xs = test_data["index_best_xs"]
    x_accepted = test_data["history_x"][index_best_xs]

    inputs_dict = {
        "history": history,
        "main_model": main_model,
        "model_improving_points": test_data["model_improving_points"],
        "model_indices": test_data["model_indices"],
        "x_accepted": x_accepted,
        "n_modelpoints": n_modelpoints,
        "delta": test_data["delta"],
        "criterion": criterion,
        "lower_bounds": None,
        "upper_bounds": None,
    }

    expected_dict = {
        "model_indices_expected": test_data["model_indices_expected"],
        "history_x_expected": test_data["history_x_expected"],
    }

    return inputs_dict, expected_dict


@pytest.fixture
def data_get_interpolation_matrices_residual_model():
    test_data = pd.read_pickle(
        TEST_FIXTURES_DIR / "get_interpolation_matrices_residual_model.pkl"
    )

    history = LeastSquaresHistory()
    history.add_entries(
        test_data["history_x"],
        np.zeros(test_data["history_x"].shape),
    )

    n = 3
    inputs_dict = {
        "history": history,
        "x_accepted": test_data["x_accepted"],
        "model_indices": test_data["model_indices"],
        "delta": test_data["delta"],
        "c2": 10,
        "theta2": 1e-4,
        "n_maxinterp": 2 * n + 1,
        "n_modelpoints": test_data["n_modelpoints"],
    }

    expected_dict = {
        "x_sample_monomial_basis_expected": test_data[
            "x_sample_monomial_basis_expected"
        ],
        "monomial_basis_expected": test_data["monomial_basis_expected"],
        "basis_null_space_expected": test_data["basis_null_space_expected"],
        "lower_triangular_expected": test_data["lower_triangular_expected"],
        "n_modelpoints_expected": test_data["n_modelpoints_expected"],
    }

    return inputs_dict, expected_dict


@pytest.fixture(params=["4", "7"])
def data_interpolate_f(request):
    test_data = pd.read_pickle(
        TEST_FIXTURES_DIR / f"interpolate_f_iter_{request.param}.pkl"
    )

    history = LeastSquaresHistory()
    history.add_entries(
        test_data["history_x"],
        test_data["history_criterion"],
    )

    residual_model = {
        "intercepts": test_data["residuals"],
        "linear_terms": test_data["linear_terms_residual_model"],
        "square_terms": test_data["square_terms_residual_model"],
    }

    x_accepted = test_data["x_accepted"]
    model_indices = test_data["model_indices"]
    n_modelpoints = test_data["n_modelpoints"]
    delta_old = test_data["delta_old"]

    center_info = {"x": x_accepted, "radius": delta_old}
    interpolation_set = history.get_centered_xs(
        center_info, index=model_indices[:n_modelpoints]
    )

    n = 3
    inputs_dict = {
        "history": history,
        "residual_model": residual_model,
        "interpolation_set": interpolation_set,
        "model_indices": model_indices,
        "n_modelpoints": n_modelpoints,
        "n_maxinterp": 2 * n + 1,
    }

    expected_dict = {
        "interpolation_set_expected": test_data["interpolation_set_expected"],
        "f_interpolated_expected": test_data["f_interpolated_expected"],
    }

    return inputs_dict, expected_dict


@pytest.fixture
def data_get_coefficients_residual_model():
    test_data = pd.read_pickle(
        TEST_FIXTURES_DIR / "get_coefficients_residual_model.pkl"
    )

    inputs_dict = {
        "x_sample_monomial_basis": test_data["x_sample_monomial_basis"],
        "monomial_basis": test_data["monomial_basis"],
        "basis_null_space": test_data["basis_null_space"],
        "lower_triangular": test_data["lower_triangular"],
        "f_interpolated": test_data["f_interpolated"],
        "n_modelpoints": test_data["n_modelpoints"],
    }

    expected_dict = {
        "linear_terms": test_data["linear_terms_expected"],
        "square_terms": test_data["square_terms_expected"],
    }

    return inputs_dict, expected_dict


# ======================================================================================
# Test cases
# ======================================================================================


def test_update_initial_residual_model(data_update_initial_residual_model):
    inputs, residual_model_expected = data_update_initial_residual_model

    residual_model_out = update_initial_residual_model(**inputs)

    aaae(residual_model_out["intercepts"], residual_model_expected["intercepts"])
    aaae(residual_model_out["linear_terms"], residual_model_expected["linear_terms"])


def test_update_residual_model(data_update_residual_model):
    inputs, expected = data_update_residual_model

    residual_model_out = update_residual_model(**inputs)

    aaae(
        residual_model_out["linear_terms"],
        expected["linear_terms"],
    )
    aaae(
        residual_model_out["square_terms"],
        expected["square_terms"],
    )


def test_update_main_from_residual_model(data_update_main_from_residual_model):
    residual_model, main_model_expected = data_update_main_from_residual_model

    main_model_out = update_main_from_residual_model(
        residual_model, multiply_square_terms_with_residuals=True
    )

    aaae(
        main_model_out["linear_terms"],
        main_model_expected["linear_terms"],
    )
    aaae(
        main_model_out["square_terms"],
        main_model_expected["square_terms"],
        decimal=3,
    )


def test_update_residual_model_with_new_accepted_x(
    data_update_residual_model_with_new_accepted_x,
):
    (
        inputs,
        residual_model_expected,
    ) = data_update_residual_model_with_new_accepted_x

    residual_model_out = update_residual_model_with_new_accepted_x(**inputs)

    aaae(residual_model_out["intercepts"], residual_model_expected["intercepts"])
    aaae(residual_model_out["linear_terms"], residual_model_expected["linear_terms"])
    aaae(
        residual_model_out["square_terms"],
        inputs["residual_model"]["square_terms"],
    )


@pytest.mark.xfail(reason="Known rounding differences between C and Python.")
def test_update_main_model_with_new_accepted_x(
    data_update_main_model_with_new_accepted_x,
):
    (
        inputs,
        main_model_expected,
    ) = data_update_main_model_with_new_accepted_x

    main_model_out = update_main_model_with_new_accepted_x(**inputs)

    aaae(main_model_out["linear_terms"], main_model_expected["linear_terms"])


def test_find_affine_points(data_find_affine_points):
    inputs, expected = data_find_affine_points

    (
        model_improving_points_out,
        model_indices_out,
        n_modelpoints_out,
        project_x_onto_null_out,
    ) = find_affine_points(**inputs)

    aaae(
        model_improving_points_out,
        expected["model_improving_points_expected"],
    )
    aaae(model_indices_out, expected["model_indices_expected"])
    assert np.allclose(n_modelpoints_out, expected["n_modelpoints_expected"])
    assert np.allclose(project_x_onto_null_out, True)


def test_add_points_until_main_model_fully_linear(
    data_add_points_until_main_model_fully_linear,
):
    inputs, expected = data_add_points_until_main_model_fully_linear
    n = 3

    history_out, model_indices_out = add_points_to_make_main_model_fully_linear(
        **inputs, n_cores=1, batch_evaluator=joblib_batch_evaluator
    )

    aaae(model_indices_out, expected["model_indices_expected"])
    for index_added in range(n - inputs["n_modelpoints"], 0, -1):
        aaae(
            history_out.get_xs(index=-index_added),
            expected["history_x_expected"][-index_added],
        )


def test_get_interpolation_matrices_residual_model(
    data_get_interpolation_matrices_residual_model,
):
    inputs, expected = data_get_interpolation_matrices_residual_model
    (
        x_sample_monomial_basis,
        monomial_basis,
        basis_null_space,
        lower_triangular,
        n_modelpoints,
    ) = get_interpolation_matrices_residual_model(**inputs)

    aaae(x_sample_monomial_basis, expected["x_sample_monomial_basis_expected"])
    aaae(monomial_basis, expected["monomial_basis_expected"])
    aaae(basis_null_space, expected["basis_null_space_expected"])
    aaae(lower_triangular, expected["lower_triangular_expected"])
    assert np.allclose(n_modelpoints, expected["n_modelpoints_expected"])


def test_interpolate_f(data_interpolate_f):
    inputs, expected = data_interpolate_f
    f_interpolated = interpolate_f(**inputs)

    aaae(f_interpolated, expected["f_interpolated_expected"])


def test_get_coefficients_residual_model(data_get_coefficients_residual_model):
    inputs, expected = data_get_coefficients_residual_model

    coefficients_to_add = get_coefficients_residual_model(**inputs)

    aaae(
        coefficients_to_add["linear_terms"],
        expected["linear_terms"].T,
    )
    aaae(
        coefficients_to_add["square_terms"],
        expected["square_terms"],
    )
