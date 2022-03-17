"""Test the internal pounders interface."""
from functools import partial

import numpy as np
import pandas as pd
import pytest
from estimagic.batch_evaluators import joblib_batch_evaluator
from estimagic.config import TEST_FIXTURES_DIR
from estimagic.optimization.pounders import internal_solve_pounders
from numpy.testing import assert_array_almost_equal as aaae


def load_history(start_vec, solver_sub):
    start_vec_str = np.array2string(
        start_vec, precision=3, separator=",", suppress_small=False
    )

    history_x = np.genfromtxt(
        TEST_FIXTURES_DIR / f"history_x_{start_vec_str}_{solver_sub}_3_8.csv",
        delimiter=",",
    )
    history_criterion = np.genfromtxt(
        TEST_FIXTURES_DIR / f"history_criterion_{start_vec_str}_{solver_sub}_3_8.csv",
        delimiter=",",
    )

    return history_x, history_criterion


@pytest.fixture
def criterion():
    data = pd.read_csv(TEST_FIXTURES_DIR / "pounders_example_data.csv")
    endog = np.asarray(data["y"])
    exog = np.asarray(data["t"])

    def func(x: np.ndarray, exog: np.ndarray, endog: np.ndarray) -> np.ndarray:
        """User provided residual function."""
        return endog - np.exp(-x[0] * exog) / (x[1] + x[2] * exog)

    return partial(func, exog=exog, endog=endog)


@pytest.fixture()
def options():
    out = {
        "delta": 0.1,
        "delta_min": 1e-6,
        "delta_max": 1e6,
        "gamma0": 0.5,
        "gamma1": 2.0,
        "theta1": 1e-5,
        "theta2": 1e-4,
        "eta0": 0.0,
        "eta1": 0.1,
        "c1": np.sqrt(3),
        "c2": 10,
        "lower_bounds": None,
        "upper_bounds": None,
        "maxiter": 200,
    }
    return out


@pytest.fixture()
def trustregion_subproblem_options():
    out = {
        "maxiter": 20,
        "maxiter_steepest_descent": 5,
        "step_size_newton": 1e-3,
        "ftol_abs": 1e-8,
        "ftol_scaled": 1e-8,
        "xtol": 1e-8,
        "gtol_abs": 1e-8,
        "gtol_rel": 1e-8,
        "gtol_scaled": 1e-8,
        "steptol": 1e-8,
    }
    return out


start_params = [
    np.array([0.15, 0.008, 0.01]),
    np.array([1e-6, 1e-2, 1e-6]),
]

TEST_CASES = []
for subsolver in ["bntr", "gqtpar"]:
    for x0 in start_params:
        for gtol in [1e-8]:
            for subtol in [1e-8, 1e-9]:
                TEST_CASES.append(
                    (
                        x0,
                        gtol,
                        subsolver,
                        {"ftol": subtol, "xtol": subtol, "gtol": subtol},
                    )
                )


@pytest.mark.parametrize(
    "start_vec",
    [
        (np.array([0.15, 0.008, 0.01])),
        (np.array([1e-3, 1e-3, 1e-3])),
    ],
)
def test_bntr(start_vec, criterion, options, trustregion_subproblem_options):
    solver_sub = "bntr"

    result = internal_solve_pounders(
        x0=start_vec,
        criterion=criterion,
        gtol=gtol,
        solver_sub=solver_sub,
        maxiter_sub=trustregion_subproblem_options["maxiter"],
        maxiter_steepest_descent_sub=trustregion_subproblem_options[
            "maxiter_steepest_descent"
        ],
        step_size_newton_sub=trustregion_subproblem_options["step_size_newton"],
        ftol_abs_sub=trustregion_subproblem_options["ftol_abs"],
        ftol_scaled_sub=trustregion_subproblem_options["ftol_scaled"],
        xtol_sub=trustregion_subproblem_options["xtol"],
        gtol_abs_sub=trustregion_subproblem_options["gtol_abs"],
        gtol_rel_sub=trustregion_subproblem_options["gtol_rel"],
        gtol_scaled_sub=trustregion_subproblem_options["gtol_scaled"],
        steptol_sub=trustregion_subproblem_options["steptol"],
        n_cores=1,
        batch_evaluator=joblib_batch_evaluator,
        **options,
    )

    x_expected = np.array([0.1902789114691, 0.006131410288292, 0.01053088353832])
    aaae(result["solution_x"], x_expected, decimal=5)


@pytest.mark.parametrize("start_vec", [(np.array([0.15, 0.008, 0.01]))])
def test_gqtpar(start_vec, criterion, options, trustregion_subproblem_options):
    solver_sub = "gqtpar"

    result = internal_solve_pounders(
        x0=start_vec,
        criterion=criterion,
        gtol=gtol,
        solver_sub=solver_sub,
        maxiter_sub=trustregion_subproblem_options["maxiter"],
        maxiter_steepest_descent_sub=trustregion_subproblem_options[
            "maxiter_steepest_descent"
        ],
        step_size_newton_sub=trustregion_subproblem_options["step_size_newton"],
        ftol_abs_sub=trustregion_subproblem_options["ftol_abs"],
        ftol_scaled_sub=trustregion_subproblem_options["ftol_scaled"],
        xtol_sub=trustregion_subproblem_options["xtol"],
        gtol_abs_sub=trustregion_subproblem_options["gtol_abs"],
        gtol_rel_sub=trustregion_subproblem_options["gtol_rel"],
        gtol_scaled_sub=trustregion_subproblem_options["gtol_scaled"],
        steptol_sub=trustregion_subproblem_options["steptol"],
        n_cores=1,
        batch_evaluator=joblib_batch_evaluator,
        **options,
    )

    x_expected = np.array([0.1902789114691, 0.006131410288292, 0.01053088353832])
    aaae(result["solution_x"], x_expected, decimal=5)
