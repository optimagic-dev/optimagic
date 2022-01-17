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


start_params = [
    np.array([0.15, 0.008, 0.01]),
    # np.ones(3) * 0.25, # noqa: E800
    np.array([1e-6, 1e-2, 1e-6]),
]

TEST_CASES = []
for subsolver in ["L-BFGS-B", "trust-constr"]:
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
    }
    return out


@pytest.mark.parametrize(
    "start_vec, gtol, solver_sub, trustregion_subproblem_options", TEST_CASES
)
def test_solution(
    start_vec, gtol, solver_sub, trustregion_subproblem_options, criterion, options
):

    maxiter = 200

    rslt = internal_solve_pounders(
        x0=start_vec,
        criterion=criterion,
        maxiter=maxiter,
        gtol=gtol,
        ftol_sub=trustregion_subproblem_options["ftol"],
        xtol_sub=trustregion_subproblem_options["xtol"],
        gtol_sub=trustregion_subproblem_options["gtol"],
        solver_sub=solver_sub,
        n_cores=1,
        batch_evaluator=joblib_batch_evaluator,
        **options,
    )

    aaae(rslt["solution_x"], np.array([0.190279, 0.00613141, 0.0105309]), decimal=5)
