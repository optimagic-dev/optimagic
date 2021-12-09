from functools import partial

import numpy as np
import pandas as pd
import pytest
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


@pytest.mark.parametrize(
    "start_vec, gtol, solver_sub, trustregion_subproblem_options",
    [
        (
            np.array([0.15, 0.008, 0.01]),
            1e-7,
            "trust-constr",
            {"ftol": 1e-7, "xtol": 1e-7, "gtol": 1e-7},
        ),
        (
            np.array([0.1, 0.1, 0.1]),
            1e-9,
            "trust-constr",
            {"ftol": 1e-7, "xtol": 1e-7, "gtol": 1e-7},
        ),
        (
            np.array([1e-6, 1e-6, 1e-6]),
            1e-12,
            "trust-constr",
            {"ftol": 1e-8, "xtol": 1e-8, "gtol": 1e-8},
        ),
        (
            np.array([-1e-6, -1e-6, -1e-6]),
            1e-12,
            "trust-constr",
            {"ftol": 1e-12, "xtol": 1e-12, "gtol": 1e-12},
        ),
        (
            np.array([0.4, 0.4, 0.4]),
            1e-12,
            "trust-constr",
            {"ftol": 1e-12, "xtol": 1e-12, "gtol": 1e-12},
        ),
        (
            np.array([0.15, 0.008, 0.01]),
            1e-6,
            "L-BFGS-B",
            {"ftol": 1e-10, "xtol": None, "gtol": 1e-6},
        ),
        (
            np.array([1e-6, 1e-6, 1e-6]),
            1e-12,
            "L-BFGS-B",
            {"ftol": 1e-12, "xtol": None, "gtol": 1e-12},
        ),
        (
            np.array([0.5, 0.5, 0.5]),
            1e-12,
            "L-BFGS-B",
            {"ftol": 1e-12, "xtol": None, "gtol": 1e-12},
        ),
        (
            np.array([0.15, 0.008, 0.01]),
            1e-6,
            "SLSQP",
            {"ftol": 1e-10, "xtol": None, "gtol": None},
        ),
    ],
)
def test_solution_and_histories(
    start_vec, gtol, solver_sub, trustregion_subproblem_options, criterion
):
    n_obs = 214
    delta = 0.1
    delta_min = 1e-6
    delta_max = 1e6
    gamma0 = 0.5
    gamma1 = 2.0
    theta1 = 1e-5
    theta2 = 1e-4
    eta0 = 0.0
    eta1 = 0.1
    c1 = np.sqrt(3)
    c2 = 10
    maxiter = 200

    rslt = internal_solve_pounders(
        x0=start_vec,
        n_obs=n_obs,
        criterion=criterion,
        delta=delta,
        delta_min=delta_min,
        delta_max=delta_max,
        gamma0=gamma0,
        gamma1=gamma1,
        theta1=theta1,
        theta2=theta2,
        eta0=eta0,
        eta1=eta1,
        c1=c1,
        c2=c2,
        maxiter=maxiter,
        gtol=gtol,
        ftol_sub=trustregion_subproblem_options["ftol"],
        xtol_sub=trustregion_subproblem_options["xtol"],
        gtol_sub=trustregion_subproblem_options["gtol"],
        solver_sub=solver_sub,
        lower_bounds=None,
        upper_bounds=None,
    )

    aaae(rslt["solution_x"], np.array([0.190279, 0.00613141, 0.0105309]))
