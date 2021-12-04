from functools import partial

import numpy as np
import pandas as pd
import pytest
from estimagic.config import TEST_FIXTURES_DIR
from estimagic.optimization.pounders import internal_solve_pounders
from numpy.testing import assert_array_almost_equal as aaae


@pytest.fixture
def criterion():
    data = pd.read_csv(TEST_FIXTURES_DIR / "example_data.csv")
    endog = np.asarray(data["y"])
    exog = np.asarray(data["t"])

    def func(x: np.ndarray, exog: np.ndarray, endog: np.ndarray) -> np.ndarray:
        """User provided residual function."""
        return endog - np.exp(-x[0] * exog) / (x[1] + x[2] * exog)

    return partial(func, exog=exog, endog=endog)


@pytest.mark.parametrize(
    "solver_sub, trustregion_subproblem_options",
    [
        ("trust-constr", {"ftol": 1e-6, "xtol": 1e-6, "gtol": 1e-6}),
        ("L-BFGS-B", {"ftol": 1e-8, "xtol": None, "gtol": 1e-6}),
        ("SLSQP", {"ftol": 1e-8, "xtol": None, "gtol": None}),
    ],
)
def test_integration(solver_sub, trustregion_subproblem_options, criterion):
    x0 = np.array([0.15, 0.008, 0.01])
    nobs = 214
    delta = 0.1
    delta_min = 1e-6
    delta_max = 1e6
    gamma0 = 0.5
    gamma1 = 2.0
    theta1 = 1e-5
    theta2 = 1e-4
    eta0 = 0.0
    eta1 = 0.1
    c1 = np.sqrt(x0.shape[0])
    c2 = 10
    gtol = 1e-6
    maxiter = 200

    rslt = internal_solve_pounders(
        x0=x0,
        nobs=nobs,
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
