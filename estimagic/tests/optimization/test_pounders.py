"""Test the wrapper around pounders."""
import functools
import sys

import numpy as np
import pandas as pd
import pytest

from estimagic.optimization.optimize import minimize
from estimagic.optimization.pounders import minimize_pounders_np

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="Pounders is not supported on Windows."
)

NUM_AGENTS = 2_000
algorithm = "tao_pounders"


def get_random_params(length, low=0, high=1, lower_bound=-np.inf, upper_bound=np.inf):
    params = pd.DataFrame(
        {
            "value": np.random.uniform(low, high, size=length),
            "lower": lower_bound,
            "upper": upper_bound,
        }
    )

    return params


def test_robustness_1():
    np.random.seed(5470)
    true_params = get_random_params(3)
    start_params = get_random_params(3)

    def _criterion_pandas(endog, exog, crit, params):
        x = params["value"].to_numpy()
        out = _criterion(endog, exog, crit, x)
        return out

    exog, endog = _simulate_sample(NUM_AGENTS, true_params, 0.5)
    crit = "nonlinear"
    objective = functools.partial(_criterion_pandas, endog, exog, crit)
    results = minimize(objective, start_params, algorithm, logging=None)

    np.testing.assert_array_almost_equal(
        true_params["value"].values, results[0]["x"], decimal=0.1
    )


def test_robustness_2():
    np.random.seed(5471)
    true_params = get_random_params(2)
    start_params = get_random_params(2)
    bounds = tuple(true_params[["lower", "upper"]].to_numpy().T)

    exog, endog = _simulate_ols_sample(NUM_AGENTS, true_params)
    crit = "ols"
    objective = functools.partial(_criterion, endog, exog, crit)
    results = minimize_pounders_np(objective, start_params["value"].to_numpy(), bounds)
    calculated = results["x"]

    x = np.column_stack([np.ones_like(exog), exog])
    y = endog.reshape(len(endog), 1)
    expected = np.linalg.lstsq(x, y, rcond=None)[0].flatten()

    np.testing.assert_almost_equal(calculated, expected, decimal=6)


def test_box_constr():
    np.random.seed(5472)
    true_params = get_random_params(2, 0.3, 0.4, 0, 0.3)
    bounds = tuple(true_params[["lower", "upper"]].to_numpy().T)

    start_params = true_params.copy()
    start_params["value"] = get_random_params(2, 0.1, 0.2)["value"]

    exog, endog = _simulate_ols_sample(NUM_AGENTS, true_params)
    crit = "ols"
    objective = functools.partial(_criterion, endog, exog, crit)
    calculated = minimize_pounders_np(
        objective, start_params["value"].to_numpy(), bounds
    )
    assert 0 <= calculated["x"][0] <= 0.3
    assert 0 <= calculated["x"][1] <= 0.3


def test_max_iters():
    np.random.seed(5473)
    true_params = get_random_params(2, 0.3, 0.4, 0, 0.3)
    start_params = get_random_params(2, 0.1, 0.2)
    bounds = tuple(true_params[["lower", "upper"]].to_numpy().T)

    exog, endog = _simulate_ols_sample(NUM_AGENTS, true_params)
    crit = "ols"
    objective = functools.partial(_criterion, endog, exog, crit)
    calculated = minimize_pounders_np(
        objective, start_params["value"].to_numpy(), bounds, max_iterations=25
    )

    assert (
        calculated["conv"] == "user defined" or calculated["conv"] == "step size small"
    )
    if calculated["conv"] == 8:
        assert calculated["sol"][0] == 25


def test_grtol():
    np.random.seed(5474)
    true_params = get_random_params(2, 0.3, 0.4, 0, 0.3)
    start_params = get_random_params(2, 0.1, 0.2)
    bounds = tuple(true_params[["lower", "upper"]].to_numpy().T)

    exog, endog = _simulate_ols_sample(NUM_AGENTS, true_params)
    crit = "ols"
    objective = functools.partial(_criterion, endog, exog, crit)
    calculated = minimize_pounders_np(
        objective,
        start_params["value"].to_numpy(),
        bounds=bounds,
        gatol=False,
        gttol=False,
    )

    assert (
        calculated["conv"] == "grtol below critical value"
        or calculated["conv"] == "step size small"
    )

    if calculated["conv"] == 4:
        assert calculated["sol"][2] / calculated["sol"][1] < 10


def test_gatol():
    np.random.seed(5475)
    true_params = get_random_params(2, 0.3, 0.4, 0, 0.3)
    start_params = get_random_params(2, 0.1, 0.2)
    bounds = tuple(true_params[["lower", "upper"]].to_numpy().T)

    exog, endog = _simulate_ols_sample(NUM_AGENTS, true_params)
    crit = "ols"
    objective = functools.partial(_criterion, endog, exog, crit)
    calculated = minimize_pounders_np(
        objective,
        start_params["value"].to_numpy(),
        bounds=bounds,
        grtol=False,
        gttol=False,
    )

    assert (
        calculated["conv"] == "gatol below critical value"
        or calculated["conv"] == "step size small"
    )
    if calculated["conv"] == 3:
        assert calculated["sol"][2] < 1e-4


def test_gttol():
    np.random.seed(5476)
    true_params = get_random_params(2, 0.3, 0.4, 0, 0.3)
    start_params = get_random_params(2, 0.1, 0.2)
    bounds = tuple(true_params[["lower", "upper"]].to_numpy().T)

    exog, endog = _simulate_ols_sample(NUM_AGENTS, true_params)
    crit = "ols"
    objective = functools.partial(_criterion, endog, exog, crit)
    calculated = minimize_pounders_np(
        objective,
        start_params["value"].to_numpy(),
        bounds=bounds,
        grtol=False,
        gatol=False,
    )

    assert (
        calculated["conv"] == "gttol below critical value"
        or calculated["conv"] == "step size small"
    )

    if calculated["conv"] == 5:
        assert calculated["sol"][2] < 1


def test_tol():
    np.random.seed(5477)
    true_params = get_random_params(2, 0.3, 0.4, 0, 0.3)
    start_params = get_random_params(2, 0.1, 0.2)
    bounds = tuple(true_params[["lower", "upper"]].to_numpy().T)

    exog, endog = _simulate_ols_sample(NUM_AGENTS, true_params)
    crit = "ols"
    objective = functools.partial(_criterion, endog, exog, crit)
    calculated = minimize_pounders_np(
        objective,
        start_params["value"].to_numpy(),
        bounds=bounds,
        gatol=1e-7,
        grtol=1e-7,
        gttol=1e-9,
    )

    if calculated["conv"] == 3:
        assert calculated["sol"][2] < 0.00000001
    elif calculated["conv"] == 4:
        assert calculated["sol"][2] / calculated["sol"][1] < 0.00000001


def test_exception():
    np.random.seed(5478)
    with pytest.raises(Exception):
        minimize_pounders_np(_return_exception, 0)


def _criterion(endog, exog, crit, x):
    if crit == "nonlinear":
        return endog - np.exp(-x[0] * exog) / (x[1] + x[2] * exog)
    elif crit == "ols":
        return endog - x[0] - x[1] * exog


def _return_exception(x):
    raise (Exception)


def _simulate_sample(num_agents, paras, error_term_high=0.5):
    exog = np.random.uniform(0, 1, num_agents)
    error_term = np.random.normal(0, error_term_high, num_agents)
    endog = (
        np.exp(-paras.at[0, "value"] * exog)
        / (paras.at[1, "value"] + paras.at[2, "value"] * exog)
        + error_term
    )

    return exog, endog


def _simulate_ols_sample(num_agents, paras):
    exog = np.random.uniform(-5, 5, num_agents)
    error_term = np.random.normal(0, 1, num_agents)
    endog = paras.at[0, "value"] + paras.at[1, "value"] * exog + error_term

    return exog, endog
