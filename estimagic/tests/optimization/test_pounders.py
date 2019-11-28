"""Test the wrapper around pounders."""
import functools
import sys

import numpy as np
import pandas as pd
import pytest

from estimagic.optimization.pounders import minimize_pounders

pytest.skip("Requires decorators.", allow_module_level=True)

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="Pounders is not supported on Windows."
)


NUM_AGENTS = 10_000


def get_random_params(length, low=0, high=1):
    params = pd.DataFrame(
        {
            "value": np.random.uniform(low, high, size=length),
            "lower": -np.inf,
            "upper": np.inf,
            "_internal_free": True,
        }
    )

    return params


def test_robustness_1():
    np.random.seed(5470)
    true_paras = get_random_params(3)
    start = get_random_params(3)

    exog, endog = _simulate_sample(NUM_AGENTS, true_paras)
    objective = functools.partial(_nonlinear_criterion, endog, exog)
    minimize_pounders(objective, start, objective, start, {})


def test_robustness_2():
    np.random.seed(5471)
    true_params = get_random_params(2)
    start_params = get_random_params(2)

    exog, endog = _simulate_ols_sample(NUM_AGENTS, true_params)
    objective = functools.partial(_ols_criterion, endog, exog)
    results = minimize_pounders(objective, start_params, objective, start_params, {})
    calculated = results["x"]

    x = np.column_stack([np.ones_like(exog), exog])
    y = endog.reshape(len(endog), 1)
    expected = np.linalg.lstsq(x, y, rcond=None)[0].flatten()

    np.testing.assert_almost_equal(calculated, expected, decimal=2)


def test_box_constr():
    np.random.seed(5472)
    true_params = get_random_params(2, 0.3, 0.4)
    true_params["lower"] = 0
    true_params["upper"] = 0.3

    start_params = true_params.copy()
    start_params["value"] = get_random_params(2, 0.1, 0.2)["value"]

    exog, endog = _simulate_ols_sample(NUM_AGENTS, true_params)
    objective = functools.partial(_ols_criterion, endog, exog)
    calculated = minimize_pounders(objective, start_params, objective, start_params, {})
    assert 0 <= calculated["x"][0] <= 0.3
    assert 0 <= calculated["x"][1] <= 0.3


def test_max_iters():
    np.random.seed(5473)
    true_params = np.random.uniform(0.3, 0.4, size=2)
    start_params = np.random.uniform(0.1, 0.2, size=2)
    bounds = [[0, 0], [0.3, 0.3]]

    exog, endog = _simulate_ols_sample(NUM_AGENTS, true_params)
    objective = functools.partial(_ols_criterion, endog, exog)
    len_out = len(objective(start_params))
    calculated = minimize_pounders(
        objective, start_params, len_out, bounds=bounds, max_iterations=25
    )

    assert (
        calculated["conv"] == "user defined" or calculated["conv"] == "step size small"
    )
    if calculated["conv"] == 8:
        assert calculated["sol"][0] == 25


def test_grtol():
    np.random.seed(5474)
    true_params = np.random.uniform(0.3, 0.4, size=2)
    start_params = np.random.uniform(0.1, 0.2, size=2)
    bounds = [[0, 0], [0.3, 0.3]]

    exog, endog = _simulate_ols_sample(NUM_AGENTS, true_params)
    objective = functools.partial(_ols_criterion, endog, exog)
    len_calculated = len(objective(start_params))
    calculated = minimize_pounders(
        objective, start_params, len_calculated, bounds=bounds, gatol=False, gttol=False
    )

    assert (
        calculated["conv"] == "grtol below critical value"
        or calculated["conv"] == "step size small"
    )

    if calculated["conv"] == 4:
        assert calculated["sol"][2] / calculated["sol"][1] < 10


def test_gatol():
    np.random.seed(5475)
    true_params = np.random.uniform(0.3, 0.4, size=2)
    start_params = np.random.uniform(0.1, 0.2, size=2)
    bounds = [[0, 0], [0.3, 0.3]]

    exog, endog = _simulate_ols_sample(NUM_AGENTS, true_params)
    objective = functools.partial(_ols_criterion, endog, exog)
    len_out = len(objective(start_params))
    calculated = minimize_pounders(
        objective, start_params, len_out, bounds=bounds, grtol=False, gttol=False
    )

    assert (
        calculated["conv"] == "gatol below critical value"
        or calculated["conv"] == "step size small"
    )
    if calculated["conv"] == 3:
        assert calculated["sol"][2] < 1e-4


def test_gttol():
    np.random.seed(5476)
    true_params = np.random.uniform(0.3, 0.4, size=2)
    start_params = np.random.uniform(0.1, 0.2, size=2)
    bounds = [[0, 0], [0.3, 0.3]]

    exog, endog = _simulate_ols_sample(NUM_AGENTS, true_params)
    objective = functools.partial(_ols_criterion, endog, exog)
    len_out = len(objective(start_params))
    calculated = minimize_pounders(
        objective, start_params, len_out, bounds=bounds, grtol=False, gatol=False
    )

    assert (
        calculated["conv"] == "gttol below critical value"
        or calculated["conv"] == "step size small"
    )

    if calculated["conv"] == 5:
        assert calculated["sol"][2] < 1


def test_tol():
    np.random.seed(5477)
    true_params = np.random.uniform(0.3, 0.4, size=2)
    start_params = np.random.uniform(0.1, 0.2, size=2)
    bounds = [[0, 0], [0.3, 0.3]]

    exog, endog = _simulate_ols_sample(NUM_AGENTS, true_params)
    objective = functools.partial(_ols_criterion, endog, exog)
    len_out = len(objective(start_params))
    calculated = minimize_pounders(
        objective,
        start_params,
        len_out,
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
        minimize_pounders(_return_exception, 0)


def _nonlinear_criterion(endog, exog, x):
    return (
        endog
        - np.exp(-x.at[0, "value"] * exog)
        / (x.at[1, "value"] + x.at[2, "value"] * exog)
    ) ** 2


def _ols_criterion(endog, exog, x):
    return (endog - x.at[0, "value"] - x.at[1, "value"] * exog) ** 2


def _return_exception(x):
    raise (Exception)


def _simulate_sample(num_agents, paras):
    exog = np.random.uniform(0, 1, num_agents)
    error_term = np.random.normal(0, 0.5, num_agents)
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
