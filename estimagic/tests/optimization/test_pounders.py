"""
Testing the wrapper around pounders
"""
"""

"""
import numpy as np
from functools import partial

from estimagic.optimization.solver_pounders import solve


def test_robustness_1():
    # get random args
    paras = np.random.uniform(size=3)
    start = np.random.uniform(size=3)
    num_agents = 10000
    objective, x = _set_up_test_1(paras, start, num_agents)
    len_out = len(objective(x))
    out = solve(objective, x, len_out), start, paras

    return out


def test_robustness_2():
    # get random args
    paras = np.random.uniform(size=2)
    start = np.random.uniform(size=2)

    # Simulate a sample
    num_agents = 10000
    objective, x, exog, endog = _set_up_test_2_ols(paras, start, num_agents)

    # Obtain result with Pounders
    len_out = len(objective(x))
    out = solve(objective, x, len_out), start, paras

    # Obtain result via ols
    X = np.concatenate(
        (np.ones(len(exog)).reshape(len(exog), 1), exog.reshape(len(exog), 1)), axis=1
    ).reshape(len(exog), 2)
    y = endog.reshape(len(endog), 1)
    ols = np.linalg.lstsq(X, y)

    # compare
    np.testing.assert_almost_equal(
        ols[0], np.array(out[0]["solution"]).reshape(2, 1), decimal=1
    )

    return out


def test_box_constr():
    paras = np.random.uniform(0.3, 0.4, size=2)
    start = np.random.uniform(0.1, 0.2, size=2)
    bounds = [[0, 0], [0.3, 0.3]]

    num_agents = 10000
    objective, x = _set_up_test_2(paras, start, num_agents)
    len_out = len(objective(x))
    out = solve(objective, x, len_out, bounds=bounds)
    assert 0 <= out["solution"][0] <= 0.3
    assert 0 <= out["solution"][1] <= 0.3


def test_max_iters():
    paras = np.random.uniform(0.3, 0.4, size=2)
    start = np.random.uniform(0.1, 0.2, size=2)
    bounds = [[0, 0], [0.3, 0.3]]
    num_agents = 10000
    objective, x = _set_up_test_2(paras, start, num_agents)
    len_out = len(objective(x))
    out = solve(objective, x, len_out, bounds=bounds, max_iterations=25)

    assert out["conv"] == "user defined" or out["conv"] == "step size small"
    if out["conv"] == 8:
        assert out["sol"][0] == 25


def test_grtol():
    paras = np.random.uniform(0.3, 0.4, size=2)
    start = np.random.uniform(0.1, 0.2, size=2)
    bounds = [[0, 0], [0.3, 0.3]]

    num_agents = 10000
    objective, x = _set_up_test_2(paras, start, num_agents)
    len_out = len(objective(x))
    out = solve(
        objective,
        x,
        len_out,
        tol={"grtol": 10, "gatol": 1, "gttol": 1},
        bounds=bounds,
        gatol=False,
        gttol=False,
    )

    assert (
        out["conv"] == "grtol below critical value" or out["conv"] == "step size small"
    )

    if out["conv"] == 4:
        assert out["sol"][2] / out["sol"][1] < 10


def test_gatol():
    paras = np.random.uniform(0.3, 0.4, size=2)
    start = np.random.uniform(0.1, 0.2, size=2)
    bounds = [[0, 0], [0.3, 0.3]]

    num_agents = 10000
    objective, x = _set_up_test_2(paras, start, num_agents)
    len_out = len(objective(x))
    out = solve(
        objective,
        x,
        len_out,
        tol={"grtol": 1, "gatol": 0.00001, "gttol": 1},
        bounds=bounds,
        grtol=False,
        gttol=False,
    )
    assert (
        out["conv"] == "gatol below critical value" or out["conv"] == "step size small"
    )

    if out["conv"] == 3:
        assert out["sol"][2] < 0.00001


def test_gttol():
    paras = np.random.uniform(0.3, 0.4, size=2)
    start = np.random.uniform(0.1, 0.2, size=2)
    bounds = [[0, 0], [0.3, 0.3]]

    num_agents = 10000
    objective, x = _set_up_test_2(paras, start, num_agents)
    len_out = len(objective(x))
    out = solve(
        objective,
        x,
        len_out,
        tol={"grtol": 1, "gatol": 1, "gttol": 1},
        bounds=bounds,
        grtol=False,
        gatol=False,
    )
    assert (
        out["conv"] == "gttol below critical value" or out["conv"] == "step size small"
    )

    if out["conv"] == 5:
        assert out["sol"][2] < 1


def test_tol():
    paras = np.random.uniform(0.3, 0.4, size=2)
    start = np.random.uniform(0.1, 0.2, size=2)
    bounds = [[0, 0], [0.3, 0.3]]

    num_agents = 10000
    objective, x = _set_up_test_2(paras, start, num_agents)
    len_out = len(objective(x))
    out = solve(
        objective,
        x,
        len_out,
        bounds=bounds,
        tol={"gatol": 0.00000001, "grtol": 0.00000001, "gttol": 0.0000000001},
    )

    if out["conv"] == 3:
        assert out["sol"][2] < 0.00000001
    elif out["conv"] == 4:
        assert out["sol"][2] / out["sol"][1] < 0.00000001


def _set_up_test_1(paras, start, num_agents):
    """
    This is a bit ad hoc

    """
    # Simulate values
    exog, endog = _simulate_sample(num_agents, paras)
    # Initialize class container
    func = _return_obj_func(_return_dev, endog, exog)
    return func, start


def _set_up_test_2(paras, start, num_agents):
    """
    """
    exog, endog = _simulate_ols_sample(num_agents, paras)
    func = _return_obj_func(_return_dev_ols, endog, exog)
    return func, start


def _set_up_test_2_ols(paras, start, num_agents):
    """
    """
    exog, endog = _simulate_ols_sample(num_agents, paras)
    func = _return_obj_func(_return_dev_ols, endog, exog)
    return func, start, exog, endog


def _return_dev(endog, exog, x):
    dev = (endog - np.exp(-x[0] * exog) / (x[1] + x[2] * exog)) ** 2
    return dev


def _return_obj_func(func, endog, exog):
    out = partial(func, endog, exog)
    return out


def _return_dev_ols(endog, exog, x):
    dev = (endog - x[0] - x[1] * exog) ** 2
    return dev


def _simulate_sample(num_agents, paras):
    exog = np.random.uniform(0, 1, num_agents)
    error_term = np.random.normal(0, 0.5, num_agents)
    endog = np.exp(-paras[0] * exog) / (paras[1] + paras[2] * exog) + error_term
    return exog, endog


def _simulate_ols_sample(num_agents, paras):
    exog = np.random.uniform(-5, 5, num_agents)
    error_term = np.random.normal(0, 1, num_agents)
    endog = paras[0] + paras[1] * exog + error_term
    return exog, endog
