import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal as aae
from pandas.testing import assert_frame_equal as afe

from estimagic.inference.bootstrap import _check_inputs
from estimagic.inference.bootstrap_ci import _jackknife
from estimagic.inference.bootstrap_ci import compute_ci
from estimagic.inference.bootstrap_estimates import _get_cluster_index
from estimagic.inference.bootstrap_estimates import get_bootstrap_estimates

FACTORS = list("cni")


@pytest.fixture
def setup():
    out = {}

    out["df"] = pd.DataFrame(
        np.array([[1, 10], [2, 7], [3, 6], [4, 5]]), columns=["x1", "x2"]
    )

    out["cluster_df"] = pd.DataFrame(
        np.array([[1, 10, 2], [2, 7, 2], [3, 6, 1], [4, 5, 2]]),
        columns=["x1", "x2", "stratum"],
    )

    out["seeds"] = [1, 2, 3, 4, 5]

    x = np.array([[2.0, 8.0], [2.0, 8.0], [2.5, 7.0], [3.0, 6.0], [3.25, 5.75]])
    out["estimates"] = pd.DataFrame(x, columns=["x1", "x2"])

    return out


@pytest.fixture
def expected():
    out = {}

    out["percentile_ci"] = np.array([[2, 3.225], [5.775, 8.0]])

    out["normal_ci"] = np.array(
        [
            [1.5006105396891194, 3.499389460310881],
            [5.130313521781885, 8.869686478218114],
        ]
    )

    out["basic_ci"] = np.array([[1.775, 3.0], [6.0, 8.225]])

    out["bc_ci"] = np.array([[2, 3.2342835077057543], [5.877526959881923, 8]])

    out["bca_ci"] = np.array([[2, 3.2058815797003826], [5.9404612332752915, 8]])

    out["t_ci"] = np.array([[1.775, 3], [6.0, 8.225]])

    out["jk_estimates"] = np.array(
        [
            [3, 6],
            [2.6666666666666665, 7],
            [2.3333333333333335, 7.333333333333333],
            [2, 7.666666666666667],
        ]
    )

    out["cluster_index"] = [np.array([0, 1, 3]), np.array([2])]

    return out


def g(data):
    return data.mean(axis=0)


def test_percentile_ci(setup, expected):
    percentile_ci = compute_ci(
        setup["df"], g, setup["estimates"], ci_method="percentile"
    )
    aae(percentile_ci, expected["percentile_ci"])


def test_normal_ci(setup, expected):
    normal_ci = compute_ci(setup["df"], g, setup["estimates"], ci_method="normal")
    aae(normal_ci, expected["normal_ci"])


def test_basic_ci(setup, expected):
    basic_ci = compute_ci(setup["df"], g, setup["estimates"], ci_method="basic")
    aae(basic_ci, expected["basic_ci"])


def test_bc_ci(setup, expected):
    bc_ci = compute_ci(setup["df"], g, setup["estimates"], ci_method="bc")
    aae(bc_ci, expected["bc_ci"])


def test_bca_ci(setup, expected):
    bca_ci = compute_ci(setup["df"], g, setup["estimates"], ci_method="bca")
    aae(bca_ci, expected["bca_ci"])


def test_t_ci(setup, expected):
    t_ci = compute_ci(setup["df"], g, setup["estimates"], ci_method="t")
    aae(t_ci, expected["t_ci"])


def test_get_bootstrap_estimates(setup, expected):
    estimates1 = get_bootstrap_estimates(data=setup["df"], f=g, seeds=setup["seeds"])
    estimates2 = get_bootstrap_estimates(data=setup["df"], f=g, seeds=setup["seeds"])
    afe(estimates1, estimates2)


def test_get_bootstrap_estimates_cluster(setup, expected):
    estimates1 = get_bootstrap_estimates(
        data=setup["cluster_df"], f=g, cluster_by="stratum", seeds=setup["seeds"]
    )
    estimates2 = get_bootstrap_estimates(
        data=setup["cluster_df"], f=g, cluster_by="stratum", seeds=setup["seeds"]
    )
    afe(estimates1, estimates2)


def test_jackknife(setup, expected):
    jk_estimates = _jackknife(setup["df"], g)
    aae(jk_estimates, expected["jk_estimates"])


def test_get_cluster_index(setup, expected):
    cluster_index = _get_cluster_index(setup["cluster_df"], cluster_by="stratum")
    for i in range(len(cluster_index)):
        aae(cluster_index[i], expected["cluster_index"][i])


def test_check_inputs_data(setup, expected):
    data = "this is not a data frame"
    with pytest.raises(ValueError) as excinfo:
        _check_inputs(data=data)
    assert "Input 'data' must be DataFrame." == str(excinfo.value)


def test_check_inputs_cluster_by(setup, expected):
    cluster_by = "this is not a column name of df"
    with pytest.raises(ValueError) as excinfo:
        _check_inputs(data=setup["df"], cluster_by=cluster_by)
    assert "Input 'cluster_by' must be None or a column name of DataFrame." == str(
        excinfo.value
    )


def test_check_inputs_ci_method(setup, expected):
    ci_method = 4
    with pytest.raises(ValueError) as excinfo:
        _check_inputs(data=setup["df"], ci_method=ci_method)
    assert "ci_method must be 'percentile', 'bc',"
    " 'bca', 't', 'basic' or 'normal', '{method}'"
    f" was supplied" == str(excinfo.value)


def test_check_inputs_alpha(setup, expected):
    alpha = 666
    with pytest.raises(ValueError) as excinfo:
        _check_inputs(data=setup["df"], alpha=alpha)
    assert "Input 'alpha' must be in [0,1]." == str(excinfo.value)
