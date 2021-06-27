import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal as aaae
from pandas.testing import assert_series_equal as ase

from estimagic.inference.bootstrap_ci import _jackknife
from estimagic.inference.bootstrap_ci import check_inputs
from estimagic.inference.bootstrap_ci import compute_ci
from estimagic.inference.bootstrap_ci import concatenate_functions


@pytest.fixture
def setup():
    out = {}

    out["df"] = pd.DataFrame(
        np.array([[1, 10], [2, 7], [3, 6], [4, 5]]), columns=["x1", "x2"]
    )

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

    out["concat_function_pandas"] = pd.Series(
        [2.5, 7, 5, 14], index=["x1", "x2", "x1", "x2"]
    )
    out["concat_function_numpy"] = np.array([2.5, 7, 5, 14])

    return out


def g(data):
    return data.mean(axis=0)


def g_arr(data):
    return np.array(data.mean(axis=0))


def h(data):
    return 2 * g(data)


def h_arr(data):
    return 2 * g_arr(data)


def test_percentile_ci(setup, expected):
    percentile_ci = compute_ci(
        setup["df"], g, setup["estimates"], ci_method="percentile"
    )
    aaae(percentile_ci, expected["percentile_ci"])


def test_normal_ci(setup, expected):
    normal_ci = compute_ci(setup["df"], g, setup["estimates"], ci_method="normal")
    aaae(normal_ci, expected["normal_ci"])


def test_basic_ci(setup, expected):
    basic_ci = compute_ci(setup["df"], g, setup["estimates"], ci_method="basic")
    aaae(basic_ci, expected["basic_ci"])


def test_bc_ci(setup, expected):
    bc_ci = compute_ci(setup["df"], g, setup["estimates"], ci_method="bc")
    aaae(bc_ci, expected["bc_ci"])


def test_bca_ci(setup, expected):
    bca_ci = compute_ci(setup["df"], g, setup["estimates"], ci_method="bca")
    aaae(bca_ci, expected["bca_ci"])


def test_t_ci(setup, expected):
    t_ci = compute_ci(setup["df"], g, setup["estimates"], ci_method="t")
    aaae(t_ci, expected["t_ci"])


def test_jackknife(setup, expected):
    jk_estimates = _jackknife(setup["df"], g)
    aaae(jk_estimates, expected["jk_estimates"])


def test_check_inputs_data(setup, expected):
    data = "this is not a data frame"
    with pytest.raises(ValueError) as excinfo:
        check_inputs(data=data)
    assert "Input 'data' must be DataFrame." == str(excinfo.value)


def test_check_inputs_cluster_by(setup, expected):
    cluster_by = "this is not a column name of df"
    with pytest.raises(ValueError) as excinfo:
        check_inputs(data=setup["df"], cluster_by=cluster_by)
    assert "Input 'cluster_by' must be None or a column name of DataFrame." == str(
        excinfo.value
    )


def test_check_inputs_ci_method(setup, expected):
    ci_method = 4
    with pytest.raises(ValueError) as excinfo:
        check_inputs(data=setup["df"], ci_method=ci_method)
    expected_msg = (
        "ci_method must be 'percentile', 'bc',"
        f" 'bca', 't', 'basic' or 'normal', '{ci_method}'"
        f" was supplied"
    )
    assert str(excinfo.value) == expected_msg


def test_check_inputs_alpha(setup, expected):
    alpha = 666
    with pytest.raises(ValueError) as excinfo:
        check_inputs(data=setup["df"], alpha=alpha)
    assert "Input 'alpha' must be in [0,1]." == str(excinfo.value)


def test_concatenate_functions(setup, expected):

    f = concatenate_functions([g, h], setup["df"])
    f_mixed = concatenate_functions([g_arr, h], setup["df"])
    f_arr = concatenate_functions([g_arr, h_arr], setup["df"])

    ase(f(setup["df"]), expected["concat_function_pandas"])
    aaae(f_mixed(setup["df"]), expected["concat_function_numpy"])
    aaae(f_arr(setup["df"]), expected["concat_function_numpy"])
