import itertools

import numpy as np
import pandas as pd
import pytest
from estimagic.inference.bootstrap_ci import compute_ci
from estimagic.inference.bootstrap_helpers import check_inputs
from estimagic.parameters.tree_registry import get_registry
from numpy.testing import assert_array_almost_equal as aaae
from pybaum import tree_just_flatten


@pytest.fixture
def setup():
    out = {}

    out["df"] = pd.DataFrame(
        np.array([[1, 10], [2, 7], [3, 6], [4, 5]]), columns=["x1", "x2"]
    )
    out["estimates"] = np.array(
        [[2.0, 8.0], [2.0, 8.0], [2.5, 7.0], [3.0, 6.0], [3.25, 5.75]]
    )

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

    out["t_ci"] = np.array([[1.775, 3], [6.0, 8.225]])

    return out


def g(data):
    return data.mean(axis=0)


def g_dict(data):
    return data.mean(axis=0).to_dict()


def g_arr(data):
    return np.array(data.mean(axis=0))


TEST_CASES = itertools.product(
    [g, g_dict, g_arr],
    ["percentile", "normal", "basic", "bc", "t"],
)


@pytest.mark.parametrize("outcome, method", TEST_CASES)
def test_ci(outcome, method, setup, expected):
    registry = get_registry(extended=True)

    def outcome_flat(data):
        return tree_just_flatten(outcome(data), registry=registry)

    base_outcome = outcome_flat(setup["df"])
    lower, upper = compute_ci(base_outcome, setup["estimates"], ci_method=method)

    aaae(lower, expected[method + "_ci"][:, 0])
    aaae(upper, expected[method + "_ci"][:, 1])


def test_check_inputs_data():
    data = "this is not a data frame"
    with pytest.raises(ValueError) as excinfo:
        check_inputs(data=data)
    assert "Input 'data' must be DataFrame." == str(excinfo.value)


def test_check_inputs_cluster_by(setup):
    cluster_by = "this is not a column name of df"
    with pytest.raises(ValueError) as excinfo:
        check_inputs(data=setup["df"], cluster_by=cluster_by)
    assert "Input 'cluster_by' must be None or a column name of DataFrame." == str(
        excinfo.value
    )


def test_check_inputs_ci_method(setup):
    ci_method = 4
    with pytest.raises(ValueError) as excinfo:
        check_inputs(data=setup["df"], ci_method=ci_method)
    expected_msg = (
        "ci_method must be 'percentile', 'bc',"
        f" 't', 'basic' or 'normal', '{ci_method}'"
        f" was supplied"
    )
    assert str(excinfo.value) == expected_msg


def test_check_inputs_ci_level(setup):
    ci_level = 666
    with pytest.raises(ValueError) as excinfo:
        check_inputs(data=setup["df"], ci_level=ci_level)
    assert "Input 'ci_level' must be in [0,1]." == str(excinfo.value)
