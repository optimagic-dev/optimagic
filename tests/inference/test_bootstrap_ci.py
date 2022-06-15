import itertools

import numpy as np
import pandas as pd
import pytest
from estimagic.inference.bootstrap import bootstrap_from_outcomes
from estimagic.inference.bootstrap_ci import compute_ci
from estimagic.inference.bootstrap_ci import compute_p_values
from estimagic.inference.bootstrap_helpers import check_inputs
from estimagic.parameters.tree_registry import get_registry
from numpy.testing import assert_array_almost_equal as aaae
from pandas.testing import assert_series_equal as ase
from pybaum import tree_just_flatten

# ======================================================================================
# ci
# ======================================================================================


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


def test_check_inputs_alpha(setup):
    alpha = 666
    with pytest.raises(ValueError) as excinfo:
        check_inputs(data=setup["df"], alpha=alpha)
    assert "Input 'alpha' must be in [0,1]." == str(excinfo.value)


# ======================================================================================
# p-values
# ======================================================================================


def h(data):
    return pd.Series(data.x.sum() / data.u.sum(), index=["x"])


@pytest.fixture
def example_data1():
    out = {}

    out["df"] = pd.DataFrame(
        np.array([-3.75, -3, -1.5, 1.25, 2, 2.5, 3, 3.5]), columns=["x"]
    )
    out["estimates"] = np.array([-3, -1.5, 1.25, 2.5, 2.5, 3, 3.5, 3.5])

    return out


@pytest.fixture
def example_data2():
    out = {}

    out["df"] = pd.DataFrame(
        np.array(
            [
                [138, 143],
                [93, 104],
                [61, 69],
                [179, 260],
                [48, 75],
                [37, 63],
                [29, 50],
                [23, 48],
                [30, 111],
                [2, 50],
            ]
        ),
        columns=["u", "x"],
    )
    out["estimates"] = np.array(
        [
            1.485370,
            1.492780,
            1.380952,
            2.223776,
            1.650231,
            1.221790,
            1.590734,
            1.784173,
            1.348977,
            1.726562,
            1.376404,
            1.412081,
            1.726449,
            1.587349,
            1.629423,
            1.290547,
            1.387464,
            1.727749,
            1.371567,
        ]
    )

    return out


TEST_CASES = [("example_data1", g, 0.875), ("example_data2", h, 0.05263158)]


@pytest.mark.parametrize("example_data, outcome, expected", TEST_CASES)
def test_p_values(example_data, outcome, expected, request):
    setup = request.getfixturevalue(example_data)
    registry = get_registry(extended=True)

    def outcome_flat(data):
        return tree_just_flatten(outcome(data), registry=registry)

    base_outcome = outcome_flat(setup["df"])

    pvalue = compute_p_values(base_outcome, setup["estimates"], alpha=0.05)
    assert np.allclose(pvalue, expected)


@pytest.mark.parametrize("example_data, outcome, expected", TEST_CASES)
def test_p_values_from_results(example_data, outcome, expected, request):
    setup = request.getfixturevalue(example_data)

    registry = get_registry(extended=True)
    bootstrap_outcomes = tree_just_flatten(setup["estimates"], registry=registry)

    result = bootstrap_from_outcomes(
        base_outcome=outcome(setup["df"]),
        bootstrap_outcomes=bootstrap_outcomes,
    )

    pvalue = result.p_values()
    ase(pvalue.round(6), pd.Series(expected, index=["x"]).round(6))
