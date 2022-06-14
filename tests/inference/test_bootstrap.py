import numpy as np
import pandas as pd
import pytest
from estimagic.inference.bootstrap import bootstrap_from_outcomes
from numpy.testing import assert_array_almost_equal as aaae
from pandas.testing import assert_frame_equal as afe
from pandas.testing import assert_series_equal as ase


@pytest.fixture
def setup():
    out = {}

    out["df"] = pd.DataFrame(
        np.array([[1, 10], [2, 7], [3, 6], [4, 5]]), columns=["x1", "x2"]
    )

    y = np.array([[2.0, 8.0], [2.0, 8.0], [2.5, 7.0], [3.0, 6.0], [3.25, 5.75]])
    out["estimates_arr"] = y
    out["estimates_df"] = pd.DataFrame(y, columns=["x1", "x2"])
    out["estimates_dict"] = {"x1": [2, 2, 2.5, 3, 3.25], "x2": [8, 8, 7, 6, 5.75]}
    out["estimates_pytree"] = [pd.Series(row, index=["x1", "x2"]) for row in y]

    return out


@pytest.fixture
def expected():
    out = {}

    z = np.array([[2.55, 0.5701, 2, 3.225], [6.95, 1.0665, 5.775, 8]])
    cov = np.array([[0.325, -0.60625], [-0.60625, 1.1375]])
    p_values = np.array([1.15831306e-05, 5.26293752e-11])
    lower_ci = np.array([2, 5.775])
    upper_ci = np.array([3.225, 8])

    out["summary"] = pd.DataFrame(
        z, columns=["mean", "std", "lower_ci", "upper_ci"], index=["x1", "x2"]
    )
    out["lower_ci"] = pd.Series(lower_ci, index=["x1", "x2"])
    out["upper_ci"] = pd.Series(upper_ci, index=["x1", "x2"])
    out["cov"] = pd.DataFrame(cov, columns=["x1", "x2"], index=["x1", "x2"])
    out["se"] = pd.Series(np.sqrt(np.diagonal(cov)), index=["x1", "x2"])
    out["p_values"] = pd.Series(p_values, index=["x1", "x2"])

    return out


def g(data):
    return data.mean(axis=0)


def test_bootstrap_from_outcomes(setup, expected):

    result = bootstrap_from_outcomes(
        base_outcome=g(setup["df"]),
        bootstrap_outcomes=setup["estimates_pytree"],
    )

    outcomes = result.outcomes()
    lower_ci, upper_ci = result.ci()
    covariance = result.cov()
    standard_errors = result.se()

    # Use rounding to adjust precision because there is no other way of handling this
    # such that it is compatible across all supported pandas versions.
    aaae(outcomes, setup["estimates_pytree"])
    ase(lower_ci.round(2), expected["lower_ci"].round(2))
    ase(upper_ci.round(2), expected["upper_ci"].round(2))
    afe(covariance.round(2), expected["cov"].round(2))
    ase(standard_errors.round(2), expected["se"].round(2))


@pytest.mark.parametrize("input_type", ["arr", "df", "dict"])
def test_wrong_input_types(input_type, setup):
    with pytest.raises(TypeError):
        assert bootstrap_from_outcomes(
            base_outcome=g(setup["df"]),
            bootstrap_outcomes=setup["estimates_" + input_type],
        )


@pytest.mark.parametrize("return_type", ["array", "dataframe", "pytree"])
def test_cov_correct_return_type(return_type, setup):
    result = bootstrap_from_outcomes(
        base_outcome=g(setup["df"]),
        bootstrap_outcomes=setup["estimates_pytree"],
    )
    _ = result.cov(return_type=return_type)


def test_cov_wrong_return_type(setup):
    result = bootstrap_from_outcomes(
        base_outcome=g(setup["df"]),
        bootstrap_outcomes=setup["estimates_pytree"],
    )

    expected_msg = "return_type must be one of pytree, array, or dataframe, not dict."

    with pytest.raises(TypeError) as error:
        assert result.cov(return_type="dict")

    assert str(error.value) == expected_msg
