import numpy as np
import pandas as pd
import pytest
from estimagic.inference.bootstrap import bootstrap
from estimagic.inference.bootstrap import bootstrap_from_outcomes
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
    out["estimates_pytree_x1"] = [pd.Series(row, index=["x1"]) for row in y[:, 0]]

    return out


@pytest.fixture
def expected():
    out = {}

    z = np.array([[2.55, 0.5701, 2, 3.225], [6.95, 1.0665, 5.775, 8]])
    cov = np.array([[0.325, -0.60625], [-0.60625, 1.1375]])
    p_values = np.array([0.0, 0.0])
    lower_ci = np.array([2, 5.775])
    upper_ci = np.array([3.225, 8])

    out["summary"] = pd.DataFrame(
        z, columns=["mean", "std", "lower_ci", "upper_ci"], index=["x1", "x2"]
    )
    out["lower_ci"] = pd.Series(lower_ci, index=["x1", "x2"])
    out["upper_ci"] = pd.Series(upper_ci, index=["x1", "x2"])
    out["lower_ci_x1"] = pd.Series(lower_ci[0], index=["x1"])
    out["upper_ci_x1"] = pd.Series(upper_ci[0], index=["x1"])
    out["cov"] = pd.DataFrame(cov, columns=["x1", "x2"], index=["x1", "x2"])
    out["se"] = pd.Series(np.sqrt(np.diagonal(cov)), index=["x1", "x2"])
    out["p_values"] = pd.Series(p_values, index=["x1", "x2"])
    out["p_value_x1"] = pd.Series(p_values[0], index=["x1"])

    return out


def _outcome_func(data, shift=0):
    """Compute column means.

    Args:
        data (pd.Series or pd.DataFrame): The data set.
        shift (float): Scalar that is added to the column means.

    Returns:
        pd.Series: Series where the k-th row corresponds to the mean
            of the k-th column of the input data.
    """
    # Return pd.Series when .mean() is applied to a Series
    # Only applying .mean() to a pd.Series would yield a float
    return pd.DataFrame(data).mean(axis=0) + shift


@pytest.mark.parametrize("shift", [0, 10, -10])
def test_bootstrap_with_outcome_kwargs(shift, setup):
    result = bootstrap(
        data=setup["df"],
        outcome=_outcome_func,
        seed=123,
        outcome_kwargs={"shift": shift},
    )

    expected = pd.Series([2.5, 7.0], index=["x1", "x2"])
    ase(result.base_outcome, expected + shift)


def test_bootstrap_from_outcomes(setup, expected):

    result = bootstrap_from_outcomes(
        base_outcome=_outcome_func(setup["df"]),
        bootstrap_outcomes=setup["estimates_pytree"],
    )

    outcomes = result.outcomes()
    lower_ci, upper_ci = result.ci()
    covariance = result.cov()
    standard_errors = result.se()

    with pytest.raises(NotImplementedError) as error:
        assert result._p_values()
    assert str(error.value) == "Bootstrapped p-values are not implemented yet."

    with pytest.raises(NotImplementedError) as error:
        assert result._summary()
    assert str(error.value) == "summary is not implemented yet."

    # Use rounding to adjust precision and ensure reproducibility accross all
    # supported pandas versions.
    for i in range(len(outcomes)):
        ase(outcomes[i], setup["estimates_pytree"][i])

    ase(lower_ci.round(2), expected["lower_ci"].round(2))
    ase(upper_ci.round(2), expected["upper_ci"].round(2))
    afe(covariance.round(2), expected["cov"].round(2))
    ase(standard_errors.round(2), expected["se"].round(2))


def test_bootstrap_from_outcomes_private_methods(setup, expected):

    result = bootstrap_from_outcomes(
        base_outcome=_outcome_func(setup["df"]),
        bootstrap_outcomes=setup["estimates_pytree"],
    )

    outcomes = result._outcomes
    lower_ci, upper_ci = result._ci
    covariance = result._cov
    standard_errors = result._se

    with pytest.raises(NotImplementedError) as error:
        assert result._p_values()
    assert str(error.value) == "Bootstrapped p-values are not implemented yet."

    with pytest.raises(NotImplementedError) as error:
        assert result._summary()
    assert str(error.value) == "summary is not implemented yet."

    for i in range(len(outcomes)):
        ase(outcomes[i], setup["estimates_pytree"][i])

    ase(lower_ci.round(2), expected["lower_ci"].round(2))
    ase(upper_ci.round(2), expected["upper_ci"].round(2))
    afe(covariance.round(2), expected["cov"].round(2))
    ase(standard_errors.round(2), expected["se"].round(2))


def test_bootstrap_from_outcomes_single_outcome(setup, expected):

    result = bootstrap_from_outcomes(
        base_outcome=_outcome_func(setup["df"]["x1"]),
        bootstrap_outcomes=setup["estimates_pytree_x1"],
    )

    outcomes = result.outcomes()
    lower_ci, upper_ci = result.ci()

    for i in range(len(outcomes)):
        ase(outcomes[i], setup["estimates_pytree_x1"][i])

    ase(lower_ci.round(2), expected["lower_ci_x1"].round(2))
    ase(upper_ci.round(2), expected["upper_ci_x1"].round(2))


@pytest.mark.parametrize("input_type", ["arr", "df", "dict"])
def test_wrong_input_types(input_type, setup):
    with pytest.raises(TypeError):
        assert bootstrap_from_outcomes(
            base_outcome=_outcome_func(setup["df"]),
            bootstrap_outcomes=setup["estimates_" + input_type],
        )


@pytest.mark.parametrize("return_type", ["array", "dataframe", "pytree"])
def test_cov_correct_return_type(return_type, setup):
    result = bootstrap_from_outcomes(
        base_outcome=_outcome_func(setup["df"]),
        bootstrap_outcomes=setup["estimates_pytree"],
    )
    _ = result.cov(return_type=return_type)


def test_cov_wrong_return_type(setup):
    result = bootstrap_from_outcomes(
        base_outcome=_outcome_func(setup["df"]),
        bootstrap_outcomes=setup["estimates_pytree"],
    )

    expected_msg = "return_type must be one of pytree, array, or dataframe, not dict."

    with pytest.raises(TypeError) as error:
        assert result.cov(return_type="dict")

    assert str(error.value) == expected_msg
