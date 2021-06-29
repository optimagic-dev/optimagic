import functools

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal as afe

from estimagic.inference.bootstrap_estimates import (
    _get_bootstrap_estimates_from_indices,
)
from estimagic.inference.bootstrap_estimates import get_bootstrap_estimates


@pytest.fixture
def data():
    df = pd.DataFrame([[1, 10], [2, 7], [3, 6], [4, 5]], columns=["x1", "x2"])
    return df


def test_get_bootstrap_estimates_runs(data):
    get_bootstrap_estimates(
        data=data,
        outcome=functools.partial(np.mean, axis=0),
        n_draws=5,
    )


def test_bootstrap_estimates_from_indices_without_errors(data):
    calculated = _get_bootstrap_estimates_from_indices(
        indices=[np.array([1, 3]), np.array([0, 2])],
        data=data,
        outcome=functools.partial(np.mean, axis=0),
        n_cores=1,
        error_handling="raise",
    )

    expected = pd.DataFrame([[3.0, 6.0], [2, 8]], columns=["x1", "x2"])
    afe(calculated, expected)


def test_get_bootstrap_estimates_with_error_and_raise(data):
    def _raise_assertion_error(data):
        assert 1 == 2

    with pytest.raises(AssertionError):
        get_bootstrap_estimates(
            data=data, outcome=_raise_assertion_error, n_draws=2, error_handling="raise"
        )


def test_get_bootstrap_estimates_with_all_errors_and_continue(data):
    def _raise_assertion_error(data):
        assert 1 == 2

    with pytest.warns(UserWarning):
        with pytest.raises(RuntimeError):
            get_bootstrap_estimates(
                data=data,
                outcome=_raise_assertion_error,
                n_draws=2,
                error_handling="continue",
            )


def test_get_bootstrap_estimates_with_some_errors_and_continue(data):
    def _raise_assertion_error_sometimes(data):
        assert np.random.uniform() > 0.5
        return data.mean()

    with pytest.warns(UserWarning):
        res = get_bootstrap_estimates(
            data=data,
            outcome=_raise_assertion_error_sometimes,
            n_draws=100,
            error_handling="continue",
            seed=123,
        )

    assert 30 <= len(res) <= 70
