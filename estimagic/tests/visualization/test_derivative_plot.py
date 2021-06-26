import itertools

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from pandas.testing import assert_series_equal

from estimagic.visualization.derivative_plot import (
    _get_dims_from_data_if_no_user_input_else_forward,
)
from estimagic.visualization.derivative_plot import (
    _select_derivative_with_minimal_error,
)
from estimagic.visualization.derivative_plot import (
    _select_eval_with_lowest_and_highest_step,
)


def test__get_dims_from_data_if_no_user_input_else_forward():
    dim_x = [0, 1, 2, 3, 4]
    dim_f = [0, 1, 2]
    data = itertools.product(dim_x, dim_f)
    df = pd.DataFrame(data, columns=["dim_x", "dim_f"])

    # with input
    got_with_input = _get_dims_from_data_if_no_user_input_else_forward(
        df, [0, 1], [1, 2]
    )
    expected_with_input = (np.array([0, 1]), np.array([1, 2]))

    for got, expected in zip(got_with_input, expected_with_input):
        assert_array_equal(got, expected)

    # without input
    got_without_input = _get_dims_from_data_if_no_user_input_else_forward(
        df, None, None
    )
    expected_without_input = (np.array([0, 1, 2, 3, 4]), np.array([0, 1, 2]))

    for got, expected in zip(got_without_input, expected_without_input):
        assert_array_equal(got, expected)


def test__select_derivative_with_minimal_error():
    data = [
        ["forward", 0, 0, 0, 0.1, 1],
        ["forward", 1, 0, 0, 0.2, 2],
        ["central", 0, 0, 0, 0.05, 1.1],
        ["central", 1, 0, 0, 0.07, 1.2],
    ]
    df_jac_cand = pd.DataFrame(
        data, columns=["method", "num_term", "dim_x", "dim_f", "err", "der"]
    )
    df_jac_cand = df_jac_cand.set_index(["method", "num_term", "dim_x", "dim_f"])
    got = _select_derivative_with_minimal_error(df_jac_cand)
    expected = pd.DataFrame([[0, 0, 1.1]], columns=["dim_x", "dim_f", "der"])
    expected = expected.set_index(["dim_x", "dim_f"])["der"]
    assert_series_equal(got, expected)


def test__select_derivative_with_minimal_error_given():
    data = [
        ["forward", 0, 0, 0, 0.1, 1],
        ["forward", 1, 0, 0, 0.2, 2],
        ["central", 0, 0, 0, 0.05, 1.1],
        ["central", 1, 0, 0, 0.07, 1.2],
    ]
    df_jac_cand = pd.DataFrame(
        data, columns=["method", "num_term", "dim_x", "dim_f", "err", "der"]
    )
    df_jac_cand = df_jac_cand.set_index(["method", "num_term", "dim_x", "dim_f"])
    got = _select_derivative_with_minimal_error(df_jac_cand, given_method=True)
    expected = pd.DataFrame(
        [["forward", 0, 0, 1], ["central", 0, 0, 1.1]],
        columns=["method", "dim_x", "dim_f", "der"],
    )
    expected = expected.set_index(["method", "dim_x", "dim_f"])["der"].sort_index()
    assert_series_equal(got, expected)


def test__select_eval_with_lowest_and_highest_step():
    data = [
        [1, 1, 0, 0, 0.0, 1.1],
        [1, 2, 0, 0, 0.1, 0.2],
        [1, 3, 0, 0, 0.2, -0.5],
        [1, 4, 0, 0, 0.3, 10],
        [1, 5, 0, 0, 0.4, np.nan],
    ]
    df_evals = pd.DataFrame(
        data, columns=["sign", "step_number", "dim_x", "dim_f", "step", "eval"]
    )
    df_evals = df_evals.set_index(["sign", "step_number", "dim_x", "dim_f"])

    got = _select_eval_with_lowest_and_highest_step(df_evals, sign=1, dim_x=0, dim_f=0)
    expected = np.array([[0.0, 1.1], [0.3, 10]])

    assert_array_equal(got, expected)
