import numpy as np
import pandas as pd
import pytest
from estimagic.differentiation.derivatives import first_derivative
from estimagic.visualization.derivative_plot import (
    _select_derivative_with_minimal_error,
)
from estimagic.visualization.derivative_plot import (
    _select_eval_with_lowest_and_highest_step,
)
from estimagic.visualization.derivative_plot import derivative_plot
from numpy.testing import assert_array_equal
from pandas.testing import assert_series_equal


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


def f1(x):
    y1 = np.sin(x[0]) + np.cos(x[1]) + x[2]
    return y1


def f2(x):
    y1 = (x[0] - 1) ** 2 + x[1]
    y2 = (x[1] - 1) ** 3
    return np.array([y1, y2])


def f3(x):
    y1 = np.exp(x[0])
    y2 = np.cos(x[0])
    return np.array([y1, y2])


example_functions = [(f1, np.ones(3)), (f2, np.ones(2)), (f3, np.ones(1))]


@pytest.mark.slow()
@pytest.mark.parametrize("func_and_params", example_functions)
@pytest.mark.parametrize("n_steps", range(2, 5))
@pytest.mark.parametrize("grid", [True, False])
def test_derivative_plot(func_and_params, n_steps, grid):
    func, params = func_and_params
    derivative = first_derivative(
        func,
        params,
        n_steps=n_steps,
        return_func_value=True,
        return_info=True,
    )

    derivative_plot(derivative, combine_plots_in_grid=grid)
