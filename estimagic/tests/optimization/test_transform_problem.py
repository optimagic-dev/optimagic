import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal as aae
from pandas.testing import assert_frame_equal as afe

import estimagic.optimization.transform_problem as tp
from estimagic.decorators import expand_criterion_output


@pytest.fixture()
def minimal_params():
    user_input = pd.DataFrame(data=[[1], [2], [6]], columns=["value"])
    return user_input


@pytest.fixture()
def incomplete_params():
    user_input = pd.DataFrame()
    user_input["value"] = [1, 2.5, 9]
    user_input["upper"] = [3, None, None]
    user_input["lower"] = [None, 2, 8]
    user_input["group"] = ["coeff", None, None]
    user_input["name"] = ["educ", "cutoff1", None]
    return user_input


def test_process_algorithm_correct_input():
    res = tp._process_algorithm("nlopt_neldermead")
    expected = ("nlopt", "neldermead")
    assert res == expected


def test_process_algorithm_wrong_algo_name():
    with pytest.raises(NotImplementedError):
        tp._process_algorithm("nlopt_neldremead")


def test_process_algorithm_wrong_origin():
    with pytest.raises(NotImplementedError):
        tp._process_algorithm("nlpot_neldermead")


def test_set_params_defaults_if_missing_minimal_params(minimal_params):
    user_input = minimal_params
    expected = user_input.copy()
    expected["lower"] = -np.inf
    expected["upper"] = np.inf
    expected["group"] = "All Parameters"
    expected["name"] = [str(x) for x in expected.index]
    res = tp._set_params_defaults_if_missing(user_input)
    afe(res, expected)


def test_set_params_defaults_if_missing_partial_params(incomplete_params):
    res = tp._set_params_defaults_if_missing(incomplete_params)

    expected = pd.DataFrame()
    expected["value"] = [1, 2.5, 9]
    expected["upper"] = [3, np.inf, np.inf]
    expected["lower"] = [-np.inf, 2, 8]
    expected["group"] = ["coeff", None, None]
    expected["name"] = ["educ", "cutoff1", None]

    afe(res, expected)


def test_check_params_compliant(minimal_params, incomplete_params):
    # these should just run through without error
    tp._check_params(minimal_params)
    tp._check_params(incomplete_params)


def test_check_params_duplicate_index(incomplete_params):
    user_input = incomplete_params
    user_input.loc[-1, "name"] = "cutoff1"
    user_input.set_index(["group", "name"], inplace=True)
    with pytest.raises(AssertionError):
        tp._check_params(user_input)


def test_check_params_internal_col(minimal_params):
    df_with_internal_col = minimal_params.copy()
    df_with_internal_col["_fixed"] = False
    with pytest.raises(ValueError):
        tp._check_params(df_with_internal_col)


def test_evaluate_criterion_scalar(minimal_params):
    def crit_func(params, useless_arg):
        return params["value"].mean()

    expanded_crit = expand_criterion_output(crit_func)
    crit_kwargs = {"useless_arg": "hello world"}

    expected_fitness_eval = 3
    expected_comparison_plot_data = pd.DataFrame()
    expected_comparison_plot_data["value"] = [np.nan]
    res_fitness, res_cp_data = tp._evaluate_criterion(
        expanded_crit, minimal_params, crit_kwargs
    )
    assert res_fitness == expected_fitness_eval
    afe(res_cp_data, expected_comparison_plot_data)


def test_evaluate_criterion_returns_nan(minimal_params):
    def introduce_nan(params, useless_arg):
        params.loc[0, "value"] = np.nan
        return params["value"].to_numpy()

    expanded_crit = expand_criterion_output(introduce_nan)
    crit_kwargs = {"useless_arg": "hello world"}

    with pytest.raises(ValueError):
        tp._evaluate_criterion(expanded_crit, minimal_params, crit_kwargs)


def test_evaluate_criterion_array(minimal_params):
    def return_array(params, useless_arg):
        return params["value"].to_numpy()

    expanded_crit = expand_criterion_output(return_array)
    crit_kwargs = {"useless_arg": "hello world"}

    expected_fitness_eval = 13.66666666666666666666666
    expected_comparison_plot_data = minimal_params
    res_fitness, res_cp_data = tp._evaluate_criterion(
        expanded_crit, minimal_params, crit_kwargs
    )
    assert res_fitness == expected_fitness_eval
    afe(res_cp_data, expected_comparison_plot_data)


# not testing _create_internal_criterion at the moment

# not testing _create_internal_gradient at the moment


def test_get_internal_bounds():
    params = pd.DataFrame()
    params["_internal_free"] = [True, False, False, True]
    params["_internal_lower"] = [-np.inf, 2, 3, 5]
    params["_internal_upper"] = [-10, 3, 5, np.inf]
    expected = (np.array([-np.inf, 5]), np.array([-10, np.inf]))
    res = tp._get_internal_bounds(params)
    assert len(res) == len(expected)
    for arr_res, arr_expected in zip(res, expected):
        aae(arr_res, arr_expected)
