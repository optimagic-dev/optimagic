import json
from os.path import join

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from estimagic.visualization.comparison_plot import _add_color_column
from estimagic.visualization.comparison_plot import _add_dodge_and_binned_x
from estimagic.visualization.comparison_plot import _add_plot_specs_to_df
from estimagic.visualization.comparison_plot import _create_bounds_and_rect_widths
from estimagic.visualization.comparison_plot import _determine_figure_height
from estimagic.visualization.comparison_plot import _determine_plot_heights
from estimagic.visualization.comparison_plot import _df_with_all_results
from estimagic.visualization.comparison_plot import _find_next_lower
from estimagic.visualization.comparison_plot import _find_next_upper
from estimagic.visualization.comparison_plot import _flatten_dict
from estimagic.visualization.comparison_plot import _prep_result_df
from estimagic.visualization.comparison_plot import MEDIUMELECTRICBLUE

FIX_PATH = "estimagic/tests/visualization/comparison_plot_fixtures/"


# ===========================================================================
# FIXTURES
# ===========================================================================


@pytest.fixture
def minimal_res_dict():
    with open(join(FIX_PATH, "minimal_res_dict.json"), "r") as f:
        res_dict = json.load(f)
    for model in res_dict.keys():
        data_as_dict = res_dict[model]["result_df"]
        df = pd.DataFrame.from_dict(data_as_dict)
        res_dict[model]["result_df"] = df
    return res_dict


@pytest.fixture
def res_dict_with_model_class(minimal_res_dict):
    res_dict = minimal_res_dict
    res_dict["mod2"]["model_class"] = "large"
    res_dict["mod4"]["model_class"] = "large"
    return res_dict


@pytest.fixture
def res_dict_with_cis():
    with open(join(FIX_PATH, "res_dict_with_cis.json"), "r") as f:
        res_dict = json.load(f)
    for model in res_dict.keys():
        data_as_dict = res_dict[model]["result_df"]
        df = pd.DataFrame.from_dict(data_as_dict)
        res_dict[model]["result_df"] = df
    return res_dict


@pytest.fixture
def df():
    df = pd.read_csv(join(FIX_PATH, "df_minimal.csv"))
    return df


@pytest.fixture
def nested_dict():
    nested_dict = {
        "g1": {"p1": "val1"},
        "g2": {"p2": "val2"},
        "g3": {"p3": "val3", "p31": "val4"},
    }
    return nested_dict


def _make_df_similar(raw):
    df = raw.set_index(["model", "full_name"])
    df.sort_index(level=["model", "full_name"], inplace=True)
    df["index"] = df["index"].astype(int)
    return df


# _prep_result_df
# ===========================================================================


def test_prep_result_df(minimal_res_dict):
    model = "mod1"
    model_dict = minimal_res_dict[model]
    res = _make_df_similar(_prep_result_df(model_dict, model))
    expected = _make_df_similar(pd.read_csv(join(FIX_PATH, "single_df_prepped.csv")))
    pdt.assert_frame_equal(res, expected, check_like=True)


# _df_with_all_results
# ===========================================================================


def test_df_with_minimal_results(minimal_res_dict, df):
    res = _make_df_similar(_df_with_all_results(minimal_res_dict))
    expected = _make_df_similar(df)
    pdt.assert_frame_equal(res, expected)


def test_df_with_results_with_model_classes(res_dict_with_model_class):
    res = _make_df_similar(_df_with_all_results(res_dict_with_model_class))
    expected = pd.read_csv(join(FIX_PATH, "df_with_model_classes.csv"))
    expected = _make_df_similar(expected)
    pdt.assert_frame_equal(res, expected)


# _create_bounds_and_rect_widths
# ===========================================================================


def test_create_bounds_and_rect_widths(df):
    lower, upper, rect_widths = _create_bounds_and_rect_widths(df)
    exp_lower = pd.Series([1.35, 0.52, 1.58, -1.25], index=["a", "b", "c", "d"])
    exp_upper = pd.Series([2.01, 2.73, 2.49, -0.95], index=["a", "b", "c", "d"])
    assert lower.to_dict() == exp_lower.to_dict()
    assert upper.to_dict() == exp_upper.to_dict()


def test_create_bounds_and_rect_widths_with_cis(res_dict_with_cis):
    df = _df_with_all_results(res_dict_with_cis)
    # have one entry with negative lower and positive upper
    df.loc[16, "final_value"] = 0.02
    lower, upper, rect_widths = _create_bounds_and_rect_widths(df)
    exp_upper = {"a": 2.11, "b": 2.93, "c": 2.5900000000000003, "d": 0.02}

    exp_lower = {"a": 1.35, "b": 0.42000000000000015, "c": 1.43, "d": -1.4}
    assert lower.to_dict() == exp_lower
    assert upper.to_dict() == exp_upper


# _determine_plot_heights
# ===========================================================================


def test_determine_plot_heights(df):
    res = _determine_plot_heights(df=df, figure_height=400)
    expected = pd.Series([80, 160, 80, 80], index=["a", "b", "c", "d"])
    assert res.to_dict() == expected.to_dict()


# _determine_figure_height
# ===========================================================================


def test_determine_figure_height_none(df):
    expected = 8 * 10 * 10
    assert _determine_figure_height(df, None) == expected


def test_determine_figure_height_given(df):
    expected = 400
    assert _determine_figure_height(df, 400) == expected


# _add_plot_specs_to_df
# ===========================================================================


def test_add_plot_specs_to_df():
    df = pd.DataFrame()
    df["final_value"] = [0.5, 0.2, 5.5, 4.5, -0.2, -0.1, 4.3, 6.1]
    df["group"] = list("aabb") * 2
    df["model_class"] = ["c"] * 4 + ["d"] * 4
    df["full_name"] = ["a_1", "a_2", "b_1", "b_2"] * 2
    df["conf_int_lower"] = np.nan
    df["conf_int_upper"] = np.nan
    lower, upper, rect_widths = _create_bounds_and_rect_widths(df)
    color_dict = {}

    expected = df.copy()
    expected["color"] = MEDIUMELECTRICBLUE
    expected["dodge"] = 0.5
    expected["lower_edge"] = [0.500, 0.192, 5.488, 4.480, -0.200, -0.102, 4.3, 6.100]
    expected["upper_edge"] = [0.514, 0.206, 5.524, 4.516, -0.186, -0.088, 4.336, 6.136]
    expected["binned_x"] = expected[["upper_edge", "lower_edge"]].mean(axis=1)

    _add_plot_specs_to_df(df, rect_widths, lower, upper, color_dict)
    df[["lower_edge", "upper_edge"]] = df[["lower_edge", "upper_edge"]].round(3)

    pdt.assert_frame_equal(df, expected)


# _add_color_column
# ===========================================================================


def test_add_color_column_no_dict():
    color_dict = {}
    df = pd.DataFrame()
    df["model_class"] = list("ababab")
    expected = df.copy(deep=True)
    expected["color"] = MEDIUMELECTRICBLUE
    _add_color_column(df=df, color_dict=color_dict)
    pdt.assert_frame_equal(df, expected)


def test_add_color_column_with_dict():
    color_dict = {"a": "green"}
    df = pd.DataFrame()
    df["model_class"] = list("ababab")
    expected = df.copy(deep=True)
    expected["color"] = ["green", MEDIUMELECTRICBLUE] * 3
    _add_color_column(df=df, color_dict=color_dict)
    pdt.assert_frame_equal(df, expected)


# _add_dodge_and_binned_x
# ===========================================================================


def test_add_dodge_and_binned_x_without_class():
    df = pd.DataFrame()
    df["final_value"] = [1, 5, 3, 3.5, 0.1, 2.5, 0.2, 0.3, 10]
    df["full_name"] = "param"
    df["model_class"] = "no class"

    expected = df.copy(deep=True)
    expected["lower_edge"] = [1.0, 5.0, 3.0, 3.0, 0.0, 2.0, 0.0, 0.0, 6.0]
    expected["upper_edge"] = [2.0, 6.0, 4.0, 4.0, 1.0, 3.0, 1.0, 1.0, np.nan]
    expected["dodge"] = [np.nan, np.nan, 0.5, 1.5, 0.5, np.nan, 1.5, 2.5, np.nan]
    expected["binned_x"] = [1.5, 5.5, 3.5, 3.5, 0.5, 2.5, 0.5, 0.5, np.nan]

    param = "param"
    bins = np.array([-2, -1, 0, 1, 2, 3, 4, 5, 6], dtype=float)
    _add_dodge_and_binned_x(df, param, bins)
    pdt.assert_frame_equal(df, expected)


# _find_next_lower
# ===========================================================================

sorted_arr = np.arange(15)
unsorted_arr = np.array([3, 5, 1, 0, -10, -5, 0])
empty_arr = np.array([])

find_lower_fixtures = [
    (sorted_arr, 5.5, 5),
    (sorted_arr, 5, 5),
    (unsorted_arr, -2.5, -5),
    (unsorted_arr, -20, np.nan),
    (empty_arr, 45, np.nan),
]


@pytest.mark.parametrize("arr, val, expected", find_lower_fixtures)
def test_find_next_lower(arr, val, expected):
    res = _find_next_lower(arr, val)
    if np.isnan(expected):
        assert np.isnan(res)
    else:
        assert res == expected


# _find_next_upper
# ===========================================================================

find_upper_fixtures = [
    (sorted_arr, 5.5, 6),
    (sorted_arr, 5, 6),
    (unsorted_arr, -2.5, 0),
    (unsorted_arr, 350, np.nan),
    (empty_arr, 45, np.nan),
]


@pytest.mark.parametrize("arr, val, expected", find_upper_fixtures)
def test_find_next_upper(arr, val, expected):
    res = _find_next_upper(arr, val)
    if np.isnan(expected):
        assert np.isnan(res)
    else:
        assert res == expected


# _flatten_dict
# ===========================================================================

flatten_dict_fixtures = [
    (None, ["val1", "val2", "val3", "val4"]),
    ("p31", ["val1", "val2", "val3"]),
]


@pytest.mark.parametrize("exclude_key, expected", flatten_dict_fixtures)
def test_flatten_dict_without_exclude_key(nested_dict, exclude_key, expected):
    flattened = _flatten_dict(nested_dict, exclude_key)
    assert flattened == expected
