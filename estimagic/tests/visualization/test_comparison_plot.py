import json
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from estimagic.visualization import comparison_plot

FIX_PATH = Path(__file__).resolve().parent / "comparison_plot_fixtures"

# ===========================================================================
# FIXTURES
# ===========================================================================


@pytest.fixture
def minimal_res_dict():
    with open(FIX_PATH / "minimal_res_dict.json", "r") as f:
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
    with open(FIX_PATH / "res_dict_with_cis.json", "r") as f:
        res_dict = json.load(f)
    for model in res_dict.keys():
        data_as_dict = res_dict[model]["result_df"]
        df = pd.DataFrame.from_dict(data_as_dict)
        res_dict[model]["result_df"] = df
    return res_dict


@pytest.fixture
def df():
    df = pd.read_csv(FIX_PATH / "df_minimal.csv")
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
    df = raw.set_index(["model", "name"])
    df.sort_index(level=["model", "name"], inplace=True)
    df["index"] = df["index"].astype(int)
    return df


# _reset_index_without_losing_information
# ===========================================================================

df1 = pd.DataFrame()
df1["a"] = ["a", "b", "c", "d"]
df1["b"] = np.arange(4) + 5
df1["c"] = [100, 104, 108, 150]
df1.index = np.arange(4) + 1

reset_index_fixtures = [
    (df1, "just_reset", df1.reset_index()),
    (df1.set_index(["a", "b"]), "just_reset_multiindex", df1.reset_index(drop=True)),
    (
        df1.set_index(["a", "b"], drop=False),
        "compatible columns",
        df1.reset_index(drop=True),
    ),
]


@pytest.mark.parametrize("df,model,expected", reset_index_fixtures)
def test_reset_index_without_losing_information(df, model, expected):
    res = comparison_plot._reset_index_without_losing_information(df, model)
    pdt.assert_frame_equal(res, expected, check_dtype=False)


def test_reset_index_without_losing_information_raise():
    df2 = pd.DataFrame()
    df2["a"] = ["a", "b", "c", "d"]
    df2["b"] = np.arange(4) + 5
    df2["c"] = [100, 104, 108, 150]
    df2.index = pd.Index(np.arange(4), name="b")

    with pytest.raises(ValueError):
        comparison_plot._reset_index_without_losing_information(
            df2, "incompatible index"
        )


# _prep_result_df
# ===========================================================================


def test_prep_result_df(minimal_res_dict):
    model = "mod1"
    model_dict = minimal_res_dict[model]
    res = comparison_plot._prep_result_df(model_dict, model)
    res = _make_df_similar(res)
    expected = _make_df_similar(pd.read_csv(FIX_PATH / "single_df_prepped.csv"))
    pdt.assert_frame_equal(res, expected, check_like=True)


# _df_with_all_results
# ===========================================================================


def test_df_with_minimal_results(minimal_res_dict, df):
    res = _make_df_similar(comparison_plot._df_with_all_results(minimal_res_dict))
    expected = _make_df_similar(df)
    pdt.assert_frame_equal(res, expected)


def test_df_with_results_with_model_classes(res_dict_with_model_class):
    res = _make_df_similar(
        comparison_plot._df_with_all_results(res_dict_with_model_class)
    )
    expected = pd.read_csv(FIX_PATH / "df_with_model_classes.csv")
    expected = _make_df_similar(expected)
    pdt.assert_frame_equal(res, expected)


# _create_bounds_and_rect_widths
# ===========================================================================


def test_create_bounds_and_rect_widths(df):
    lower, upper, rect_widths = comparison_plot._create_bounds_and_rect_widths(df)
    exp_lower = pd.Series([1.35, 0.52, 1.58, -1.25], index=["a", "b", "c", "d"])
    exp_upper = pd.Series([2.01, 2.73, 2.49, -0.95], index=["a", "b", "c", "d"])
    assert lower.to_dict() == exp_lower.to_dict()
    assert upper.to_dict() == exp_upper.to_dict()


def test_create_bounds_and_rect_widths_with_cis(res_dict_with_cis):
    df = comparison_plot._df_with_all_results(res_dict_with_cis)
    # have one entry with negative lower and positive upper
    df.loc[16, "value"] = 0.02
    lower, upper, rect_widths = comparison_plot._create_bounds_and_rect_widths(df)
    exp_upper = {"a": 2.11, "b": 2.93, "c": 2.5900000000000003, "d": 0.02}
    exp_lower = {"a": 1.35, "b": 0.42000000000000015, "c": 1.43, "d": -1.4}
    assert lower.to_dict() == exp_lower
    assert upper.to_dict() == exp_upper


# _determine_plot_height
# ===========================================================================


def test_determine_plot_height_height_given(df):
    res = comparison_plot._determine_plot_height(df=df, figure_height=800)
    expected = 60
    assert res == expected


def test_determine_plot_height_no_height(df):
    res = comparison_plot._determine_plot_height(df=df, figure_height=None)
    expected = 200
    assert res == expected


# _df_with_plot_specs
# ===========================================================================


def test_df_with_plot_specs():
    df = pd.DataFrame()
    df["value"] = [0.5, 0.2, 5.5, 4.5, -0.2, -0.1, 4.3, 6.1]
    df["group"] = list("aabb") * 2
    df["model_class"] = ["c"] * 4 + ["d"] * 4
    df["name"] = ["a_1", "a_2", "b_1", "b_2"] * 2
    df["conf_int_lower"] = np.nan
    df["conf_int_upper"] = np.nan
    lower, upper, rect_widths = comparison_plot._create_bounds_and_rect_widths(df)
    color_dict = {}

    expected = df.copy()
    expected["color"] = comparison_plot.MEDIUMELECTRICBLUE
    expected["dodge"] = 0.5
    expected["lower_edge"] = [0.500, 0.192, 5.488, 4.480, -0.200, -0.102, 4.3, 6.100]
    expected["upper_edge"] = [0.514, 0.206, 5.524, 4.516, -0.186, -0.088, 4.336, 6.136]
    expected["binned_x"] = expected[["upper_edge", "lower_edge"]].mean(axis=1)

    res = comparison_plot._df_with_plot_specs(df, rect_widths, lower, upper, color_dict)

    pdt.assert_frame_equal(res, expected, check_less_precise=True)


# _df_with_color_column
# ===========================================================================


def test_df_with_color_column_no_dict():
    color_dict = {}
    df = pd.DataFrame()
    df["model_class"] = list("ababab")
    expected = df.copy(deep=True)
    expected["color"] = comparison_plot.MEDIUMELECTRICBLUE
    res = comparison_plot._df_with_color_column(df=df, color_dict=color_dict)
    pdt.assert_frame_equal(res, expected)


def test_df_with_color_column_with_dict():
    color_dict = {"a": "green"}
    df = pd.DataFrame()
    df["model_class"] = list("ababab")
    expected = df.copy(deep=True)
    expected["color"] = ["green", comparison_plot.MEDIUMELECTRICBLUE] * 3
    res = comparison_plot._df_with_color_column(df=df, color_dict=color_dict)
    pdt.assert_frame_equal(res, expected)


# _add_dodge_and_binned_x
# ===========================================================================


def test_add_dodge_and_binned_x_without_class():
    df = pd.DataFrame()
    df["value"] = [1, 5, 3, 3.5, 0.1, 2.5, 0.2, 0.3, 10]
    df["name"] = "param"
    df["model_class"] = "no class"
    df["group"] = "all"

    expected = df.copy(deep=True)
    expected["lower_edge"] = [1.0, 5.0, 3.0, 3.0, 0.0, 2.0, 0.0, 0.0, 6.0]
    expected["upper_edge"] = [2.0, 6.0, 4.0, 4.0, 1.0, 3.0, 1.0, 1.0, np.nan]
    expected["dodge"] = [np.nan, np.nan, 0.5, 1.5, 0.5, np.nan, 1.5, 2.5, np.nan]
    expected["binned_x"] = [1.5, 5.5, 3.5, 3.5, 0.5, 2.5, 0.5, 0.5, np.nan]

    param = "param"
    bins = np.array([-2, -1, 0, 1, 2, 3, 4, 5, 6], dtype=float)
    comparison_plot._add_dodge_and_binned_x(df, "all", param, bins)
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
    res = comparison_plot._find_next_lower(arr, val)
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
    res = comparison_plot._find_next_upper(arr, val)
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
    flattened = comparison_plot._flatten_dict(nested_dict, exclude_key)
    assert flattened == expected


# _df_with_fake_points
# ===========================================================================


def test_df_with_fake_points(df):
    df = pd.DataFrame()
    df["value"] = [0.5, 0.2, 5.5, 4.5, -0.2, -0.1, 4.3, 6.1]
    df["group"] = list("aabb") * 2
    df["model_class"] = ["c"] * 4 + ["d"] * 4
    df["name"] = ["a_1", "a_2", "b_1", "b_2"] + ["a_1", "a_3", "b_1", "b_2"]
    df["conf_int_lower"] = np.nan
    df["conf_int_upper"] = np.nan
    df["model"] = ["mod1"] * 4 + ["mod2"] * 4

    res = comparison_plot._df_with_fake_points(df)

    expected = df.copy()
    expected["fake"] = False

    # add a_3 to mod1
    expected.loc[8, "group"] = "a"
    expected.loc[8, "model_class"] = "c"
    expected.loc[8, "name"] = "a_3"
    expected.loc[8, "model"] = "mod1"
    expected.loc[8, "fake"] = True
    expected.loc[8, "dodge"] = -10
    expected.loc[8, "binned_x"] = -0.1
    # add a_2 to mod2
    expected.loc[9, "group"] = "a"
    expected.loc[9, "model_class"] = "d"
    expected.loc[9, "name"] = "a_2"
    expected.loc[9, "model"] = "mod2"
    expected.loc[9, "fake"] = True
    expected.loc[9, "dodge"] = -10
    expected.loc[9, "binned_x"] = 0.2

    pdt.assert_frame_equal(res, expected)
