import json

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from estimagic.visualization.comparison_plot import _create_bounds_and_rect_widths
from estimagic.visualization.comparison_plot import _determine_figure_height
from estimagic.visualization.comparison_plot import _determine_plot_heights
from estimagic.visualization.comparison_plot import _df_with_all_results
from estimagic.visualization.comparison_plot import _flatten_dict
from estimagic.visualization.comparison_plot import _prep_result_df

# ===========================================================================
# FIXTURES
# ===========================================================================


@pytest.fixture()
def minimal_res_dict():
    groups = ["a", "a", "b", "b", "b", "b", "c"]
    names = ["u", "i", "a", "e", "n", "r", "t"]

    df1 = pd.DataFrame()
    df1["group"] = groups
    df1["name"] = names
    df1["final_value"] = [1.58, 2.01, 2.73, 1.62, 2.18, 1.75, 2.25]

    df2 = pd.DataFrame()
    df2["group"] = groups + ["c", "d", "d"]
    df2["name"] = names + ["x", "v", "l"]
    df2["final_value"] = [1.48, 1.82, 1.12, 2.15, 1.65, 1.93, 2.39, 1.68, -1.24, -0.95]

    df3 = df1.copy()
    df3["final_value"] += [-0.23, -0.2, -0.11, 0.03, -0.13, -0.21, 0.17]

    df4 = df2.copy()
    df4["final_value"] += [0.4, -0.2, -0.6, -0.0, 0.2, -0.1, 0.1, -0.1, 0.0, -0.3]

    res_dict = {
        "mod1": {"result_df": df1},
        "mod2": {"result_df": df2},
        "mod3": {"result_df": df3},
        "mod4": {"result_df": df4},
    }
    return res_dict


@pytest.fixture()
def res_dict_with_model_class(minimal_res_dict):
    res_dict = minimal_res_dict
    res_dict["mod2"]["model_class"] = "large"
    res_dict["mod4"]["model_class"] = "large"
    return res_dict


@pytest.fixture()
def res_dict_with_cis(res_dict_with_model_class):
    res_dict = res_dict_with_model_class
    diff1 = [0.3, 0.1, 0.2, 0.1, 0.3, 0.4, 0.2]
    diff2 = [0.1, 0.1, 0.2, 0.3, 0.3, 0.2, 0.4]
    res_dict["mod1"]["result_df"]["conf_int_upper"] = res_dict["mod1"]["result_df"][
        "final_value"
    ] + np.array(diff1)
    res_dict["mod1"]["result_df"]["conf_int_lower"] = res_dict["mod1"]["result_df"][
        "final_value"
    ] - np.array(diff2)

    diff1 += [0.3] * 3
    diff2 += [0.3] * 3
    res_dict["mod4"]["result_df"]["conf_int_upper"] = res_dict["mod4"]["result_df"][
        "final_value"
    ] + 0.5 * np.array(diff1)
    res_dict["mod4"]["result_df"]["conf_int_lower"] = res_dict["mod4"]["result_df"][
        "final_value"
    ] - 0.5 * np.array(diff2)

    return res_dict


@pytest.fixture()
def df():
    with open("estimagic/tests/visualization/minimal_expected_df.json", "r") as f:
        df = pd.DataFrame(json.load(f))
        df["conf_int_upper"] = pd.np.nan
        df["conf_int_lower"] = pd.np.nan
    return df


# _prep_result_df
# ===========================================================================


def test_prep_result_df(minimal_res_dict):
    model = "mod1"
    model_dict = minimal_res_dict[model]
    res = _prep_result_df(model_dict, model)
    expected = pd.DataFrame.from_dict(
        {
            "group": {0: "a", 1: "a", 2: "b", 3: "b", 4: "b", 5: "b", 6: "c"},
            "name": {0: "u", 1: "i", 2: "a", 3: "e", 4: "n", 5: "r", 6: "t"},
            "final_value": {
                0: 1.58,
                1: 2.01,
                2: 2.73,
                3: 1.62,
                4: 2.18,
                5: 1.75,
                6: 2.25,
            },
            "index": {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6},
            "full_name": {
                0: "a_u",
                1: "a_i",
                2: "b_a",
                3: "b_e",
                4: "b_n",
                5: "b_r",
                6: "c_t",
            },
            "model_class": {
                0: "no class",
                1: "no class",
                2: "no class",
                3: "no class",
                4: "no class",
                5: "no class",
                6: "no class",
            },
            "model": {
                0: "mod1",
                1: "mod1",
                2: "mod1",
                3: "mod1",
                4: "mod1",
                5: "mod1",
                6: "mod1",
            },
        }
    )
    assert_frame_equal(res, expected, check_like=True)


# _df_with_all_results
# ===========================================================================


def test_df_with_minimal_results(minimal_res_dict):
    with open("estimagic/tests/visualization/minimal_expected_df.json", "r") as f:
        expected = pd.DataFrame(json.load(f))
        expected["conf_int_upper"] = pd.np.nan
        expected["conf_int_lower"] = pd.np.nan
    res = _df_with_all_results(minimal_res_dict)
    expected.set_index(["model", "full_name"], inplace=True)
    res.set_index(["model", "full_name"], inplace=True)
    assert_frame_equal(res, expected)


def test_df_with_results_with_model_classes(res_dict_with_model_class):
    with open(
        "estimagic/tests/visualization/with_model_class_expected_df.json", "r"
    ) as f:
        expected = pd.DataFrame(json.load(f))
        expected["conf_int_upper"] = pd.np.nan
        expected["conf_int_lower"] = pd.np.nan
    res = _df_with_all_results(res_dict_with_model_class)
    expected.set_index(["model", "full_name"], inplace=True)
    res.set_index(["model", "full_name"], inplace=True)
    assert_frame_equal(res, expected)


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
    df.loc[
        16, "final_value"
    ] = 0.02  # have one entry with negative lower and positive upper
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


# _add_color_column
# ===========================================================================


# _add_dodge_and_binned_x
# ===========================================================================


# _find_next_lower
# ===========================================================================


# _find_next_upper
# ===========================================================================


# _flatten_dict
# ===========================================================================


def test_flatten_dict_without_exclude_key():
    d = {
        "g1": {"p1": "val1"},
        "g2": {"p2": "val2"},
        "g3": {"p3": "val3", "p31": "val4"},
    }

    flattened = _flatten_dict(d)
    expected = ["val1", "val2", "val3", "val4"]
    assert flattened == expected


def test_flatten_dict_with_exclude_key():
    d = {
        "g1": {"p1": "val1"},
        "g2": {"p2": "val2"},
        "g3": {"p3": "val3", "p31": "val4"},
    }

    flattened = _flatten_dict(d, "p31")
    expected = ["val1", "val2", "val3"]
    assert flattened == expected
