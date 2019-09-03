"""Tests for the comparison_plot_data_preparation functions."""
from collections import namedtuple

import pandas as pd
import pandas.testing as pdt
import pytest

from estimagic.visualization.comparison_plot import _flatten_dict
from estimagic.visualization.comparison_plot_data_preparation import (
    _construct_model_names,
)
from estimagic.visualization.comparison_plot_data_preparation import _create_plot_info
from estimagic.visualization.comparison_plot_data_preparation import (
    _determine_plot_height,
)
from estimagic.visualization.comparison_plot_data_preparation import (
    _replace_by_bin_midpoint,
)

OPT_RES = namedtuple("optimization_result", ["params", "info"])

# consolidate_parameter_attribute
# ================================


def test_consolidate_parameter_attribute_standard_wildcards():
    pass


def test_consolidate_parameter_attribute_custom_wildcards():
    pass


def test_consolidate_parameter_attribute_different_results_indices():
    pass


def test_consolidate_parameter_attribute_uncompatible():
    pass


# construct_model_names
# ======================


def test_construct_model_names_no_names():
    params = pd.DataFrame()
    info1 = {"model_class": "small", "foo": "bar"}
    info2 = {"model_class": "large", "foo": "bar2"}
    no_name_results = [OPT_RES(params, info1), OPT_RES(params, info2)]
    res = _construct_model_names(results=no_name_results)
    expected = ["0", "1"]
    assert res == expected


def test_construct_model_names_unique_names():
    params = pd.DataFrame()
    info1 = {"model_name": "small_1", "foo": "bar"}
    info2 = {"model_name": "small_2", "foo": "bar2"}
    unique_name_results = [OPT_RES(params, info1), OPT_RES(params, info2)]
    res = _construct_model_names(results=unique_name_results)
    expected = ["small_1", "small_2"]
    assert res == expected


def test_construct_model_names_duplicate_names():
    params = pd.DataFrame()
    info1 = {"model_name": "small_1", "foo": "bar"}
    info2 = {"model_name": "small_1", "foo": "bar2"}
    results_with_duplicate_names = [OPT_RES(params, info1), OPT_RES(params, info2)]

    with pytest.raises(AssertionError):
        _construct_model_names(results_with_duplicate_names)


def test_construct_model_names_only_some_names():
    params = pd.DataFrame()
    info1 = {"model_name": "small_1", "foo": "bar"}
    info2 = {"foo": "bar2"}
    results_with_only_some_names = [OPT_RES(params, info1), OPT_RES(params, info2)]

    with pytest.raises(AssertionError):
        _construct_model_names(results_with_only_some_names)


# add_model_class_and_color
# ==========================


def test_add_model_class_and_color_no_color_dict():
    pass


def test_add_model_class_and_color_with_color_dict():
    pass


def test_add_model_class_and_color_no_model_class():
    pass


def test_add_model_class_and_color_unknown_model_class():
    pass


def test_add_model_class_and_color_known_model_class():
    pass


# combine_params_data
# ====================


def test_combine_params_data():
    pass


def test_combine_params_data_complicated_index():
    pass


# ensure_correct_conf_ints
# ==========================


def test_ensure_correct_conf_ints_missing():
    pass


def test_ensure_correct_conf_ints_present():
    pass


def test_ensure_correct_conf_ints_raise_error():
    pass


# calculate_x_bounds
# ===================


def test_calculate_x_bounds_without_nan():
    pass


def test_calculate_x_bounds_with_nan():
    pass


# calculate_bins_and_rectangle_width
# ===================================


def test_calculate_bins_and_rectangle_width():
    pass


# replace_by_midpoint
# ====================


def test_replace_by_midpoint_without_nan():
    ind = ["model1", "model2", "model3"]
    values = pd.Series([0.1, 0.2, 0.6], index=ind)
    group_bins = pd.Series([0.0, 0.15, 0.3, 0.45, 0.6, 0.75], name="group1")
    res = _replace_by_bin_midpoint(values, group_bins)
    expected = pd.Series([0.075, 0.225, 0.525], index=ind)
    pdt.assert_series_equal(res, expected)


def test_replace_by_midpoint_with_nan():
    pass


# calculate dodge
# ================


def test_calculate_dodge_without_nan():
    pass


def test_calculate_dodge_with_nan():
    pass


# create_plot_info
# =================


def test_create_plot_info():
    ind = pd.Index(["group1", "group2", "group3"], name="group")
    x_min = pd.Series([0.0, 5.0, -3.5], index=ind, name="x_min")
    x_max = pd.Series([1.0, 149.3, -1.1], index=ind, name="x_max")
    rect_width = pd.Series([0.1, 20, 0.5], index=ind, name="width")
    res = _create_plot_info(
        x_min=x_min, x_max=x_max, rect_width=rect_width, y_max=10, plot_height=50
    )

    expected = {
        "plot_height": 50,
        "y_range": (0, 10),
        "group_info": {
            "group1": {"x_range": (0.0, 1.0), "width": 0.1},
            "group2": {"x_range": (5.0, 149.3), "width": 20},
            "group3": {"x_range": (-3.5, -1.1), "width": 0.5},
        },
    }
    assert res == expected


# determine_plot_height
# ======================


def test_determine_plot_height_none():
    res = _determine_plot_height(figure_height=None, y_max=10, n_params=10, n_groups=4)
    expected = 300
    assert res == expected


def test_determine_plot_height_given():
    res = _determine_plot_height(figure_height=500, y_max=5, n_params=5, n_groups=2)
    expected = 80
    assert res == expected


def test_determine_plot_height_warning():
    with pytest.warns(Warning):
        _determine_plot_height(figure_height=100, y_max=5, n_params=5, n_groups=3)


# comparison_plot_inputs
# =======================


def test_comparison_plot_inputs():
    print("THIS TEST IS STILL MISSING")
    pass


# flatten_dict
# ==============


@pytest.fixture
def nested_dict():
    nested_dict = {
        "g1": {"p1": "val1"},
        "g2": {"p2": "val2"},
        "g3": {"p3": "val3", "p31": "val4"},
    }
    return nested_dict


flatten_dict_fixtures = [
    (None, ["val1", "val2", "val3", "val4"]),
    ("p31", ["val1", "val2", "val3"]),
]


@pytest.mark.parametrize("exclude_key, expected", flatten_dict_fixtures)
def test_flatten_dict_without_exclude_key(nested_dict, exclude_key, expected):
    flattened = _flatten_dict(nested_dict, exclude_key)
    assert flattened == expected
