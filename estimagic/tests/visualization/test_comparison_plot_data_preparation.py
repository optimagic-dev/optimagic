"""Tests for the comparison_plot_data_preparation functions."""
import pytest

from estimagic.visualization.comparison_plot import _flatten_dict
from estimagic.visualization.comparison_plot_data_preparation import (
    _determine_plot_height,
)

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


# add_model_name
# ===============


def test_add_model_name_no_name():
    pass


def test_add_model_name_name():
    pass


def test_add_model_name_already_used_name():
    pass


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
    pass


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
    pass


# determine_plot_height
# ======================


def test_determine_plot_height_none():
    res = _determine_plot_height(figure_height=None, y_max=10, n_params=10, n_groups=4)
    expected = 200
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
