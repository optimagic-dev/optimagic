"""Test helper functions for the dashboard."""
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
from bokeh.io import output_file
from bokeh.io import save
from bokeh.models import ColumnDataSource

import estimagic.dashboard.utilities as utils


def test_create_short_database_names_no_conflicts_in_last_element():
    inputs = ["a/db1.db", "b/db2.db", "c/db3.csv"]
    expected_keys = ["db1", "db2", "db3"]
    expected = {k: v for k, v in zip(expected_keys, inputs)}
    res = utils.create_short_database_names(inputs)
    assert expected == res


def test_create_short_database_names_different_stems_same_name():
    inputs = ["a/db.db", "b/db.db", "c/db.csv"]
    expected_keys = ["a/db", "b/db", "c/db"]
    expected = {k: v for k, v in zip(expected_keys, inputs)}
    res = utils.create_short_database_names(inputs)
    assert expected == res


def test_create_short_database_names_mixed_stems_mixed_names():
    inputs = ["a/db.db", "a/db2.db", "c/db.csv"]
    expected_keys = ["a/db", "db2", "c/db"]
    expected = {k: v for k, v in zip(expected_keys, inputs)}
    res = utils.create_short_database_names(inputs)
    assert expected == res


def test_name_clash_no_clash():
    candidate = ("a", "db")
    path_list = [Path("b/db"), Path("c/db"), Path("a/db2")]
    expected = False
    res = utils._name_clash(candidate, path_list)
    assert expected == res


def test_name_clash_with_clash():
    candidate = ("db",)
    path_list = [Path("a/db"), Path("b/db"), Path("c/db2")]
    expected = True
    res = utils._name_clash(candidate, path_list)
    assert expected == res


# no tests for create_dashboard_link


def test_create_styled_figure():
    utils.create_styled_figure("Hello World")


def test_get_color_palette_1():
    colors = utils.get_color_palette(1)
    assert colors == ["#547482"]


def test_get_color_palette_2():
    colors = utils.get_color_palette(2)
    assert colors == ["#547482", "#C87259"]


def test_get_color_palette_5():
    colors = utils.get_color_palette(5)
    expected = ["#547482", "#C87259", "#C2D8C2", "#F1B05D", "#818662"]
    assert colors == expected


def test_get_color_palette_50():
    # only testing that the call works.
    colors = utils.get_color_palette(50)
    assert len(colors) == 50


# not testing find_free_port


def test_plot_time_series_with_large_initial_values():
    cds = ColumnDataSource({"y": [2e17, 1e16, 1e5], "x": [1, 2, 3]})
    title = "Are large initial values shown?"
    fig = utils.plot_time_series(data=cds, y_keys=["y"], x_name="x", title=title)
    title = "Test _plot_time_series can handle large initial values."
    output_file("time_series_initial_value.html", title=title)
    path = save(obj=fig)
    webbrowser.open_new_tab("file://" + path)


# ====================================================================================
# map_group_to_params
# ====================================================================================


def test_map_groups_to_params_group_none():
    params = pd.DataFrame()
    params["value"] = [0, 1, 2, 3]
    params["group"] = None
    params["name"] = ["a", "b", "c", "d"]
    params.index = ["a", "b", "c", "d"]
    expected = {}
    res = utils.map_groups_to_params(params)
    assert expected == res


def test_map_groups_to_params_group_nan():
    params = pd.DataFrame()
    params["value"] = [0, 1, 2, 3]
    params["group"] = np.nan
    params["name"] = ["a", "b", "c", "d"]
    params.index = ["a", "b", "c", "d"]
    expected = {}
    res = utils.map_groups_to_params(params)
    assert expected == res


def test_map_groups_to_params_group_empty():
    params = pd.DataFrame()
    params["value"] = [0, 1, 2, 3]
    params["group"] = ["", "", "x", "x"]
    params["name"] = ["a", "b", "c", "d"]
    params.index = ["a", "b", "c", "d"]
    expected = {"x": ["c", "d"]}
    res = utils.map_groups_to_params(params)
    assert expected == res


def test_map_groups_to_params_group_false():
    params = pd.DataFrame()
    params["value"] = [0, 1, 2, 3]
    params["group"] = [False, False, "x", "x"]
    params["name"] = ["a", "b", "c", "d"]
    params.index = ["a", "b", "c", "d"]
    expected = {"x": ["c", "d"]}
    res = utils.map_groups_to_params(params)
    assert expected == res


def test_map_groups_to_params_group_not_none():
    params = pd.DataFrame()
    params["value"] = [0, 1, 2, 3]
    params["group"] = [None, "A", "B", "B"]
    params.index = ["a", "b", "c", "d"]
    params["name"] = ["a", "b", "c", "d"]
    expected = {"A": ["b"], "B": ["c", "d"]}
    res = utils.map_groups_to_params(params)
    assert expected == res


def test_map_groups_to_params_group_int_index():
    params = pd.DataFrame()
    params["value"] = [0, 1, 2, 3]
    params.index = ["0", "1", "2", "3"]
    params["name"] = ["0", "1", "2", "3"]
    params["group"] = [None, "A", "B", "B"]
    expected = {"A": ["1"], "B": ["2", "3"]}
    res = utils.map_groups_to_params(params)
    assert expected == res


def test_map_groups_to_params_group_multi_index():
    params = pd.DataFrame()
    params["value"] = [0, 1, 2, 3]
    params["group"] = [None, "A", "B", "B"]
    params["ind1"] = ["beta", "beta", "cutoff", "cutoff"]
    params["ind2"] = ["edu", "exp", 1, 2]
    params.set_index(["ind1", "ind2"], inplace=True)
    params["name"] = ["beta_edu", "beta_exp", "cutoff_1", "cutoff_2"]
    expected = {"A": ["beta_exp"], "B": ["cutoff_1", "cutoff_2"]}
    res = utils.map_groups_to_params(params)
    assert expected == res


def test_rearrange_to_list_of_twos_single_entry():
    elements = [1]
    expected = [[1, None]]
    res = utils.rearrange_to_list_of_twos(elements)
    assert res == expected


def test_rearrange_to_list_of_twos_even():
    elements = [
        1,
        2,
        3,
        4,
    ]
    expected = [[1, 2], [3, 4]]
    res = utils.rearrange_to_list_of_twos(elements)
    assert res == expected


def test_rearrange_to_list_of_twos_uneven():
    elements = [1, 2, 3, 4, 5]
    expected = [[1, 2], [3, 4], [5, None]]
    res = utils.rearrange_to_list_of_twos(elements)
    assert res == expected
