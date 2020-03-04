"""Test helper functions for the dashboard."""
from pathlib import Path

import bokeh.palettes

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

# no tests for create_standard_figure


def test_get_color_palette_1():
    colors = utils.get_color_palette(1)
    assert colors == ["firebrick"]


def test_get_color_palette_2():
    colors = utils.get_color_palette(2)
    assert colors == ["darkslateblue", "goldenrod"]


def test_get_color_palette_5():
    colors = utils.get_color_palette(5)
    assert colors == bokeh.palettes.Category10[5]


def test_get_color_palette_50():
    # only testing that the call works.
    colors = utils.get_color_palette(50)
    assert len(colors) == 50


# not testing find_free_port
