"""Test the functions of the monitoring app."""
from pathlib import Path

import pandas as pd
from bokeh.document import Document

import estimagic.dashboard.monitoring_app as monitoring


def test_monitoring_app():
    # only testing that no Error is raised
    # this implicitely tests _setup_convergence_tab and _plot_time_series
    doc = Document()
    short_name = "test_db"
    current_dir_path = Path(__file__).resolve().parent
    full_path = current_dir_path / "db1.db"
    monitoring.monitoring_app(doc=doc, short_name=short_name, full_path=full_path)


def test_map_groups_to_params_group_none():
    params = pd.DataFrame()
    params["value"] = [0, 1, 2, 3]
    params["group"] = None
    params.index = ["a", "b", "c", "d"]
    expected = {}
    res = monitoring._map_groups_to_params(params)
    assert expected == res


def test_map_groups_to_params_group_not_none():
    params = pd.DataFrame()
    params["value"] = [0, 1, 2, 3]
    params["group"] = [None, "A", "B", "B"]
    params.index = ["a", "b", "c", "d"]
    expected = {"A": ["b"], "B": ["c", "d"]}
    res = monitoring._map_groups_to_params(params)
    assert expected == res


def test_map_groups_to_params_group_int_index():
    params = pd.DataFrame()
    params["value"] = [0, 1, 2, 3]
    params["group"] = [None, "A", "B", "B"]
    expected = {"A": ["1"], "B": ["2", "3"]}
    res = monitoring._map_groups_to_params(params)
    assert expected == res


def test_map_groups_to_params_group_multi_index():
    params = pd.DataFrame()
    params["value"] = [0, 1, 2, 3]
    params["group"] = [None, "A", "B", "B"]
    params["ind1"] = ["beta", "beta", "cutoff", "cutoff"]
    params["ind2"] = ["edu", "exp", 1, 2]
    params.set_index(["ind1", "ind2"], inplace=True)
    expected = {"A": ["beta_exp"], "B": ["cutoff_1", "cutoff_2"]}
    res = monitoring._map_groups_to_params(params)
    assert expected == res


# not testing _dashboard_toggle

# not testing _update_monitoring_tab
