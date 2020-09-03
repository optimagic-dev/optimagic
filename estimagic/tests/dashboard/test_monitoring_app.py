"""Test the functions of the monitoring app."""
from pathlib import Path

import numpy as np
import pandas as pd
from bokeh.document import Document
from bokeh.models import ColumnDataSource

import estimagic.dashboard.monitoring_app as monitoring


def test_monitoring_app():
    """Integration test that no Error is raised when calling the monitoring app."""
    doc = Document()
    database_name = "test_db"
    current_dir_path = Path(__file__).resolve().parent
    session_data = {
        "last_retrieved": 0,
        "database_path": current_dir_path / "db1.db",
    }

    monitoring.monitoring_app(
        doc=doc,
        database_name=database_name,
        session_data=session_data,
        rollover=10_000,
        jump=False,
        frequency=0.1,
        update_chunk=30,
    )


def test_create_cds_for_monitoring_app():
    start_params = pd.DataFrame()
    start_params["group"] = ["g1", "g1", None, "g2", "g2", None, "g3"]
    start_params["name"] = ["hello", "world", "test", "p1", "p2", "p3", "1"]
    d = {
        "hello": [],
        "world": [],
        "test": [],
        "p1": [],
        "p2": [],
        "p3": [],
        "1": [],
        "iteration": [],
    }
    expected_param_cds = ColumnDataSource(data=d, name="params_history_cds")
    _, params_history = monitoring._create_cds_for_monitoring_app(start_params)
    assert expected_param_cds.data == params_history.data


def test_set_last_retrieved(monkeypatch):
    session_data = {}

    def fake_read_last_rows(**kwargs):
        return [{"rowid": 20}]

    monkeypatch.setattr(
        "estimagic.dashboard.monitoring_app.read_last_rows", fake_read_last_rows
    )

    monitoring._set_last_retrieved(
        session_data=session_data, database=False, rollover=10, jump=True
    )

    assert session_data == {"last_retrieved": 10}


def test_set_last_retrieved_no_negative_value(monkeypatch):
    session_data = {}

    def fake_read_last_rows(**kwargs):
        return [{"rowid": 20}]

    monkeypatch.setattr(
        "estimagic.dashboard.monitoring_app.read_last_rows", fake_read_last_rows
    )

    monitoring._set_last_retrieved(
        session_data=session_data, database=False, rollover=30, jump=True
    )

    assert session_data == {"last_retrieved": 0}


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
    res = monitoring._map_groups_to_params(params)
    assert expected == res


def test_map_groups_to_params_group_nan():
    params = pd.DataFrame()
    params["value"] = [0, 1, 2, 3]
    params["group"] = np.nan
    params["name"] = ["a", "b", "c", "d"]
    params.index = ["a", "b", "c", "d"]
    expected = {}
    res = monitoring._map_groups_to_params(params)
    assert expected == res


def test_map_groups_to_params_group_empty():
    params = pd.DataFrame()
    params["value"] = [0, 1, 2, 3]
    params["group"] = ["", "", "x", "x"]
    params["name"] = ["a", "b", "c", "d"]
    params.index = ["a", "b", "c", "d"]
    expected = {"x": ["c", "d"]}
    res = monitoring._map_groups_to_params(params)
    assert expected == res


def test_map_groups_to_params_group_false():
    params = pd.DataFrame()
    params["value"] = [0, 1, 2, 3]
    params["group"] = [False, False, "x", "x"]
    params["name"] = ["a", "b", "c", "d"]
    params.index = ["a", "b", "c", "d"]
    expected = {"x": ["c", "d"]}
    res = monitoring._map_groups_to_params(params)
    assert expected == res


def test_map_groups_to_params_group_not_none():
    params = pd.DataFrame()
    params["value"] = [0, 1, 2, 3]
    params["group"] = [None, "A", "B", "B"]
    params.index = ["a", "b", "c", "d"]
    params["name"] = ["a", "b", "c", "d"]
    expected = {"A": ["b"], "B": ["c", "d"]}
    res = monitoring._map_groups_to_params(params)
    assert expected == res


def test_map_groups_to_params_group_int_index():
    params = pd.DataFrame()
    params["value"] = [0, 1, 2, 3]
    params.index = ["0", "1", "2", "3"]
    params["name"] = ["0", "1", "2", "3"]
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
    params["name"] = ["beta_edu", "beta_exp", "cutoff_1", "cutoff_2"]
    expected = {"A": ["beta_exp"], "B": ["cutoff_1", "cutoff_2"]}
    res = monitoring._map_groups_to_params(params)
    assert expected == res
