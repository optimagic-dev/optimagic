"""Test the functions of the monitoring app."""
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
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
        update_frequency=0.1,
        update_chunk=30,
        start_immediately=False,
    )


def test_create_cds_for_monitoring_app():
    start_params = pd.DataFrame()
    start_params["group"] = ["g1", "g1", None, "g2", "g2", None, "g3"]
    start_params["id"] = ["hello", "world", "test", "p1", "p2", "p3", "1"]
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
    group_to_param_ids = {"g1": ["hello"], "g2": ["p1", "p2"]}
    expected_param_data = {
        k: v for k, v in d.items() if k in ["hello", "p1", "p2", "iteration"]
    }
    expected_param_cds = ColumnDataSource(
        data=expected_param_data, name="params_history_cds"
    )
    _, params_history = monitoring._create_cds_for_monitoring_app(
        start_params, group_to_param_ids
    )
    assert expected_param_cds.data == params_history.data


def test_calculate_strat_point(monkeypatch):
    def fake_read_last_rows(**kwargs):
        return [{"rowid": 20}]

    monkeypatch.setattr(
        "estimagic.dashboard.monitoring_app.read_last_rows", fake_read_last_rows
    )

    res = monitoring._calculate_start_point(database=False, rollover=10, jump=True)

    assert res == 10


def test_calculate_start_point_no_negative_value(monkeypatch):
    def fake_read_last_rows(**kwargs):
        return [{"rowid": 20}]

    monkeypatch.setattr(
        "estimagic.dashboard.monitoring_app.read_last_rows", fake_read_last_rows
    )

    res = monitoring._calculate_start_point(database=False, rollover=30, jump=True)

    assert res == 0


def test_create_id_column_single_index():
    start_params = pd.DataFrame()
    start_params["value"] = [1, 2, 3, 4]
    start_params["group"] = ["a", "a", "b", "b"]
    start_params["name"] = ["this", "repeats"] * 2
    start_params.index = [2, 3, 4, 5]

    res = monitoring._create_id_column(start_params)
    expected = pd.Series(list("2345"), index=start_params.index)
    pdt.assert_series_equal(res, expected)


def test_create_id_column_multi_index():
    multi_params = pd.DataFrame()
    multi_params["value"] = [1, 2, 3, 4]
    multi_params["group"] = ["a", "a", "b", "b"]
    multi_params["name"] = ["this", "repeats"] * 2
    multi_params["3rd level"] = [3, 4, 5, 6]
    multi_params.set_index(["group", "name", "3rd level"], inplace=True)

    res = monitoring._create_id_column(multi_params)
    expected = pd.Series(
        ["a_this_3", "a_repeats_4", "b_this_5", "b_repeats_6"], index=multi_params.index
    )
    pdt.assert_series_equal(res, expected)


# ====================================================================================
# map_group_to_params
# ====================================================================================


def test_map_groups_to_param_ids_group_none():
    params = pd.DataFrame()
    params["value"] = [0, 1, 2, 3]
    params["group"] = None
    params["id"] = ["a", "b", "c", "d"]
    params.index = ["a", "b", "c", "d"]
    expected = {}
    res = monitoring._map_groups_to_param_ids(params)
    assert expected == res


def test_map_groups_to_param_ids_group_nan():
    params = pd.DataFrame()
    params["value"] = [0, 1, 2, 3]
    params["group"] = np.nan
    params["id"] = ["a", "b", "c", "d"]
    params.index = ["a", "b", "c", "d"]
    expected = {}
    res = monitoring._map_groups_to_param_ids(params)
    assert expected == res


def test_map_groups_to_param_ids_group_empty():
    params = pd.DataFrame()
    params["value"] = [0, 1, 2, 3]
    params["group"] = ["", "", "x", "x"]
    params["id"] = ["a", "b", "c", "d"]
    params.index = ["a", "b", "c", "d"]
    expected = {"x": ["c", "d"]}
    res = monitoring._map_groups_to_param_ids(params)
    assert expected == res


def test_map_groups_to_param_ids_group_false():
    params = pd.DataFrame()
    params["value"] = [0, 1, 2, 3]
    params["group"] = [False, False, "x", "x"]
    params["id"] = ["a", "b", "c", "d"]
    params.index = ["a", "b", "c", "d"]
    expected = {"x": ["c", "d"]}
    res = monitoring._map_groups_to_param_ids(params)
    assert expected == res


def test_map_groups_to_param_ids_group_not_none():
    params = pd.DataFrame()
    params["value"] = [0, 1, 2, 3]
    params["group"] = [None, "A", "B", "B"]
    params.index = ["a", "b", "c", "d"]
    params["id"] = ["a", "b", "c", "d"]
    expected = {"A": ["b"], "B": ["c", "d"]}
    res = monitoring._map_groups_to_param_ids(params)
    assert expected == res


def test_map_groups_to_param_ids_group_int_index():
    params = pd.DataFrame()
    params["value"] = [0, 1, 2, 3]
    params.index = ["0", "1", "2", "3"]
    params["id"] = ["0", "1", "2", "3"]
    params["group"] = [None, "A", "B", "B"]
    expected = {"A": ["1"], "B": ["2", "3"]}
    res = monitoring._map_groups_to_param_ids(params)
    assert expected == res


def test_map_groups_to_param_ids_group_multi_index():
    params = pd.DataFrame()
    params["value"] = [0, 1, 2, 3]
    params["group"] = [None, "A", "B", "B"]
    params["ind1"] = ["beta", "beta", "cutoff", "cutoff"]
    params["ind2"] = ["edu", "exp", 1, 2]
    params.set_index(["ind1", "ind2"], inplace=True)
    params["id"] = ["beta_edu", "beta_exp", "cutoff_1", "cutoff_2"]
    expected = {"A": ["beta_exp"], "B": ["cutoff_1", "cutoff_2"]}
    res = monitoring._map_groups_to_param_ids(params)
    assert expected == res
