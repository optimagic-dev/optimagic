"""Test the functions of the monitoring app."""
import io
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from bokeh.document import Document
from bokeh.models import ColumnDataSource
from pandas.testing import assert_frame_equal

import estimagic.dashboard.monitoring_app as monitoring


@pytest.fixture
def start_params():
    raw_csv_str = """
    ind1,ind2,group,name,value
    1,n,a,hi,1
    1,r,a,world,2
    1,t,b,hi,3
    2,d,b,again,4
    2,s,b,there,5
    """
    csv_str = textwrap.dedent(raw_csv_str)
    start_params = pd.read_csv(io.StringIO(csv_str))
    return start_params.set_index(["ind1", "ind2"])


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
    )


def test_get_start_params_with_id_from_database_singleindex(monkeypatch, start_params):
    def fake_read_last_rows(database, **kwargs):
        start_params = database.copy()
        return {"params": [start_params]}

    monkeypatch.setattr(
        "estimagic.dashboard.monitoring_app.read_last_rows", fake_read_last_rows
    )
    # we can pass the start_params directly to the function because
    # of the monkeypatch of read_last_rows
    single_index_params = start_params.reset_index(level=0)
    res = monitoring._get_start_params_with_id_from_database(single_index_params)
    expected = single_index_params.copy()
    expected["dashboard_id"] = ["n", "r", "t", "d", "s"]
    assert_frame_equal(expected, res)


def test_get_start_params_with_id_from_database_multiindex(monkeypatch, start_params):
    def fake_read_last_rows(database, **kwargs):
        start_params = database.copy()
        return {"params": [start_params]}

    monkeypatch.setattr(
        "estimagic.dashboard.monitoring_app.read_last_rows", fake_read_last_rows
    )
    # we can pass the start_params directly to the function because
    # of the monkeypatch of read_last_rows
    res = monitoring._get_start_params_with_id_from_database(start_params)
    expected = start_params.copy()
    expected["dashboard_id"] = ["1_n", "1_r", "1_t", "2_d", "2_s"]
    assert_frame_equal(expected, res)


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
