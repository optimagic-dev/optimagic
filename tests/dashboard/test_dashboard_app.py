"""Test the functions of the dashboard app."""
import estimagic.dashboard.dashboard_app as dashboard_app
import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest
from bokeh.document import Document
from bokeh.models import ColumnDataSource
from estimagic import minimize


def pybaum_sphere(params):
    """Sphere function expecting a dictionary with floats and a np.array.

    Args:
        params (dict): keys are "a", "b", "c". The first two values are floats, the
            third is a np.array.

    Returns:
        float: criterion value.

    """
    return params["a"] ** 2 + params["b"] ** 2 + (params["c"] ** 2).sum()


def pandas_sphere(params):
    """Sphere function expecting a pandas DataFrame.

    Args:
        params (pd.DataFrame): expected to have a "value" column with float entries.

    Returns:
        float: criterion value.

    """
    return (params["value"] ** 2).sum()


@pytest.mark.parametrize(
    "criterion, start_params",
    [
        (pybaum_sphere, {"a": 2, "b": 4, "c": np.arange(4)}),
        (pandas_sphere, pd.DataFrame({"value": np.ones(6)})),
    ],
)
def test_dashboard_app(criterion, start_params, tmpdir):
    """Integration test that no Error is raised when calling the dashboard app."""
    doc = Document()

    # create database
    db_path = tmpdir / "test_db.db"
    minimize(
        criterion=criterion,
        params=start_params,
        logging=db_path,
        algorithm="scipy_lbfgsb",
    )

    session_data = {
        "last_retrieved": 0,
        "database_path": db_path,
        "callbacks": {},
    }
    updating_options = {
        "rollover": 10_000,
        "jump": False,
        "update_frequency": 0.1,
        "update_chunk": 30,
        "stride": 1,
    }

    dashboard_app.dashboard_app(
        doc=doc,
        session_data=session_data,
        updating_options=updating_options,
    )


def test_create_cds_for_dashboard():
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
    _, params_history = dashboard_app._create_cds_for_dashboard(group_to_param_ids)
    assert expected_param_cds.data == params_history.data


def test_calculate_start_point(monkeypatch):
    def fake_read_last_rows(**kwargs):
        return [{"rowid": 20}]

    monkeypatch.setattr(
        "estimagic.dashboard.dashboard_app.read_last_rows", fake_read_last_rows
    )

    updating_options = {
        "rollover": 10,
        "stride": 1,
        "jump": True,
    }
    res = dashboard_app._calculate_start_point(
        database=False,
        updating_options=updating_options,
    )

    assert res == 10


def test_calculate_start_point_no_negative_value(monkeypatch):
    def fake_read_last_rows(**kwargs):
        return [{"rowid": 20}]

    monkeypatch.setattr(
        "estimagic.dashboard.dashboard_app.read_last_rows", fake_read_last_rows
    )

    res = dashboard_app._calculate_start_point(
        database=False,
        updating_options={"rollover": 30, "stride": 1, "jump": True},
    )

    assert res == 0


def test_create_id_column():
    start_params = pd.DataFrame(index=[2, 4, 6, 8, 10, 12])
    start_params["group"] = ["g1", "g2", None, "", False, np.nan]
    res = dashboard_app._create_id_column(start_params)
    expected = pd.Series(["0", "1"] + ["None"] * 4, index=start_params.index)
    pdt.assert_series_equal(res, expected)


# ====================================================================================
# map_group_to_params
# ====================================================================================

ignore_groups = [None, np.nan, False, ""]


@pytest.mark.parametrize("group_val", ignore_groups)
def test_map_groups_to_param_ids_group_none(group_val):
    params = pd.DataFrame()
    params["value"] = [0, 1, 2, 3]
    params["group"] = group_val
    params["id"] = ["a", "b", "c", "d"]
    params.index = ["a", "b", "c", "d"]
    expected = {}
    res = dashboard_app._map_group_to_other_column(params, "id")
    assert expected == res


ind_and_ids = [
    (["a", "b", "c", "d"], ["0", "1", "2", "3"]),
    ([2, 3, 4, 5], ["0", "1", "2", "3"]),
]


@pytest.mark.parametrize("index, ids", ind_and_ids)
def test_map_groups_to_param_ids_group_not_none(index, ids):
    params = pd.DataFrame()
    params["value"] = [0, 1, 2, 3]
    params["group"] = [None, "A", "B", "B"]
    params.index = index
    params["id"] = ids
    expected = {"A": ["1"], "B": ["2", "3"]}
    res = dashboard_app._map_group_to_other_column(params, "id")
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
    res = dashboard_app._map_group_to_other_column(params, "id")
    assert expected == res
