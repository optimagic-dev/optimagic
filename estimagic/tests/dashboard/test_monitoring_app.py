"""Test the functions of the monitoring app."""
from pathlib import Path

from bokeh.document import Document
from bokeh.models import ColumnDataSource

import estimagic.dashboard.monitoring_app as monitoring


def test_monitoring_app():
    """Integration test that no Error is raised when calling the monitoring app."""
    doc = Document()
    database_name = "test_db"
    current_dir_path = Path(__file__).resolve().parent
    session_data = {"last_retrieved": 0, "database_path": current_dir_path / "db1.db"}

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
    group_to_params = {"g1": ["hello", "world"], "g2": ["p1", "p2", 1]}
    d = {
        "hello": [],
        "world": [],
        "p1": [],
        "p2": [],
        "1": [],
        "iteration": [],
    }
    expected_param_cds = ColumnDataSource(data=d, name="params_history_cds")
    _, params_history = monitoring._create_cds_for_monitoring_app(group_to_params)
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
