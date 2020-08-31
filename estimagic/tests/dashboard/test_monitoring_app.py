"""Test the functions of the monitoring app."""
from pathlib import Path

from bokeh.document import Document

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
