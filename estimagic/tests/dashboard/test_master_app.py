"""Test the functions of the dashboards master_app.py."""
from pathlib import Path

from bokeh.document import Document

import estimagic.dashboard.master_app as master_app


def test_master_app():
    # just testing that this does not raise an Error
    doc = Document()
    current_dir_path = Path(__file__).resolve().parent
    name_to_path = {
        "db1": current_dir_path / "db1.db",
        "db2": current_dir_path / "db2.db",
    }
    master_app.master_app(doc=doc, database_name_to_path=name_to_path)


# not testing _create_section_to_elements():

# not testing name_to_bokeh_row_elements

# not testing _setup_tabs
