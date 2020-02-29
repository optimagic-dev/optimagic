"""Test the functions of the monitoring app."""
import webbrowser
from pathlib import Path

import pandas as pd
from bokeh.document import Document
from bokeh.io import output_file
from bokeh.io import save
from bokeh.models import ColumnDataSource

import estimagic.dashboard.monitoring_app as monitoring


def test_plot_time_series_with_large_initial_values():
    cds = ColumnDataSource({"y": [2e17, 1e16, 1e5], "x": [1, 2, 3]})
    title = "Are large initial values shown?"
    fig = monitoring._plot_time_series(data=cds, y_keys=["y"], x_name="x", title=title)
    title = "Test _plot_time_series can handle large initial values."
    output_file("time_series_initial_value.html", title=title)
    path = save(obj=fig)
    webbrowser.open_new_tab("file://" + path)


def test_monitoring_app():
    """Integration test that no Error is raised when calling the monitoring app."""
    doc = Document()
    database_name = "test_db"
    current_dir_path = Path(__file__).resolve().parent
    session_data = {"last_retrieved": 0, "database_path": current_dir_path / "db1.db"}

    monitoring.monitoring_app(
        doc=doc, database_name=database_name, session_data=session_data
    )


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
