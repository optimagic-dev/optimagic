"""Test helper functions for the dashboard."""
import webbrowser

from bokeh.io import output_file
from bokeh.io import save
from bokeh.models import ColumnDataSource

import estimagic.dashboard.plot_functions as plot_functions


def test_create_styled_figure():
    plot_functions.create_styled_figure("Hello World")


# not testing find_free_port


def test_plot_time_series_with_large_initial_values():
    cds = ColumnDataSource({"y": [2e17, 1e16, 1e5], "x": [1, 2, 3]})
    title = "Are large initial values shown?"
    fig = plot_functions.plot_time_series(
        data=cds, y_keys=["y"], x_name="x", title=title, plot_width=1000
    )
    title = "Test _plot_time_series can handle large initial values."
    output_file("time_series_initial_value.html", title=title)
    path = save(obj=fig)
    webbrowser.open_new_tab("file://" + path)
