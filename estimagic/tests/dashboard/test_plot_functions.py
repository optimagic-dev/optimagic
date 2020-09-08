"""Test helper functions for the dashboard."""
import webbrowser

from bokeh.io import output_file
from bokeh.io import save
from bokeh.models import ColumnDataSource

import estimagic.dashboard.plot_functions as plot_functions


def test_create_styled_figure():
    plot_functions.create_styled_figure("Hello World")


def test_get_color_palette_1():
    colors = plot_functions.get_color_palette(1)
    assert colors == ["#547482"]


def test_get_color_palette_2():
    colors = plot_functions.get_color_palette(2)
    assert colors == ["#547482", "#C87259"]


def test_get_color_palette_5():
    colors = plot_functions.get_color_palette(5)
    expected = ["#547482", "#C87259", "#C2D8C2", "#F1B05D", "#818662"]
    assert colors == expected


def test_get_color_palette_50():
    # only testing that the call works.
    colors = plot_functions.get_color_palette(50)
    assert len(colors) == 50


# not testing find_free_port


def test_plot_time_series_with_large_initial_values():
    cds = ColumnDataSource({"y": [2e17, 1e16, 1e5], "x": [1, 2, 3]})
    title = "Are large initial values shown?"
    fig = plot_functions.plot_time_series(
        data=cds, y_keys=["y"], x_name="x", title=title
    )
    title = "Test _plot_time_series can handle large initial values."
    output_file("time_series_initial_value.html", title=title)
    path = save(obj=fig)
    webbrowser.open_new_tab("file://" + path)
