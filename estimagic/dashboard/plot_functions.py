"""Helper functions for the dashboard."""
from bokeh.models import HoverTool
from bokeh.models import Legend
from bokeh.plotting import figure

from estimagic.visualization.colors import get_colors


def plot_time_series(
    data,
    y_keys,
    x_name,
    title,
    name=None,
    y_names=None,
    logscale=False,
    plot_width=None,
):
    """Plot time series linking the *y_keys* to a common *x_name* variable.

    Args:
        data (ColumnDataSource): data that contain the y_keys and x_name
        y_keys (list): list of the entries in the data that are to be plotted.
        x_name (str): name of the entry in the data that will be on the x axis.
        title (str): title of the plot.
        name (str, optional): name of the plot for later retrieval with bokeh.
        y_names (list, optional): if given these replace the y keys as line names.
        logscale (bool, optional): Whether to have a logarithmic scale or a linear one.

    Returns:
        plot (bokeh Figure)

    """
    if y_names is None:
        y_names = [str(key) for key in y_keys]

    plot = create_styled_figure(
        title=title, name=name, logscale=logscale, plot_width=plot_width
    )
    # this ensures that the y range spans at least 0.1
    plot.y_range.range_padding = 0.05
    plot.y_range.range_padding_units = "absolute"

    colors = get_colors("categorical", len(y_keys))

    legend_items = [(" " * 60, [])]
    for color, y_key, y_name in zip(colors, y_keys, y_names):
        if len(y_name) <= 25:
            label = y_name
        else:
            label = y_name[:22] + "..."
        line_glyph = plot.line(
            source=data,
            x=x_name,
            y=y_key,
            line_width=2,
            color=color,
            muted_color=color,
            muted_alpha=0.2,
        )
        legend_items.append((label, [line_glyph]))

    tooltips = [(x_name, "@" + x_name)]
    tooltips += [("param_name", y_name), ("param_value", "@" + y_key)]
    hover = HoverTool(renderers=[line_glyph], tooltips=tooltips)
    plot.tools.append(hover)

    legend = Legend(items=legend_items, border_line_color=None, label_width=100)
    legend.click_policy = "mute"
    plot.add_layout(legend, "right")

    return plot


def create_styled_figure(
    title, name=None, tooltips=None, logscale=False, plot_width=None
):
    """Return a styled, empty figure of predetermined height and width.

    Args:
        title (str): Title of the figure.
        name (str): Name of the plot for later retrieval by bokeh. If not given the
            title is set as name
        tooltips (list, optional): List of bokeh tooltips to add to the figure.
        logscale (bool, optional): Whether to have a logarithmic scale or a linear one.

    Returns:
        fig (bokeh Figure)

    """
    plot_width = plot_width if plot_width is not None else 750

    name = name if name is not None else title
    y_axis_type = "log" if logscale else "linear"
    fig = figure(
        plot_height=250,
        plot_width=plot_width,
        title=title.title(),
        tooltips=tooltips,
        name=name,
        y_axis_type=y_axis_type,
    )
    fig.title.text_font_size = "15pt"

    # set minimum borders
    fig.min_border_left = 50
    fig.min_border_right = 50
    fig.min_border_top = 20
    fig.min_border_bottom = 50

    # remove toolbar
    fig.toolbar_location = None

    # remove grid
    fig.grid.visible = False
    # remove minor ticks
    fig.axis.minor_tick_line_color = None
    # remove tick lines
    fig.axis.major_tick_out = 0
    fig.axis.major_tick_in = 0
    # remove outline
    fig.outline_line_width = 0

    return fig
