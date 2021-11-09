"""Helper functions for the dashboard."""
from bokeh.models import HoverTool
from bokeh.models import Legend
from bokeh.plotting import figure
from estimagic.config import GRID_VISIBLE
from estimagic.config import LEGEND_LABEL_TEXT_FONT_SIZE
from estimagic.config import LEGEND_SPACING
from estimagic.config import MAJOR_TICK_IN
from estimagic.config import MAJOR_TICK_OUT
from estimagic.config import MIN_BORDER_BOTTOM
from estimagic.config import MIN_BORDER_LEFT
from estimagic.config import MIN_BORDER_RIGHT
from estimagic.config import MIN_BORDER_TOP
from estimagic.config import MINOR_TICK_LINE_COLOR
from estimagic.config import OUTLINE_LINE_WIDTH
from estimagic.config import PLOT_HEIGHT
from estimagic.config import PLOT_WIDTH
from estimagic.config import TOOLBAR_LOCATION
from estimagic.config import Y_RANGE_PADDING
from estimagic.config import Y_RANGE_PADDING_UNITS
from estimagic.visualization.colors import get_colors


def plot_time_series(
    data,
    y_keys,
    x_name,
    title,
    name=None,
    y_names=None,
    plot_width=PLOT_WIDTH,
):
    """Plot time series linking the *y_keys* to a common *x_name* variable.

    Args:
        data (ColumnDataSource): data that contain the y_keys and x_name
        y_keys (list): list of the entries in the data that are to be plotted.
        x_name (str): name of the entry in the data that will be on the x axis.
        title (str): title of the plot.
        name (str, optional): name of the plot for later retrieval with bokeh.
        y_names (list, optional): if given these replace the y keys as line names.

    Returns:
        plot (bokeh Figure)

    """
    if y_names is None:
        y_names = [str(key) for key in y_keys]

    plot = create_styled_figure(title=title, name=name, plot_width=plot_width)
    # this ensures that the y range spans at least 0.1
    plot.y_range.range_padding = Y_RANGE_PADDING
    plot.y_range.range_padding_units = Y_RANGE_PADDING_UNITS

    colors = get_colors("categorical", len(y_keys))

    legend_items = []
    for color, y_key, y_name in zip(colors, y_keys, y_names):
        if len(y_name) <= 35:
            label = y_name
        else:
            label = "..." + y_name[-32:]
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
    legend_items.append((" " * 60, []))

    tooltips = [(x_name, "@" + x_name)]
    tooltips += [("param_name", y_name), ("param_value", "@" + y_key)]
    hover = HoverTool(renderers=[line_glyph], tooltips=tooltips)
    plot.tools.append(hover)

    legend = Legend(
        items=legend_items,
        border_line_color=None,
        label_width=100,
        label_text_font_size=LEGEND_LABEL_TEXT_FONT_SIZE,
        spacing=LEGEND_SPACING,
    )
    legend.click_policy = "mute"
    plot.add_layout(legend, "right")

    return plot


def create_styled_figure(
    title,
    name=None,
    tooltips=None,
    plot_width=PLOT_WIDTH,
):
    """Return a styled, empty figure of predetermined height and width.

    Args:
        title (str): Title of the figure.
        name (str): Name of the plot for later retrieval by bokeh. If not given the
            title is set as name
        tooltips (list, optional): List of bokeh tooltips to add to the figure.

    Returns:
        fig (bokeh Figure)

    """
    assert plot_width is not None

    name = name if name is not None else title
    fig = figure(
        plot_height=PLOT_HEIGHT,
        plot_width=plot_width,
        title=title.title(),
        tooltips=tooltips,
        name=name,
        y_axis_type="linear",
        sizing_mode="scale_width",
    )
    fig.title.text_font_size = "15pt"

    # set minimum borders
    fig.min_border_left = MIN_BORDER_LEFT
    fig.min_border_right = MIN_BORDER_RIGHT
    fig.min_border_top = MIN_BORDER_TOP
    fig.min_border_bottom = MIN_BORDER_BOTTOM

    # remove toolbar
    fig.toolbar_location = TOOLBAR_LOCATION

    # remove grid
    fig.grid.visible = GRID_VISIBLE
    # remove minor ticks
    fig.axis.minor_tick_line_color = MINOR_TICK_LINE_COLOR
    # remove tick lines
    fig.axis.major_tick_out = MAJOR_TICK_OUT
    fig.axis.major_tick_in = MAJOR_TICK_IN
    # remove outline
    fig.outline_line_width = OUTLINE_LINE_WIDTH

    return fig
