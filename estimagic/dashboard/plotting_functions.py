"""Library with low level Bokeh functions for building the dashboard."""
import random

import bokeh.palettes
from bokeh.core.properties import value
from bokeh.plotting import figure


def create_wide_figure(title, tooltips=None):
    """Return an empty figure of predetermined height and width."""
    fig = figure(plot_height=350, plot_width=700, title=title, tooltips=tooltips)
    fig.title.text_font_size = "15pt"
    fig.min_border_left = 50
    fig.min_border_right = 50
    fig.min_border_top = 20
    fig.min_border_bottom = 100
    return fig


def get_color_palette(nr_colors):
    """Return list of colors depending on the number needed."""
    # color tone palettes: bokeh.palettes.Blues9, Greens9, Reds9, Purples9.
    if nr_colors == 1:
        return ["firebrick"]
    elif nr_colors == 2:
        return ["darkslateblue", "goldenrod"]
    elif nr_colors < 20:
        return bokeh.palettes.Category20[nr_colors]
    else:
        return random.choices(bokeh.palettes.Category20[20], k=nr_colors)


def plot_with_lines(data, y_keys, x_name, title, y_names=None):
    """
    Plot lines linking the *y_keys* to a common *x_name* variable.

    Args:
        data (ColumnDataSource):
            data that contain the y_keys and x_name
        y_keys (list):
            list of the entries in the data that are to be plotted
        x_name (str):
            name of the entry in the data that will be on the x axis
        title (str):
            title of the plot
        y_names (list):
            if given these replace the y keys for the names of the lines

    """
    if y_names is None:
        y_names = y_keys

    if x_name == "XxXxITERATIONxXxX":
        tooltips = [("iteration", "@" + x_name)]
    else:
        tooltips = [(x_name, "@" + x_name)]
    if "fitness" not in y_names:
        tooltips += [("fitness", "@fitness")]
    tooltips += [(name, "@" + key) for name, key in zip(y_names, y_keys)]

    plot = create_wide_figure(title=title, tooltips=tooltips)

    colors = get_color_palette(nr_colors=len(y_keys))
    for color, y_key, y_name in zip(colors, y_keys, y_names):
        plot.line(
            source=data,
            x=x_name,
            y=y_key,
            line_width=1,
            legend=value(y_name),
            color=color,
            muted_color=color,
            muted_alpha=0.2,
        )

    plot.legend.click_policy = "mute"

    plot.xaxis.axis_label = x_name if x_name != "XxXxITERATIONxXxX" else "iteration"
    plot.xaxis.axis_label_text_font_style = "normal"
    plot.xaxis.axis_label_text_font_size = "12pt"

    return plot
