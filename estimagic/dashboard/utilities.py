"""Helper functions for the dashboard."""
import socket
from contextlib import closing
from pathlib import Path

from bokeh.models import HoverTool
from bokeh.models import Legend
from bokeh.models.widgets import Div
from bokeh.plotting import figure


def create_short_database_names(path_list):
    """Generate short but unique names from each path for each full database path.

    Args:
        path_list (list): Strings or pathlib.Paths to the optimizations' databases.

    Returns:
        short_name_to_path (dict): Mapping from the new unique names to their full path.

    Example:

    >>> pl = ["bla/blubb/blabb.db", "a/b", "bla/blabb"]
    >>> create_short_database_names(pl)
    {'blubb/blabb': 'bla/blubb/blabb.db', 'b': 'a/b', 'bla/blabb': 'bla/blabb'}

    """
    no_suffixes = [Path(p).resolve().with_suffix("") for p in path_list]
    # The assert statement makes sure that the while loop terminates
    assert len(set(no_suffixes)) == len(
        no_suffixes
    ), "path_list must not contain duplicates."
    short_name_to_path = {}
    for path, path_with_suffix in zip(no_suffixes, path_list):
        parts = tuple(reversed(path.parts))
        needed_parts = 1
        candidate = parts[:needed_parts]
        while _name_clash(candidate, no_suffixes):
            needed_parts += 1
            candidate = parts[:needed_parts]

        short_name = "/".join(reversed(candidate))
        short_name_to_path[short_name] = path_with_suffix
    return short_name_to_path


def _name_clash(candidate, path_list, allowed_occurences=1):
    """Determine if candidate leads to a name clash.

    Args:
        candidate (tuple): Tuple with parts of a path.
        path_list (list): List of pathlib.Paths.
        allowed_occurences (int): How often a name can occur before we call it a clash.

    Returns:
        bool

    """
    duplicate_counter = -allowed_occurences
    for path in path_list:
        parts = tuple(reversed(path.parts))
        if len(parts) >= len(candidate) and parts[: len(candidate)] == candidate:
            duplicate_counter += 1
    return duplicate_counter > 0


def create_dashboard_link(name):
    """Create a link refering to *name*'s monitoring app.

    Args:
        name (str): Uniqe name of the database.

    Returns:
        div (bokeh.models.widgets.Div): Link to the database's monitoring page.
    """
    div_name = f"link_{name}"
    open_in_new_tab = r'target="_blank"'
    text = f"<a href=./{name} {open_in_new_tab}> {name}!</a>"
    div = Div(text=text, name=div_name, width=400)
    return div


def create_styled_figure(
    title, name=None, tooltips=None, logscale=False, plot_width=800
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
    name = name if name is not None else title
    y_axis_type = "log" if logscale else "linear"
    fig = figure(
        plot_height=300,
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


def get_color_palette(nr_colors):
    """Return list of colors depending on the number needed.

    Args:
        nr_colors (int): Number of colors needed.

    Returns:
        list

    """
    our_colors = [
        "#547482",
        "#C87259",
        "#C2D8C2",
        "#F1B05D",
        "#818662",
        "#6C4A4D",
        "#7A8C87",
        "#EE8445",
        "#C8B05C",
        "#3C2030",
        "#C89D64",
        "#2A3B49",
    ]
    n_reps = nr_colors // len(our_colors)
    remainder = nr_colors % len(our_colors)
    return n_reps * our_colors + our_colors[:remainder]


def find_free_port():
    """Find a free port on the localhost.

    Adapted from https://stackoverflow.com/a/45690594

    """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("localhost", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def plot_time_series(
    data,
    y_keys,
    x_name,
    title,
    name=None,
    y_names=None,
    logscale=False,
    plot_width=800,
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
    colors = get_color_palette(nr_colors=len(y_keys))

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


def map_groups_to_params(params):
    """Map the group name to the ColumnDataSource friendly parameter names.

    Args:
        params (pd.DataFrame):
            DataFrame with the parameter values and additional information such as the
            "group" column and Index.

    Returns:
        group_to_params (dict):
            Keys are the values of the "group" column. The values are lists with
            bokeh friendly strings of the index tuples identifying the parameters
            that belong to this group. Parameters where group is None, "" or False
            are ignored.

    """
    group_to_params = {}
    for group in params["group"].unique():
        if group is not None and group == group and group != "" and group is not False:
            group_to_params[group] = list(params[params["group"] == group]["name"])
    return group_to_params


def rearrange_to_list_of_twos(elements):
    """Return list elements in pairs of two."""
    paired = []
    for i, entry in enumerate(elements):
        if i % 2 == 1:
            paired.append([elements[i - 1], entry])
        elif i + 1 == len(elements):
            paired.append([entry, None])
    return paired
