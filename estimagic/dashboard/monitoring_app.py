"""Show the development of one optimization's criterion and parameters over time."""
from bokeh.layouts import Column
from bokeh.layouts import Row
from bokeh.models import HoverTool
from bokeh.models import Panel
from bokeh.models import Tabs

from estimagic.dashboard_old.plotting_functions import create_wide_figure
from estimagic.dashboard_old.plotting_functions import get_color_palette
from estimagic.logging.create_database import load_database
from estimagic.logging.read_database import read_last_iterations
from estimagic.logging.read_database import read_scalar_field
from estimagic.optimization.utilities import index_element_to_string


def monitoring_app(doc, database_path):
    """Create plots showing the development of the criterion and parameters until now.

    Args:
        doc (bokeh.Document): argument required by bokeh
        database_path (str or pathlib.Path): path to the database
    """
    database = load_database(database_path)
    data_dict = read_last_iterations(
        database=database,
        tables=["criterion_history", "params_history"],
        n=-1,
        return_type="bokeh",
    )

    params = read_scalar_field(database, "start_params")
    db_options = read_scalar_field(database, "db_options")

    tab1 = _setup_convergence_tab(data=data_dict, params=params, db_options=db_options)
    tabs = Tabs(tabs=[tab1])
    doc.add_root(tabs)


def _setup_convergence_tab(data, params, db_options):
    """Create the figures and plot available time series of the criterion and parameters.

    Args:
        data (dict): dictionary containing ColumnDataSources with the time series of
            the criterion values and parameter values.

    Returns:
        tab (bokeh.Panel)
    """

    criterion_plot = _plot_time_series(
        data=data["criterion_history"],
        x_name="iteration",
        y_keys=["value"],
        y_names=["criterion"],
        title="Criterion",
    )
    plots = [Row(criterion_plot)]
    group_to_params = _map_groups_to_params(params)
    for g, group_params in group_to_params.items():
        group_plot = _plot_time_series(
            data=data["params_history"],
            y_keys=group_params,
            x_name="iteration",
            title=g,
        )
        plots.append(Row(group_plot))

    tab = Panel(child=Column(*plots), title="Convergence Tab")
    return tab


def _plot_time_series(data, y_keys, x_name, title, y_names=None):
    """
    Plot time series linking the *y_keys* to a common *x_name* variable.

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

    plot = create_wide_figure(title=title)

    colors = get_color_palette(nr_colors=len(y_keys))
    for color, y_key, y_name in zip(colors, y_keys, y_names):
        line_glyph = plot.line(
            source=data,
            x=x_name,
            y=y_key,
            line_width=2,
            legend_label=y_name,
            color=color,
            muted_color=color,
            muted_alpha=0.2,
        )
    tooltips = [(x_name, "@" + x_name)]
    tooltips += [("param_name", y_name), ("param_value", "@" + y_key)]
    hover = HoverTool(renderers=[line_glyph], tooltips=tooltips)
    plot.tools.append(hover)

    if len(y_key) == 1:
        plot.legend.visible = False
    else:
        plot.legend.click_policy = "mute"
        plot.legend.location = "top_left"

    return plot


def _map_groups_to_params(params):
    """Map the group name to the ColumnDataSource friendly parameter names."""
    group_to_params = {}
    for group in params["group"].unique():
        if group is not None:
            tup_params = params[params["group"] == group].index
            str_params = [index_element_to_string(tup) for tup in tup_params]
            group_to_params[group] = str_params
    return group_to_params
