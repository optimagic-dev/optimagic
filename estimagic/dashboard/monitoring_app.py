"""Show the development of one optimization's criterion and parameters over time."""
from functools import partial

import numpy as np
from bokeh.layouts import Column
from bokeh.layouts import Row
from bokeh.models import ColumnDataSource
from bokeh.models import HoverTool
from bokeh.models import Panel
from bokeh.models import Tabs

from estimagic.dashboard.utilities import create_wide_figure
from estimagic.dashboard.utilities import get_color_palette
from estimagic.logging.create_database import load_database
from estimagic.logging.read_database import read_last_iterations
from estimagic.logging.read_database import read_scalar_field
from estimagic.optimization.utilities import index_element_to_string


def monitoring_app(doc, short_name, full_path):
    """Create plots showing the development of the criterion and parameters until now.

    Args:
        doc (bokeh.Document): argument required by bokeh
        short_name (str): short and unique name of the database
        full_path (str or pathlib.Path): path to the database.
    """
    database = load_database(full_path)
    start_params = read_scalar_field(database, "start_params")
    db_options = read_scalar_field(database, "db_options")

    data_dict = read_last_iterations(
        database=database,
        tables=["criterion_history", "params_history"],
        n=-1,
        return_type="bokeh",
    )

    criterion_history = ColumnDataSource(
        data=data_dict["criterion_history"], name=f"{short_name}_criterion_history_cds"
    )
    params_history = ColumnDataSource(
        data=data_dict["params_history"], name=f"{short_name}_params_history_cds"
    )

    tab1 = _setup_convergence_tab(
        criterion_history=criterion_history,
        params_history=params_history,
        start_params=start_params,
    )
    tabs = Tabs(tabs=[tab1])
    doc.add_root(tabs)
    plot_new_data = partial(
        _plot_new_data, doc=doc, short_name=short_name, full_path=full_path
    )
    doc.add_periodic_callback(plot_new_data, period_milliseconds=500)


def _setup_convergence_tab(criterion_history, params_history, start_params):
    """Create the figures and plot available time series of the criterion and parameters.

    Args:
        criterion_history (bokeh.ColumnDataSource):
            history of the criterion's values, loaded from the optimization's database.
        params_history (bokeh.ColumnDataSource):
            history of the parameters' values, loaded from the optimization's database.
        start_params (pd.DataFrame):
            DataFrame with the initial parameter values and additional columns,
            in particular the "group" column.

    Returns:
        tab (bokeh.Panel)
    """
    criterion_plot = _plot_time_series(
        data=criterion_history,
        x_name="iteration",
        y_keys=["value"],
        y_names=["criterion"],
        title="Criterion",
    )
    plots = [Row(criterion_plot)]
    group_to_params = _map_groups_to_params(start_params)
    for g, group_params in group_to_params.items():
        group_plot = _plot_time_series(
            data=params_history, y_keys=group_params, x_name="iteration", title=g,
        )
        plots.append(Row(group_plot))

    tab = Panel(child=Column(*plots), title="Convergence Tab")
    return tab


def _plot_time_series(data, y_keys, x_name, title, y_names=None):
    """Plot time series linking the *y_keys* to a common *x_name* variable.

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

    Returns:
        plot (bokeh Figure)

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
    """Map the group name to the ColumnDataSource friendly parameter names.

    Args:
        params (pd.DataFrame):
            DataFrame with the parameter values and additional information such as the
            "group" column and Index.
    """
    group_to_params = {}
    for group in params["group"].unique():
        if group is not None:
            tup_params = params[params["group"] == group].index
            str_params = [index_element_to_string(tup) for tup in tup_params]
            group_to_params[group] = str_params
    return group_to_params


def _plot_new_data(doc, short_name, full_path):
    """Callback to look up new entries in the database and plot them.

    Args:
        doc (bokeh.Document): argument required by bokeh
        short_name (str): short and unique name of the database
        full_path (str or pathlib.Path): path to the database.

    """

    database = load_database(full_path)
    all_data = read_last_iterations(
        database=database,
        tables=["criterion_history", "params_history"],
        n=-1,
        return_type="bokeh",
    )

    old_data = {
        "criterion_history": doc.get_model_by_name(
            f"{short_name}_criterion_history_cds"
        ),
        "params_history": doc.get_model_by_name(f"{short_name}_params_history_cds"),
    }

    for name, old_cds in old_data.items():
        new_data = all_data[name]
        old_iterations = old_cds.data["iteration"]
        new_iterations = new_data["iteration"]
        new_indices = [
            i for i, x in enumerate(new_iterations) if x not in old_iterations
        ]

        to_add = {k: np.array(v)[new_indices] for k, v in new_data.items()}
        old_cds.stream(to_add, rollover=None)
