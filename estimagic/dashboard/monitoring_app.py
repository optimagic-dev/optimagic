"""Show the development of one optimization's criterion and parameters over time."""
from functools import partial

from bokeh.layouts import Column
from bokeh.layouts import Row
from bokeh.models import ColumnDataSource
from bokeh.models import HoverTool
from bokeh.models import Legend
from bokeh.models import Panel
from bokeh.models import Tabs
from bokeh.models import Toggle

from estimagic.dashboard.utilities import create_styled_figure
from estimagic.dashboard.utilities import get_color_palette
from estimagic.logging.database_utilities import load_database
from estimagic.logging.database_utilities import read_last_rows
from estimagic.logging.database_utilities import read_new_rows
from estimagic.logging.database_utilities import transpose_nested_list


def monitoring_app(doc, database_name, session_data, rollover):
    """Create plots showing the development of the criterion and parameters.

    Args:
        doc (bokeh.Document): Argument required by bokeh.
        database_name (str): Short and unique name of the database.
        session_data (dict): Infos to be passed between and within apps.
            Keys of this app's entry are:
            - last_retrieved (int): last iteration currently in the ColumnDataSource.
            - database_path (str or pathlib.Path)
            - callbacks (dict): dictionary to be populated with callbacks.

    """

    database = load_database(path=session_data["database_path"])
    optimization_problem = read_last_rows(
        database=database,
        # todo: need to adjust table_name with suffix if necessary
        table_name="optimization_problem",
        n_rows=1,
        return_type="dict_of_lists",
    )
    optimization_problem = {key: val[0] for key, val in optimization_problem.items()}

    start_params = optimization_problem["params"]
    param_names = start_params["name"].tolist()
    crit_data = {"iteration": [], "criterion": []}
    criterion_history = ColumnDataSource(crit_data, name="criterion_history_cds")

    params_data = {"iteration": []}
    for name in param_names:
        params_data[name] = []
    params_history = ColumnDataSource(params_data, name="params_history_cds")

    session_data["last_retrieved"] = 0

    # create initial bokeh elements without callbacks
    initial_convergence_plots = _create_initial_convergence_plots(
        criterion_history=criterion_history,
        params_history=params_history,
        start_params=start_params,
    )

    activation_button = Toggle(
        active=False,
        label="Start Updating from Database",
        button_type="danger",
        width=50,
        height=30,
        name="activation_button",
    )

    # add elements to bokeh Document
    bokeh_convergence_elements = [Row(activation_button)] + initial_convergence_plots
    convergence_tab = Panel(
        child=Column(*bokeh_convergence_elements), title="Convergence Tab"
    )
    tabs = Tabs(tabs=[convergence_tab])
    doc.add_root(tabs)

    tables = ["criterion_history", "params_history"]

    # add callbacks
    activation_callback = partial(
        _activation_callback,
        button=activation_button,
        doc=doc,
        database=database,
        session_data=session_data,
        rollover=rollover,
        tables=tables,
        start_params=start_params,
    )
    activation_button.on_change("active", activation_callback)


def _create_initial_convergence_plots(criterion_history, params_history, start_params):
    """Create the initial convergence plots.

    Args:
        criterion_history (bokeh ColumnDataSource)
        params_history (bokeh ColumnDataSource)
        start_params (pd.DataFrame): params DataFrame that includes the "group" column.

    Returns:
        convergence_plots (list): List of bokeh Row elements, each containing one
            convergence plot.

    """
    criterion_plot = _plot_time_series(
        data=criterion_history,
        x_name="iteration",
        y_keys=["criterion"],
        y_names=["criterion"],
        title="Criterion",
    )
    convergence_plots = [criterion_plot]

    group_to_params = _map_groups_to_params(start_params)
    for g, group_params in group_to_params.items():
        if g not in ["", None, False] and g == g:  # the last checks that g is not NaN
            param_group_plot = _plot_time_series(
                data=params_history,
                y_keys=group_params,
                x_name="iteration",
                title=str(g),
            )
            convergence_plots.append(Row(param_group_plot))
    return convergence_plots


def _plot_time_series(data, y_keys, x_name, title, y_names=None):
    """Plot time series linking the *y_keys* to a common *x_name* variable.

    Args:
        data (ColumnDataSource):
            data that contain the y_keys and x_name
        y_keys (list):
            list of the entries in the data that are to be plotted.
        x_name (str):
            name of the entry in the data that will be on the x axis.
        title (str):
            title of the plot.
        y_names (list):
            if given these replace the y keys for the names of the lines.

    Returns:
        plot (bokeh Figure)

    """
    if y_names is None:
        y_names = [str(key) for key in y_keys]

    plot = create_styled_figure(title=title)
    colors = get_color_palette(nr_colors=len(y_keys))

    legend_items = [(" " * 60, [])]
    for color, y_key, y_name in zip(colors, y_keys, y_names):
        # the last checks that y_key is not NaN
        if y_key not in ["", None, False] and y_key == y_key:
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


def _map_groups_to_params(params):
    """Map the group name to the ColumnDataSource friendly parameter names.

    Args:
        params (pd.DataFrame):
            DataFrame with the parameter values and additional information such as the
            "group" column and Index.

    Returns:
        group_to_params (dict):
            Keys are the values of the "group" column. The values are lists with
            bokeh friendly strings of the index tuples identifying the parameters
            that belong to this group. Parameters where group is None are ignored.

    """
    group_to_params = {}
    actual_groups = [group for group in params["group"].unique() if group is not None]
    for group in actual_groups:
        group_to_params[group] = list(params[params["group"] == group]["name"])
    return group_to_params


def _activation_callback(
    attr, old, new, session_data, rollover, doc, database, button, tables, start_params
):
    """Start and reset the convergence plots and their updating.

    Args:
        attr: Required by bokeh.
        old: Old state of the Button.
        new: New state of the Button.

        doc (bokeh.Document)
        database (sqlalchemy.MetaData)
        session_data (dict): This app's entry of infos to be passed between and within
            apps. The keys are:
            - last_retrieved (int): last iteration currently in the ColumnDataSource
            - database_path
        rollover (int): Maximal number of points to show in the plot.
        tables (list): List of table names to load and convert to ColumnDataSources.
        start_params (pd.DataFrame): See :ref:`params`

    """
    callback_dict = session_data["callbacks"]
    if new is True:
        plot_new_data = partial(
            _update_monitoring_tab,
            doc=doc,
            database=database,
            session_data=session_data,
            rollover=rollover,
            tables=tables,
            start_params=start_params,
        )
        callback_dict["plot_periodic_data"] = doc.add_periodic_callback(
            plot_new_data, period_milliseconds=200
        )
        # change the button color
        button.button_type = "success"
        button.label = "Reset Plot"
    else:
        doc.remove_periodic_callback(callback_dict["plot_periodic_data"])
        for table_name in ["criterion_history", "params_history"]:
            cds = doc.get_model_by_name(f"{table_name}_cds")
            column_names = cds.data.keys()
            cds.data = {name: [] for name in column_names}
        session_data["last_retrieved"] = 0
        # change the button color
        button.button_type = "danger"
        button.label = "Restart Plot"


def _update_monitoring_tab(doc, database, session_data, tables, rollover, start_params):
    """Callback to look up new entries in the database tables and plot them.

    Args:
        doc (bokeh.Document): argument required by bokeh
        database (sqlalchemy.MetaData)
        session_data (dict):
            infos to be passed between and within apps.
            Keys of this app's entry are:
            - last_retrieved (int): last iteration currently in the ColumnDataSource
            - database_path
        tables (list): list of table names to load and convert to ColumnDataSources
        rollover (int): maximal number of points to show in the plot

    """
    data, new_last = read_new_rows(
        database=database,
        table_name="optimization_iterations",
        last_retrieved=session_data["last_retrieved"],
        return_type="dict_of_lists",
        limit=20,
    )

    # update the criterion plot
    cds = doc.get_model_by_name("criterion_history_cds")
    # todo: remove None entries!
    missing = [i for i, val in enumerate(data["value"]) if val is None]
    crit_data = {
        "iteration": [id_ for i, id_ in enumerate(data["rowid"]) if i not in missing],
        "criterion": [val for i, val in enumerate(data["value"]) if i not in missing],
    }

    cds.stream(crit_data, rollover=rollover)

    # update the parameter plots
    param_names = start_params["name"].tolist()
    cds = doc.get_model_by_name("params_history_cds")
    params_data = [arr.tolist() for arr in data["external_params"]]
    params_data = transpose_nested_list(params_data)
    params_data = dict(zip(param_names, params_data))
    if params_data == {}:
        params_data = {name: [] for name in param_names}
    params_data["iteration"] = data["rowid"]
    cds.stream(params_data, rollover=rollover)

    # update last retrieved
    session_data["last_retrieved"] = new_last
