"""Show the development of one optimization's criterion and parameters over time."""
from functools import partial
from pathlib import Path

import pandas as pd
from bokeh.layouts import Column
from bokeh.layouts import Row
from bokeh.models import ColumnDataSource
from bokeh.models import Panel
from bokeh.models import Tabs
from bokeh.models import Toggle
from jinja2 import Environment
from jinja2 import FileSystemLoader

from estimagic.dashboard.monitoring_callbacks import activation_callback
from estimagic.dashboard.monitoring_callbacks import logscale_callback
from estimagic.dashboard.plot_functions import plot_time_series
from estimagic.logging.database_utilities import load_database
from estimagic.logging.database_utilities import read_last_rows


def monitoring_app(
    doc, database_name, session_data, rollover, jump, update_frequency, update_chunk
):
    """Create plots showing the development of the criterion and parameters.

    Args:
        doc (bokeh.Document): Argument required by bokeh.
        database_name (str): Short and unique name of the database.
        session_data (dict): Infos to be passed between and within apps.
            Keys of this app's entry are:
            - last_retrieved (int): last iteration currently in the ColumnDataSource.
            - database_path (str or pathlib.Path)
            - callbacks (dict): dictionary to be populated with callbacks.
        jump (bool): If True the dashboard will start at the last `rollover`
            observations and start to display the history from there.
        update_frequency (float): Number of seconds to wait between updates.
        update_chunk (int): Number of values to add at each update.

    """
    # style the Document
    template_folder = Path(__file__).resolve().parent
    # conversion to string from pathlib Path is necessary for FileSystemLoader
    env = Environment(loader=FileSystemLoader(str(template_folder)))
    doc.template = env.get_template("index.html")

    # process inputs
    database = load_database(path=session_data["database_path"])
    start_point = _calculate_start_point(database, rollover, jump)
    session_data["last_retrieved"] = start_point
    start_params, group_to_param_ids = _get_group_to_param_ids_from_database(database)
    criterion_history, params_history = _create_cds_for_monitoring_app(
        start_params, group_to_param_ids
    )

    # create elements
    button_row = _create_button_row(
        doc=doc,
        database=database,
        session_data=session_data,
        rollover=rollover,
        start_params=start_params,
        update_frequency=update_frequency,
        update_chunk=update_chunk,
    )
    monitoring_plots = _create_initial_convergence_plots(
        criterion_history=criterion_history,
        params_history=params_history,
        group_to_param_ids=group_to_param_ids,
        start_params=start_params,
    )

    # add elements to bokeh Document
    grid = Column(children=[button_row, *monitoring_plots], sizing_mode="stretch_width")
    convergence_tab = Panel(child=grid, title="Convergence Tab")
    tabs = Tabs(tabs=[convergence_tab])

    doc.add_root(tabs)


def _get_group_to_param_ids_from_database(database):
    """Map each group name to the parameters' names that belong to it.

    Args:
        database (sqlalchemy.MetaData): Bound metadata object.

    Returns:
        group_to_param_ids (dict): keys are the group names, values are parameter names.

    """
    optimization_problem = read_last_rows(
        database=database,
        table_name="optimization_problem",
        n_rows=1,
        return_type="dict_of_lists",
    )
    start_params = optimization_problem["params"][0]
    start_params["id"] = _create_id_column(start_params)
    group_to_param_ids = _map_groups_to_param_ids(start_params)
    return start_params, group_to_param_ids


def _create_id_column(df):
    if isinstance(df.index, pd.MultiIndex):
        index_df = df.index.to_frame()
        ids = index_df.astype(str).apply(lambda x: "_".join(x), axis=1)
    else:
        ids = df.index.to_series().astype(str)
    # we check in optimize that there are no duplicates in the index but
    # it is theoretically possible that the ids still have duplicate ids
    # because everything is converted to strings,
    # i.e. 1 and "1" would become identical and could introduce duplicates.
    assert not ids.duplicated().any(), "Duplicates in the dashboard's id column."
    return ids


def _map_groups_to_param_ids(params):
    """Map the group name to the ColumnDataSource friendly parameter names.

    Args:
        params (pd.DataFrame):
            DataFrame with the parameter values and additional information such as the
            "group" column and Index.

    Returns:
        group_to_param_ids (dict):
            Keys are the values of the "group" column. The values are lists with
            bokeh friendly strings of the index tuples identifying the parameters
            that belong to this group. Parameters where group is None, "" or False
            are ignored.

    """
    group_to_param_ids = {}
    for group in params["group"].unique():
        if group is not None and group == group and group != "" and group is not False:
            group_to_param_ids[group] = list(params[params["group"] == group]["id"])
    return group_to_param_ids


def _create_cds_for_monitoring_app(start_params, group_to_param_ids):
    """Create the ColumnDataSources for saving the criterion and parameter values.

    They will be periodically updated from the database.
    There is a ColumnDataSource for all parameters and one for the criterion value.
    The "x" column is called "iteration".

    Args:
        start_params (pd.DataFrame): See :ref:`params`. It includes a dashboard
            specific id column.
        group_to_param_ids ()

    Returns:
        criterion_history (bokeh.ColumnDataSource)
        params_history (bokeh.ColumnDataSource)

    """
    crit_data = {"iteration": [], "criterion": []}
    criterion_history = ColumnDataSource(crit_data, name="criterion_history_cds")

    param_ids = [id_ for id_list in group_to_param_ids.values() for id_ in id_list]
    params_data = {"iteration": []}
    for id_ in param_ids:
        params_data[id_] = []
    params_history = ColumnDataSource(params_data, name="params_history_cds")

    return criterion_history, params_history


def _calculate_start_point(database, rollover, jump):
    """Calculate the starting point.

    Args:
        database (sqlalchemy.MetaData): Bound metadata object.
        rollover (int): Upper limit to how many iterations are displayed.
        jump (bool): If True the dashboard will start at the last `rollover`
            observations and start to display the history from there.
    Returns:
        start_point (int): iteration from which to start the dashboard.

    """
    if jump:
        last_entry = read_last_rows(
            database=database,
            table_name="optimization_iterations",
            n_rows=1,
            return_type="list_of_dicts",
        )
        start_point = max(0, last_entry[0]["rowid"] - rollover)
    else:
        start_point = 0
    return start_point


def _create_initial_convergence_plots(
    criterion_history, params_history, group_to_param_ids, start_params,
):
    """Create the initial convergence plots.

    Args:
        criterion_history (bokeh ColumnDataSource)
        params_history (bokeh ColumnDataSource)
        group_to_param_ids (dict):
            Keys are the values of the "group" column. The values are lists with
            bokeh friendly strings of the index tuples identifying the parameters
            that belong to this group. Parameters where group is None, "" or False
            are ignored.
        start_params (pd.DataFrame): See :ref:`params`. It includes a dashboard
            specific id column.

    Returns:
        convergence_plots (list): List of bokeh Row elements, each containing one
            convergence plot.

    """
    param_plots = []
    for group, param_ids in group_to_param_ids.items():
        param_names = [
            start_params.query(f"id == '{id_}'")["name"].iloc[0] for id_ in param_ids
        ]
        param_group_plot = plot_time_series(
            data=params_history,
            y_keys=param_ids,
            y_names=param_names,
            x_name="iteration",
            title=str(group),
        )
        param_plots.append(param_group_plot)

    arranged_param_plots = [Row(plot) for plot in param_plots]

    linear_criterion_plot = plot_time_series(
        data=criterion_history,
        x_name="iteration",
        y_keys=["criterion"],
        y_names=["criterion"],
        title="Criterion",
        name="linear_criterion_plot",
        logscale=False,
    )
    log_criterion_plot = plot_time_series(
        data=criterion_history,
        x_name="iteration",
        y_keys=["criterion"],
        y_names=["criterion"],
        title="Criterion",
        name="log_criterion_plot",
        logscale=True,
    )
    log_criterion_plot.visible = False

    plot_list = [
        Row(linear_criterion_plot),
        Row(log_criterion_plot),
    ] + arranged_param_plots
    return plot_list


def _create_button_row(
    doc, database, session_data, rollover, start_params, update_frequency, update_chunk,
):
    """Create a row with two buttons, one for (re)starting and one for scale switching.

    Args:
        doc (bokeh.Document)
        database (sqlalchemy.MetaData): Bound metadata object.
        session_data (dict): dictionary with the last retrieved rowid
        rollover (int): Upper limit to how many iterations are displayed.
        start_params (pd.DataFrame): See :ref:`params`
        update_frequency (float): Number of seconds to wait between updates.
        update_chunk (int): Number of values to add at each update.

    Returns:
        bokeh.layouts.Row

    """
    # (Re)start convergence plot button
    activation_button = Toggle(
        active=False,
        label="Start Updating",
        button_type="danger",
        width=200,
        height=30,
        name="activation_button",
    )
    partialed_activation_callback = partial(
        activation_callback,
        button=activation_button,
        doc=doc,
        database=database,
        session_data=session_data,
        rollover=rollover,
        tables=["criterion_history", "params_history"],
        start_params=start_params,
        update_frequency=update_frequency,
        update_chunk=update_chunk,
    )
    activation_button.on_change("active", partialed_activation_callback)

    # switch between linear and logscale button
    logscale_button = Toggle(
        active=False,
        label="Show criterion plot on a logarithmic scale",
        button_type="default",
        width=200,
        height=30,
        name="logscale_button",
    )
    partialed_logscale_callback = partial(
        logscale_callback, button=logscale_button, doc=doc,
    )
    logscale_button.on_change("active", partialed_logscale_callback)

    button_row = Row(children=[activation_button, logscale_button], name="button_row")
    return button_row
