"""Show the development of one optimization's criterion and parameters over time."""
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from bokeh.layouts import Column
from bokeh.layouts import Row
from bokeh.models import ColumnDataSource
from bokeh.models import Panel
from bokeh.models import Tabs
from bokeh.models import Toggle
from estimagic.dashboard.monitoring_callbacks import activation_callback
from estimagic.dashboard.monitoring_callbacks import logscale_callback
from estimagic.dashboard.plot_functions import plot_time_series
from estimagic.logging.database_utilities import load_database
from estimagic.logging.database_utilities import read_last_rows
from estimagic.logging.read_log import read_start_params
from jinja2 import Environment
from jinja2 import FileSystemLoader


def monitoring_app(
    doc,
    database_name,
    session_data,
    updating_options,
    start_immediately,
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
        updating_options (dict): Specification how to update the plotting data.
            It contains rollover, update_frequency, update_chunk, jump and stride.

    """
    # style the Document
    template_folder = Path(__file__).resolve().parent
    # conversion to string from pathlib Path is necessary for FileSystemLoader
    env = Environment(loader=FileSystemLoader(str(template_folder)))
    doc.template = env.get_template("index.html")

    # process inputs
    database = load_database(path=session_data["database_path"])
    start_point = _calculate_start_point(database, updating_options)
    session_data["last_retrieved"] = start_point
    start_params = read_start_params(path_or_database=database)
    start_params["id"] = _create_id_column(start_params)
    group_to_param_ids = _map_group_to_other_column(start_params, "id")
    group_to_param_names = _map_group_to_other_column(start_params, "name")
    criterion_history, params_history = _create_cds_for_monitoring_app(
        group_to_param_ids
    )

    # create elements
    button_row = _create_button_row(
        doc=doc,
        database=database,
        session_data=session_data,
        start_params=start_params,
        updating_options=updating_options,
    )
    monitoring_plots = _create_initial_convergence_plots(
        criterion_history=criterion_history,
        params_history=params_history,
        group_to_param_ids=group_to_param_ids,
        group_to_param_names=group_to_param_names,
    )

    # add elements to bokeh Document
    grid = Column(children=[button_row, *monitoring_plots], sizing_mode="stretch_width")
    convergence_tab = Panel(child=grid, title="Convergence Tab")
    tabs = Tabs(tabs=[convergence_tab])

    doc.add_root(tabs)

    if start_immediately:
        activation_button = doc.get_model_by_name("activation_button")
        activation_button.active = True


def _create_id_column(df):
    """Create a column that gives the position for plotted parameters and is None else.

    Args:
        df (pd.DataFrame)

    Returns:
        ids (pd.Series): integer position in the DataFrame unless the group was
            None, False, np.nan or an empty string.

    """
    ids = pd.Series(range(len(df)), dtype=object, index=df.index)
    ids[df["group"].isin([None, False, np.nan, ""])] = None
    return ids.astype(str)


def _map_group_to_other_column(params, column_name):
    """Map the group name to lists of one column's values of the group's parameters.

    Args:
        params (pd.DataFrame): Includes the "group" and "id" columns.
        column_name (str): name of the column for which to return the parameter values.

    Returns:
        group_to_values (dict): Keys are the values of the "group" column.
            The values are lists of parameter values of the parameters belonging
            to the particular group.

    """
    to_plot = params[~params["group"].isin([None, False, np.nan, ""])]
    group_to_indices = to_plot.groupby("group").groups
    group_to_values = {}
    for group, loc in group_to_indices.items():
        group_to_values[group] = to_plot[column_name].loc[loc].tolist()
    return group_to_values


def _create_cds_for_monitoring_app(group_to_param_ids):
    """Create the ColumnDataSources for saving the criterion and parameter values.

    They will be periodically updated from the database.
    There is a ColumnDataSource for all parameters and one for the criterion value.
    The "x" column is called "iteration".

    Args:
        group_to_param_ids (dict): Keys are the groups to be plotted. The values are
            the ids of the parameters belonging to the particular group.

    Returns:
        criterion_history (bokeh.ColumnDataSource)
        params_history (bokeh.ColumnDataSource)

    """
    crit_data = {"iteration": [], "criterion": []}
    criterion_history = ColumnDataSource(crit_data, name="criterion_history_cds")

    param_ids = []
    for id_list in group_to_param_ids.values():
        param_ids += id_list
    params_data = {id_: [] for id_ in param_ids + ["iteration"]}
    params_history = ColumnDataSource(params_data, name="params_history_cds")

    return criterion_history, params_history


def _calculate_start_point(database, updating_options):
    """Calculate the starting point.

    Args:
        database (sqlalchemy.MetaData): Bound metadata object.
        updating_options (dict): Specification how to update the plotting data.
            It contains rollover, update_frequency, update_chunk, jump and stride.

    Returns:
        start_point (int): iteration from which to start the dashboard.

    """
    if updating_options["jump"]:
        last_entry = read_last_rows(
            database=database,
            table_name="optimization_iterations",
            n_rows=1,
            return_type="list_of_dicts",
        )
        nr_of_entries = last_entry[0]["rowid"]
        nr_to_go_back = updating_options["rollover"] * updating_options["stride"]
        start_point = max(0, nr_of_entries - nr_to_go_back)
    else:
        start_point = 0
    return start_point


def _create_initial_convergence_plots(
    criterion_history,
    params_history,
    group_to_param_ids,
    group_to_param_names,
):
    """Create the initial convergence plots.

    Args:
        criterion_history (bokeh ColumnDataSource)
        params_history (bokeh ColumnDataSource)
        group_to_param_ids (dict): Keys are the groups to be plotted. Values are the
            ids of the parameters belonging to the respective group.
        group_to_param_names (dict): Keys are the groups to be plotted. Values are the
            names of the parameters belonging to the respective group.

    Returns:
        convergence_plots (list): List of bokeh Row elements, each containing one
            convergence plot.

    """
    param_plots = []
    for group, param_ids in group_to_param_ids.items():
        param_names = group_to_param_names[group]
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
    doc,
    database,
    session_data,
    start_params,
    updating_options,
):
    """Create a row with two buttons, one for (re)starting and one for scale switching.

    Args:
        doc (bokeh.Document)
        database (sqlalchemy.MetaData): Bound metadata object.
        session_data (dict): dictionary with the last retrieved rowid
        start_params (pd.DataFrame): See :ref:`params`
        updating_options (dict): Specification how to update the plotting data.
            It contains rollover, update_frequency, update_chunk, jump and stride.

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
        tables=["criterion_history", "params_history"],
        start_params=start_params,
        updating_options=updating_options,
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
        logscale_callback,
        button=logscale_button,
        doc=doc,
    )
    logscale_button.on_change("active", partialed_logscale_callback)

    button_row = Row(children=[activation_button, logscale_button], name="button_row")
    return button_row
