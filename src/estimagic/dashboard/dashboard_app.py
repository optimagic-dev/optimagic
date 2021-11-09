"""Show the development of one database's criterion and parameters over time."""
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from bokeh.layouts import Column
from bokeh.layouts import Row
from bokeh.models import ColumnDataSource
from bokeh.models import Div
from bokeh.models import Toggle
from estimagic.dashboard.callbacks import reset_and_start_convergence
from estimagic.dashboard.plot_functions import plot_time_series
from estimagic.logging.database_utilities import load_database
from estimagic.logging.database_utilities import read_last_rows
from estimagic.logging.read_log import read_start_params
from jinja2 import Environment
from jinja2 import FileSystemLoader


def dashboard_app(
    doc,
    session_data,
    updating_options,
):
    """Create plots showing the development of the criterion and parameters.

    Args:
        doc (bokeh.Document): Argument required by bokeh.
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
    criterion_history, params_history = _create_cds_for_dashboard(group_to_param_ids)

    # create elements
    title_text = """<h1 style="font-size:30px;">estimagic Dashboard</h1>"""
    title = Row(
        children=[
            Div(
                text=title_text,
                sizing_mode="scale_width",
            )
        ],
        name="title",
        margin=(5, 5, -20, 5),
    )
    plots = _create_initial_plots(
        criterion_history=criterion_history,
        params_history=params_history,
        group_to_param_ids=group_to_param_ids,
        group_to_param_names=group_to_param_names,
    )

    restart_button = _create_restart_button(
        doc=doc,
        database=database,
        session_data=session_data,
        start_params=start_params,
        updating_options=updating_options,
    )
    button_row = Row(
        children=[restart_button],
        name="button_row",
    )

    # add elements to bokeh Document
    grid = Column(children=[title, button_row, *plots], sizing_mode="stretch_width")
    doc.add_root(grid)

    # start the convergence plot immediately
    # this must happen here befo
    restart_button.active = True


def _create_id_column(df):
    """Create a column that gives the position for plotted parameters and is None else.

    Args:
        df (pd.DataFrame): DataFrame with "group" column.

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


def _create_cds_for_dashboard(group_to_param_ids):
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


def _create_initial_plots(
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
        plots (list): List of bokeh Row elements, each containing one convergence plot.

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

    criterion_plot = plot_time_series(
        data=criterion_history,
        x_name="iteration",
        y_keys=["criterion"],
        y_names=["criterion"],
        title="Criterion",
        name="criterion_plot",
    )

    plots = [Row(criterion_plot)] + arranged_param_plots
    return plots


def _create_restart_button(
    doc,
    database,
    session_data,
    start_params,
    updating_options,
):
    """Create the button that restarts the convergence plots.

    Args:
        doc (bokeh.Document)
        database (sqlalchemy.MetaData): Bound metadata object.
        session_data (dict): dictionary with the last retrieved row id
        start_params (pd.DataFrame): See :ref:`params`
        updating_options (dict): Specification how to update the plotting data.
            It contains rollover, update_frequency, update_chunk, jump and stride.

    Returns:
        bokeh.layouts.Row

    """
    # (Re)start convergence plot button
    restart_button = Toggle(
        active=False,
        label="Start Updating",
        button_type="danger",
        width=200,
        height=30,
        name="restart_button",
    )
    restart_callback = partial(
        reset_and_start_convergence,
        session_data=session_data,
        doc=doc,
        database=database,
        button=restart_button,
        start_params=start_params,
        updating_options=updating_options,
    )
    restart_button.on_change("active", restart_callback)
    return restart_button
