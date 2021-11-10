from functools import partial

import numpy as np
from estimagic.logging.database_utilities import read_new_rows
from estimagic.logging.database_utilities import transpose_nested_list


def reset_and_start_convergence(
    attr,
    old,
    new,
    session_data,
    doc,
    database,
    button,
    start_params,
    updating_options,
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
        button (bokeh.models.Toggle)
        start_params (pd.DataFrame)
        updating_options (dict): Specification how to update the plotting data.
            It contains rollover, update_frequency, update_chunk, jump and stride.

    """
    callback_dict = session_data["callbacks"]
    criterion_cds = doc.get_model_by_name("criterion_history_cds")
    param_cds = doc.get_model_by_name("params_history_cds")

    if new is True:
        plot_new_data = partial(
            _update_convergence_plots,
            criterion_cds=criterion_cds,
            param_cds=param_cds,
            database=database,
            session_data=session_data,
            start_params=start_params,
            rollover=updating_options["rollover"],
            update_chunk=updating_options["update_chunk"],
            stride=updating_options["stride"],
        )
        callback_dict["plot_periodic_data"] = doc.add_periodic_callback(
            plot_new_data,
            period_milliseconds=1000 * updating_options["update_frequency"],
        )

        # change the button color
        button.button_type = "success"
        button.label = "Reset Plot"
    else:
        doc.remove_periodic_callback(callback_dict["plot_periodic_data"])
        _reset_column_data_sources([criterion_cds, param_cds])
        session_data["last_retrieved"] = 0

        # change the button color
        button.button_type = "danger"
        button.label = "Restart Plot"


def _update_convergence_plots(
    database,
    criterion_cds,
    param_cds,
    session_data,
    start_params,
    rollover,
    update_chunk,
    stride,
):
    """Callback to look up new entries in the database and plot them.

    Args:
        database (sqlalchemy.MetaData)
        session_data (dict):
            infos to be passed between and within apps.
            Keys of this app's entry are:
            - last_retrieved (int): last iteration currently in the ColumnDataSource
            - database_path
        start_params (pd.DataFrame)
        rollover (int): maximal number of points to show in the plot
        update_chunk (int): Number of values to add at each update.
        criterion_cds (bokeh.ColumnDataSource)
        param_cds (bokeh.ColumnDataSource)
        stride (int): Plot every stride_th database row in the dashboard. Note that
            some database rows only contain gradient evaluations, thus for some values
            of stride the convergence plot of the criterion function can be empty.

    """
    clip_bound = np.finfo(float).max
    data, new_last = read_new_rows(
        database=database,
        table_name="optimization_iterations",
        last_retrieved=session_data["last_retrieved"],
        return_type="dict_of_lists",
        limit=update_chunk,
        stride=stride,
    )

    # update the criterion plot
    # todo: remove None entries!
    missing = [i for i, val in enumerate(data["value"]) if val is None]
    crit_data = {
        "iteration": [id_ for i, id_ in enumerate(data["rowid"]) if i not in missing],
        "criterion": [
            np.clip(val, -clip_bound, clip_bound)
            for i, val in enumerate(data["value"])
            if i not in missing
        ],
    }
    _stream_data(cds=criterion_cds, data=crit_data, rollover=rollover)

    # update the parameter plots
    # Note: we need **all** parameter ids to correctly map them to the parameter entries
    # in the database. Only after can we restrict them to the entries we need.
    param_ids = start_params["id"].tolist()
    params_data = _create_params_data_for_update(data, param_ids, clip_bound)
    _stream_data(cds=param_cds, data=params_data, rollover=rollover)
    # update last retrieved
    session_data["last_retrieved"] = new_last


def _create_params_data_for_update(data, param_ids, clip_bound):
    """Create the dictionary to stream to the param_cds from data and param_ids.

    Args:
        data
        param_ids (list): list of the length of the arrays in data["params"]
        clip_bound (float)

    Returns:
        params_data (dict): keys are the parameter names and "iteration". The values
            are lists of values that will be added to the ColumnDataSources columns.

    """
    params_data = [
        np.clip(arr, -clip_bound, clip_bound).tolist() for arr in data["params"]
    ]
    params_data = transpose_nested_list(params_data)
    params_data = dict(zip(param_ids, params_data))
    if params_data == {}:
        params_data = {name: [] for name in param_ids}
    params_data["iteration"] = data["rowid"]
    return params_data


def _stream_data(cds, data, rollover):
    """Stream only to the available columns of a ColumnDataSource.

    Args:
        cds (bokeh.ColumnDataSource): to be updated
        data (dict): keys are the columns of the CDS to which to stream.
            The values are the entries to be appended. Keys that are not
            in the columns of **cds** will not be streamed.
        rollover (int): maximal number of points to show in the plot

    """
    available_keys = cds.data.keys()
    to_stream = {k: v for k, v in data.items() if k in available_keys}
    cds.stream(to_stream, rollover=rollover)


def _reset_column_data_sources(cds_list):
    """Empty each ColumnDataSource in a list such that it has no entries.

    Args:
        cds_list (list): list of boheh ColumnDataSources
    """
    for cds in cds_list:
        column_names = cds.data.keys()
        cds.data = {name: [] for name in column_names}
