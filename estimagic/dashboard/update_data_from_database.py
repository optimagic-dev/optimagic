from functools import partial
from time import sleep

from tornado import gen

from estimagic.logging.database_utilities import read_new_rows
from estimagic.logging.database_utilities import transpose_nested_list


def update_data_from_database(
    doc,
    flag,
    database,
    update_frequency,
    update_chunk,
    session_data,
    criterion_history,
    params_history,
    start_params,
):
    """While flag contains True check the database and update the ColumnDataSources.

    Args:
            doc (bokeh.Document): used for adding the next_tick_callback
            flag (np.array): set to False as signal to terminate by the button
    database (sqlalchemy.MetaData): Bound metadata object.
    update_frequency (float): Number of seconds to wait between updates.
    update_chunk (int): Number of values to add at each update.
    session_data (dict): Infos to be passed between and within apps.
        Keys of this app's entry are:
        - last_retrieved (int): last iteration currently in the ColumnDataSource.
        - database_path (str or pathlib.Path)
        - callbacks (dict): dictionary to be populated with callbacks.
    criterion_history (bokeh.ColumnDataSource)
    params_history (bokeh.ColumnDataSource)
    start_params (pd.DataFrame)

    """
    while flag[0]:  # careful: np.bool type!
        data, new_last = read_new_rows(
            database=database,
            table_name="optimization_iterations",
            last_retrieved=session_data["last_retrieved"],
            return_type="dict_of_lists",
            limit=update_chunk,
        )

        # todo: remove None entries!
        missing = [i for i, val in enumerate(data["value"]) if val is None]
        new_crit_data = {
            "iteration": [
                id_ for i, id_ in enumerate(data["rowid"]) if i not in missing
            ],
            "criterion": [
                val for i, val in enumerate(data["value"]) if i not in missing
            ],
        }

        param_names = start_params["name"].tolist()
        new_params_data = _convert_params_data_for_update(data, param_names)

        doc.add_next_tick_callback(
            partial(
                _update_criterion_and_param_history,
                criterion_history=criterion_history,
                params_history=params_history,
                new_crit_data=new_crit_data,
                new_params_data=new_params_data,
            )
        )

        session_data["last_retrieved"] = new_last
        sleep(update_frequency)


def _convert_params_data_for_update(data, param_names):
    """Create the dictionary to stream to the param_cds from data and param_names.

    Args:
        data
        param_names (list): list of the length of the arrays in data["external_params"]

    Returns:
        params_data (dict): keys are the parameter names and "iteration". The values
            are lists of values that will be added to the ColumnDataSources columns.

    """
    params_data = [arr.tolist() for arr in data["external_params"]]
    params_data = transpose_nested_list(params_data)
    params_data = dict(zip(param_names, params_data))
    if params_data == {}:
        params_data = {name: [] for name in param_names}
    params_data["iteration"] = data["rowid"]
    return params_data


@gen.coroutine
def _update_criterion_and_param_history(
    criterion_history, params_history, new_crit_data, new_params_data
):
    """Stream the new data to the respective ColumnDataSources.

    Args:
            criterion_history (bokeh.ColumnDataSource)
            params_history (bokeh.ColumnDataSource)
            new_crit_data (dict)
            new_params_data (dict)

    """
    criterion_history.stream(new_crit_data)
    params_history.stream(new_params_data)
