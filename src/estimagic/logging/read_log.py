"""Functions to read data from the database used for logging.

The functions in the module are meant for end users of estimagic.
They do not require any knowledge of databases.

When using them internally (e.g. in the dashboard), make sure to supply a database to
path_or_database. Otherwise, the functions may be very slow.

"""
from pathlib import Path

import pandas as pd
from estimagic.logging.database_utilities import load_database
from estimagic.logging.database_utilities import read_last_rows
from estimagic.logging.database_utilities import read_new_rows
from estimagic.logging.database_utilities import read_specific_row
from sqlalchemy import MetaData


def read_optimization_iteration(path_or_database, iteration, include_internals=False):
    """Get information about an optimization iteration.

    Args:
        path_or_database (pathlib.Path, str or sqlalchemy.MetaData)
        iteration (int): The index of the iteration that should be retrieved.
            The row_id behaves as Python list indices, i.e. ``0`` identifies the
            first iteration, ``-1`` the last one, etc.
        include_internals (bool): Whether internally used quantities like the
            internal parameter vector and the corresponding derivative etc. are included
            in the result. Default False. This should only be used by advanced users.

    Returns:
        dict: The logged information corresponding to the iteration. The keys correspond
            to database columns.

    Raises:
        KeyError: if the iteration is out of bounds.

    """
    database = load_database(**_process_path_or_database(path_or_database))
    start_params = read_start_params(database)
    if iteration >= 0:
        rowid = iteration + 1
    else:
        last_iteration = read_last_rows(
            database=database,
            table_name="optimization_iterations",
            n_rows=1,
            return_type="list_of_dicts",
        )
        highest_rowid = last_iteration[0]["rowid"]

        rowid = highest_rowid + iteration + 1

    data = read_specific_row(
        database=database,
        table_name="optimization_iterations",
        rowid=rowid,
        return_type="list_of_dicts",
    )

    if len(data) == 0:
        raise IndexError(f"Invalid iteration requested: {iteration}")
    else:
        data = data[0]

    params = start_params.copy()
    params["value"] = data.pop("params")
    data["params"] = params

    to_remove = ["distance_origin", "distance_ones"]
    if not include_internals:
        to_remove += ["internal_params", "internal_derivative"]
    for key in to_remove:
        if key in data:
            del data[key]

    return data


def read_start_params(path_or_database):
    """Load the start parameters DataFrame.

    Args:
        path_or_database (pathlib.Path, str or sqlalchemy.MetaData)

    Returns:
        params (pd.DataFrame): see :ref:`params`.

    """
    database = load_database(**_process_path_or_database(path_or_database))
    optimization_problem = read_last_rows(
        database=database,
        table_name="optimization_problem",
        n_rows=1,
        return_type="dict_of_lists",
    )
    start_params = optimization_problem["params"][0]
    return start_params


def read_optimization_histories(path_or_database):
    """Read a histories out values, parameters and other information."""
    database = load_database(**_process_path_or_database(path_or_database))

    start_params = read_start_params(path_or_database)

    raw_res, _ = read_new_rows(
        database=database,
        table_name="optimization_iterations",
        last_retrieved=0,
        return_type="dict_of_lists",
    )

    params_history = pd.DataFrame(raw_res["params"], columns=start_params.index)
    value_history = pd.Series(raw_res["value"])

    metadata = pd.DataFrame()
    metadata["timestamps"] = raw_res["timestamp"]
    metadata["valid"] = raw_res["valid"]
    metadata["has_value"] = value_history.notnull()
    metadata["has_derivative"] = [d is not None for d in raw_res["internal_derivative"]]

    histories = {
        "values": value_history.dropna(),
        "params": params_history,
        "metadata": metadata,
    }

    if "contributions" in raw_res:
        first_contrib = raw_res["contributions"][0]
        if isinstance(first_contrib, pd.Series):
            columns = first_contrib.index
        else:
            columns = None
        contributions_history = pd.DataFrame(
            raw_res["contributions"], columns=columns
        ).dropna()
        histories["contributions"] = contributions_history

    return histories


def _process_path_or_database(path_or_database):
    """Make inputs for load_database out of path_or_database.

    Args:
        path_or_database (pathlib.Path, str or sqlalchemy.MetaData)

    Returns:
        dict: The keys are "path", "metadata" and "fast_logging"

    Examples:

    >>> from sqlalchemy import MetaData
    >>> database = MetaData()
    >>> _process_path_or_database(database)
    {'path': None, 'metadata': MetaData(), 'fast_logging': False}

    """
    res = {"path": None, "metadata": None, "fast_logging": False}
    if isinstance(path_or_database, MetaData):
        res["metadata"] = path_or_database
    elif isinstance(path_or_database, (Path, str)):
        res["path"] = Path(path_or_database).resolve()
        if not res["path"].exists():
            raise FileNotFoundError(f"No such database file: {res['path']}")
    else:
        raise ValueError(
            "path_or_database must be a path or sqlalchemy.MetaData object"
        )
    return res


def read_steps_table(path_or_database):
    """Load the start parameters DataFrame.

    Args:
        path_or_database (pathlib.Path, str or sqlalchemy.MetaData)

    Returns:
        params (pd.DataFrame): see :ref:`params`.

    """
    database = load_database(**_process_path_or_database(path_or_database))
    steps_table, _ = read_new_rows(
        database=database,
        table_name="steps",
        last_retrieved=0,
        return_type="list_of_dicts",
    )
    steps_df = pd.DataFrame(steps_table)

    return steps_df
