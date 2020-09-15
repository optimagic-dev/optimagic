"""Functions to read data from the database used for logging.


The functions in the module are meant for end users of estimagic. They do not require
any knowledge of databases. The downside is that they are slower than what one could
achieve when directly working with the more low-level database utilities.

Thus, they should not be used internally (e.g. in the dashboard) but only to read the
log interactively.

"""
from estimagic.logging.database_utilities import load_database
from estimagic.logging.database_utilities import read_last_rows
from estimagic.logging.database_utilities import read_specific_row


def read_optimization_iteration(path, iteration, include_internals=False):
    """Get information about an optimization iteration.

    Args:
        path (str or pathlib.Path): Path to the sqlite database file used for logging.
            Typically, those have the file extension ``.db``.
        iteration (int): The index of the iteration that should be retrieved. The row_id
            behaves as Python list indices, i.e. ``0`` identifies the first iteration,
            ``-1`` the last one, etc.
        include_internals (bool): Whether internally used quantities like the
            internal parameter vector and the corresponding derivative etc. are included
            in the result. Default False. This should only be used by advanced users.

    Returns:
        dict: The logged information corresponding to the iteration. The keys correspond
            to database columns.

    Raises:
        KeyError: if the iteration is out of bounds.

    """
    database = load_database(path=path, fast_logging=False)
    optimization_problem = read_last_rows(
        database=database,
        table_name="optimization_problem",
        n_rows=1,
        return_type="list_of_dicts",
    )
    start_params = optimization_problem[0]["params"]

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
    params["value"] = data.pop("external_params")
    data["params"] = params

    to_remove = ["distance_origin", "distance_ones"]
    if not include_internals:
        to_remove += ["internal_params", "internal_derivative"]
    for key in to_remove:
        if key in data:
            del data[key]

    return data


def load_start_params(database=None, path=None):
    """Load the start parameters DataFrame.

    Args:
        database (sqlalchemy.MetaData)
        path (str or pathlib.Path): Path to the sqlite database file used for
            logging. This is slower than providing the database directly.

    Returns:
        params (pd.DataFrame): see :ref:`params`.

    """
    database = load_database(metadata=database, path=path, fast_logging=False)
    optimization_problem = read_last_rows(
        database=database,
        table_name="optimization_problem",
        n_rows=1,
        return_type="dict_of_lists",
    )
    start_params = optimization_problem["params"][0]
    return start_params
