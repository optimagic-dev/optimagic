"""Functions to generate, load, write to and read from databases.

The functions here are meant for internal use in estimagic, e.g. for logging during
the optimization and reading from the database in the dashboard. They do not require
detailed knowledge of databases in general but some knowledge of the schema
(e.g. table names) of the database we use for logging.

Therefore, users who simply want to read the database should use the functions in
``read_log.py`` instead.

"""
import traceback
import warnings

import sqlalchemy as sql

from estimagic.exceptions import TableExistsError
from estimagic.logging.load_database import RobustPickler


def make_optimization_iteration_table(database, if_exists="extend"):
    """Generate a table for information that is generated with each function evaluation.

    Args:
        database (DataBase): Bound metadata object.
        if_exists (str): What to do if the table already exists. Can be "extend",
            "replace" or "raise".

    Returns:
        database (sqlalchemy.MetaData):Bound metadata object with added table.

    """
    table_name = "optimization_iterations"
    _handle_existing_table(database, "optimization_iterations", if_exists)

    columns = [
        sql.Column("rowid", sql.Integer, primary_key=True),
        sql.Column("params", sql.PickleType(pickler=RobustPickler)),
        sql.Column("internal_derivative", sql.PickleType(pickler=RobustPickler)),
        sql.Column("timestamp", sql.Float),
        sql.Column("exceptions", sql.String),
        sql.Column("valid", sql.Boolean),
        sql.Column("hash", sql.String),
        sql.Column("value", sql.Float),
        sql.Column("step", sql.Integer),
        sql.Column("criterion_eval", sql.PickleType(pickler=RobustPickler)),
    ]

    sql.Table(
        table_name,
        database.metadata,
        *columns,
        sqlite_autoincrement=True,
        extend_existing=True,
    )

    database.metadata.create_all(database.engine)


def _handle_existing_table(database, table_name, if_exists):
    assert if_exists in ["replace", "extend", "raise"]

    if table_name in database.metadata.tables:
        if if_exists == "replace":
            database.metadata.tables[table_name].drop(database.engine)
        elif if_exists == "raise":
            raise TableExistsError(f"The table {table_name} already exists.")


def make_steps_table(database, if_exists="extend"):
    table_name = "steps"
    _handle_existing_table(database, table_name, if_exists)
    columns = [
        sql.Column("rowid", sql.Integer, primary_key=True),
        sql.Column("type", sql.String),  # e.g. optimization
        sql.Column("status", sql.String),  # e.g. running
        sql.Column("n_iterations", sql.Integer),  # optional
        sql.Column(
            "name", sql.String
        ),  # e.g. "optimization-1", "exploration", not unique
    ]
    sql.Table(
        table_name,
        database.metadata,
        *columns,
        extend_existing=True,
        sqlite_autoincrement=True,
    )
    database.metadata.create_all(database.engine)


def make_optimization_problem_table(database, if_exists="extend"):
    table_name = "optimization_problem"
    _handle_existing_table(database, table_name, if_exists)

    columns = [
        sql.Column("rowid", sql.Integer, primary_key=True),
        sql.Column("direction", sql.String),
        sql.Column("params", sql.PickleType(pickler=RobustPickler)),
        sql.Column("algorithm", sql.PickleType(pickler=RobustPickler)),
        sql.Column("algo_options", sql.PickleType(pickler=RobustPickler)),
        sql.Column("numdiff_options", sql.PickleType(pickler=RobustPickler)),
        sql.Column("log_options", sql.PickleType(pickler=RobustPickler)),
        sql.Column("error_handling", sql.String),
        sql.Column("error_penalty", sql.PickleType(pickler=RobustPickler)),
        sql.Column("constraints", sql.PickleType(pickler=RobustPickler)),
        sql.Column("free_mask", sql.PickleType(pickler=RobustPickler)),
    ]

    sql.Table(
        table_name,
        database.metadata,
        *columns,
        extend_existing=True,
        sqlite_autoincrement=True,
    )

    database.metadata.create_all(database.engine)


# ======================================================================================


def update_row(data, rowid, table_name, database):
    table = database.metadata.tables[table_name]
    stmt = sql.update(table).where(table.c.rowid == rowid).values(**data)

    _execute_write_statement(stmt, database)


def append_row(data, table_name, database):
    """

    Args:
        data (dict): The keys correspond to columns in the database table.
        table_name (str): Name of the database table to which the row is added.
        database (DataBase): The database to which the row is added.

    """

    stmt = database.metadata.tables[table_name].insert().values(**data)

    _execute_write_statement(stmt, database)


def _execute_write_statement(statement, database):
    try:
        # this will automatically roll back the transaction if any exception is raised
        # and then raise the exception
        with database.engine.begin() as connection:
            connection.execute(statement)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        exception_info = traceback.format_exc()
        warnings.warn(
            f"Unable to write to database. The traceback was:\n\n{exception_info}"
        )


def read_new_rows(
    database,
    table_name,
    last_retrieved,
    return_type,
    limit=None,
    stride=1,
    step=None,
):
    """Read all iterations after last_retrieved up to a limit.

    Args:
        database (DataBase)
        table_name (str): name of the table to retrieve.
        last_retrieved (int): The last iteration that was retrieved.
        return_type (str): either "list_of_dicts" or "dict_of_lists".
        path (str or pathlib.Path): location of the database file. If the file does
            not exist, it will be created. Using a path is much slower than a
            MetaData object and we advise to only use it as a fallback.
        fast_logging (bool)
        limit (int): maximum number of rows to extract from the table.
        stride (int): Only return every n-th row. Default is every row (stride=1).
        step (int): Only return iterations that belong to step.

    Returns:
        result (return_type): up to limit rows after last_retrieved of the
            `table_name` table as `return_type`.
        int: The new last_retrieved value.

    """
    last_retrieved = int(last_retrieved)
    limit = int(limit) if limit is not None else limit

    table = database.metadata.tables[table_name]
    stmt = table.select().where(table.c.rowid > last_retrieved).limit(limit)
    conditions = [table.c.rowid > last_retrieved]

    if stride != 1:
        conditions.append(table.c.rowid % stride == 0)

    if step is not None:
        conditions.append(table.c.step == int(step))

    stmt = table.select().where(sql.and_(*conditions)).limit(limit)

    data = _execute_read_statement(database, table_name, stmt, return_type)

    if return_type == "list_of_dicts":
        new_last = data[-1]["rowid"] if data else last_retrieved
    else:
        new_last = data["rowid"][-1] if data["rowid"] else last_retrieved

    return data, new_last


def read_last_rows(
    database,
    table_name,
    n_rows,
    return_type,
    stride=1,
    step=None,
):
    """Read the last n_rows rows from a table.

    If a table has less than n_rows rows, the whole table is returned.

    Args:
        database (DataBase)
        table_name (str): name of the table to retrieve.
        n_rows (int): number of rows to retrieve.
        return_type (str): either "list_of_dicts" or "dict_of_lists".
        stride (int): Only return every n-th row. Default is every row (stride=1).
        step (int): Only return rows that belong to step.

    Returns:
        result (return_type): the last rows of the `table_name` table as `return_type`.

    """
    n_rows = int(n_rows)

    table = database.metadata.tables[table_name]

    conditions = []

    if stride != 1:
        conditions.append(table.c.rowid % stride == 0)

    if step is not None:
        conditions.append(table.c.step == int(step))

    if conditions:
        stmt = (
            table.select()
            .order_by(table.c.rowid.desc())
            .where(sql.and_(*conditions))
            .limit(n_rows)
        )
    else:
        stmt = table.select().order_by(table.c.rowid.desc()).limit(n_rows)

    reversed_ = _execute_read_statement(database, table_name, stmt, return_type)
    if return_type == "list_of_dicts":
        out = reversed_[::-1]
    else:
        out = {key: val[::-1] for key, val in reversed_.items()}

    return out


def read_specific_row(database, table_name, rowid, return_type):
    """Read a specific row from a table.

    Args:
        database (sqlalchemy.MetaData)
        table_name (str): name of the table to retrieve.
        n_rows (int): number of rows to retrieve.
        return_type (str): either "list_of_dicts" or "dict_of_lists".

    Returns:
        dict or list: The requested row from the database.

    """
    rowid = int(rowid)
    table = database.metadata.tables[table_name]
    stmt = table.select().where(table.c.rowid == rowid)
    data = _execute_read_statement(database, table_name, stmt, return_type)
    return data


def read_table(database, table_name, return_type):
    table = database.metadata.tables[table_name]
    stmt = table.select()
    data = _execute_read_statement(database, table_name, stmt, return_type)
    return data


def _execute_read_statement(database, table_name, statement, return_type):
    try:
        with database.engine.begin() as connection:
            raw_result = list(connection.execute(statement))
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        exception_info = traceback.format_exc()
        warnings.warn(
            "Unable to read {table_name} from database. Try again later. The traceback "
            f"was: \n\n{exception_info}"
        )
        # if we only want to warn we must provide a raw_result to be processed below.
        raw_result = []

    columns = database.metadata.tables[table_name].columns.keys()

    if return_type == "list_of_dicts":
        result = [dict(zip(columns, row)) for row in raw_result]

    elif return_type == "dict_of_lists":
        raw_result = transpose_nested_list(raw_result)
        result = dict(zip(columns, raw_result))
        if result == {}:
            result = {col: [] for col in columns}
    else:
        raise NotImplementedError(
            "The return_type must be 'list_of_dicts' or 'dict_of_lists', "
            f"not {return_type}."
        )

    return result


# ======================================================================================


def transpose_nested_list(nested_list):
    """Transpose a list of lists.

    Args:
        nested_list (list): Nested list where all sublists have the same length.

    Returns:
        list

    Examples:
        >>> transpose_nested_list([[1, 2], [3, 4]])
        [[1, 3], [2, 4]]

    """
    return list(map(list, zip(*nested_list)))


def list_of_dicts_to_dict_of_lists(list_of_dicts):
    """Convert a list of dicts to a dict of lists.

    Args:
        list_of_dicts (list): List of dictionaries. All dictionaries have the same keys.

    Returns:
        dict

    Examples:
        >>> list_of_dicts_to_dict_of_lists([{"a": 1, "b": 2}, {"a": 3, "b": 4}])
        {'a': [1, 3], 'b': [2, 4]}

    """
    return {k: [dic[k] for dic in list_of_dicts] for k in list_of_dicts[0]}


def dict_of_lists_to_list_of_dicts(dict_of_lists):
    """Convert a dict of lists to a list of dicts.

    Args:
        dict_of_lists (dict): Dictionary of lists where all lists have the same length.

    Returns:
        list

    Examples:

        >>> dict_of_lists_to_list_of_dicts({'a': [1, 3], 'b': [2, 4]})
        [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]

    """
    return [dict(zip(dict_of_lists, t)) for t in zip(*dict_of_lists.values())]
