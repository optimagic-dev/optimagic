"""Functions to generate, load, write to and read from databases.

The functions here are meant for internal use in estimagic, e.g. for logging during
the optimization and reading from the database in the dashboard. They do not require
detailed knowledge of databases in general but some knowledge of the schema
(e.g. table names) of the database we use for logging.

Therefore, users who simply want to read the database should use the functions in
``read_log.py`` instead.

"""
import io
import traceback
import warnings

import cloudpickle
import pandas as pd
import sqlalchemy as sql

from estimagic.exceptions import TableExistsError, get_traceback


class DataBase:
    """Class containing everything to work with a logging database.

    Importantly, the class is pickle-serializable which is important to share it across
    multiple processes.

    """

    def __init__(self, metadata, path, fast_logging, engine=None):
        self.metadata = metadata
        self.path = path
        self.fast_logging = fast_logging
        if isinstance(engine, sql.Engine):
            self.engine = engine
        else:
            self.engine = _create_engine(path, fast_logging)

    def __reduce__(self):
        return (DataBase, (self.metadata, self.path, self.fast_logging))


def load_database(path_or_database, fast_logging=False):
    """Load or create a database from a path and configure it for our needs.

    This is the only acceptable way of loading or creating a database in estimagic!

    Args:
        path (str or pathlib.Path): Path to the database.
        fast_logging (bool): If True, use unsafe optimizations to speed up the logging.
            If False, only use ultra safe optimizations.

    Returns:
        database (Database): Object containing everything to work with the
            database.

    """
    if isinstance(path_or_database, DataBase):
        out = path_or_database
    else:
        engine = _create_engine(path_or_database, fast_logging)
        metadata = sql.MetaData()
        _configure_reflect()
        metadata.reflect(engine)

        out = DataBase(
            metadata=metadata,
            path=path_or_database,
            fast_logging=fast_logging,
            engine=engine,
        )
    return out


def _create_engine(path, fast_logging):
    engine = sql.create_engine(f"sqlite:///{path}")
    _configure_engine(engine, fast_logging)
    return engine


def _configure_engine(engine, fast_logging):
    """Configure the sqlite engine.

    The two functions that configure the emission of the begin statement are taken from
    the sqlalchemy documentation the documentation: https://tinyurl.com/u9xea5z and are
    the recommended way of working around a bug in the pysqlite driver.

    The other function speeds up the write process. If fast_logging is False, it does so
    using only completely safe optimizations. Of fast_logging is True, it also uses
    unsafe optimizations.

    """

    @sql.event.listens_for(engine, "connect")
    def do_connect(dbapi_connection, connection_record):  # noqa: ARG001
        # disable pysqlite's emitting of the BEGIN statement entirely.
        # also stops it from emitting COMMIT before absolutely necessary.
        dbapi_connection.isolation_level = None

    @sql.event.listens_for(engine, "begin")
    def do_begin(conn):
        # emit our own BEGIN
        conn.exec_driver_sql("BEGIN DEFERRED")

    @sql.event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):  # noqa: ARG001
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode = WAL")
        if fast_logging:
            cursor.execute("PRAGMA synchronous = OFF")
        else:
            cursor.execute("PRAGMA synchronous = NORMAL")
        cursor.close()


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


def _configure_reflect():
    """Mark all BLOB dtypes as PickleType with our custom pickle reader.

    Code ist taken from the documentation: https://tinyurl.com/y7q287jr

    """

    @sql.event.listens_for(sql.Table, "column_reflect")
    def _setup_pickletype(inspector, table, column_info):  # noqa: ARG001
        if isinstance(column_info["type"], sql.BLOB):
            column_info["type"] = sql.PickleType(pickler=RobustPickler)


class RobustPickler:
    @staticmethod
    def loads(
        data,
        fix_imports=True,  # noqa: ARG004
        encoding="ASCII",  # noqa: ARG004
        errors="strict",  # noqa: ARG004
        buffers=None,  # noqa: ARG004
    ):
        """Robust pickle loading.

        We first try to unpickle the object with pd.read_pickle. This makes no
        difference for non-pandas objects but makes the de-serialization
        of pandas objects more robust across pandas versions. If that fails, we use
        cloudpickle. If that fails, we return None but do not raise an error.

        See: https://github.com/pandas-dev/pandas/issues/16474

        """
        try:
            res = pd.read_pickle(io.BytesIO(data), compression=None)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            try:
                res = cloudpickle.loads(data)
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception:
                res = None
                tb = get_traceback()
                warnings.warn(
                    f"Unable to read PickleType column from database:\n{tb}\n "
                    "The entry was replaced by None."
                )

        return res

    @staticmethod
    def dumps(
        obj, protocol=None, *, fix_imports=True, buffer_callback=None  # noqa: ARG004
    ):
        return cloudpickle.dumps(obj, protocol=protocol)


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
