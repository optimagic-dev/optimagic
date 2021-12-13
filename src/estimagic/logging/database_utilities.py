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
from pathlib import Path

import cloudpickle
import pandas as pd
from estimagic.exceptions import get_traceback
from estimagic.exceptions import TableExistsError
from sqlalchemy import and_
from sqlalchemy import BLOB
from sqlalchemy import Boolean
from sqlalchemy import Column
from sqlalchemy import create_engine
from sqlalchemy import event
from sqlalchemy import Float
from sqlalchemy import Integer
from sqlalchemy import MetaData
from sqlalchemy import PickleType
from sqlalchemy import String
from sqlalchemy import Table
from sqlalchemy import update
from sqlalchemy.dialects.sqlite import DATETIME


def load_database(metadata=None, path=None, fast_logging=False):
    """Return a bound sqlalchemy.MetaData object for the database stored in ``path``.

    This is the only acceptable way of creating or loading databases in estimagic!

    If metadata is a bound MetaData object, it is just returned. If metadata is given
    but not bound, we bind it to an engine that connects to the database stored under
    ``path``. If only the path is provided, we generate an appropriate MetaData object
    and bind it to the database.

    For speed reasons we do not make any checks that MetaData is compatible with the
    database stored under path.

    Args:
        metadata (sqlalchemy.MetaData): MetaData object that might or might not be
            bound to the database under path. In any case it needs to be compatible
            with the database stored under ``path``. For speed reasons, this is not
            checked.
        path (str or pathlib.Path): location of the database file. If the file does
            not exist, it will be created.

    Returns:
        metadata (sqlalchemy.MetaData). MetaData object that is bound to the database
        under ``path``.

    """
    path = Path(path) if isinstance(path, str) else path

    if isinstance(metadata, MetaData):
        if metadata.bind is None:
            assert (
                path is not None
            ), "If metadata is not bound, you need to provide a path."
            engine = create_engine(f"sqlite:///{path}")
            _configure_engine(engine, fast_logging)
            metadata.bind = engine
    elif metadata is None:
        assert path is not None, "If metadata is None you need to provide a path."
        path_existed = path.exists()
        engine = create_engine(f"sqlite:///{path}")
        _configure_engine(engine, fast_logging)
        metadata = MetaData()
        metadata.bind = engine
        if path_existed:
            _configure_reflect()
            metadata.reflect()
    else:
        raise ValueError("metadata must be sqlalchemy.MetaData or None.")

    return metadata


def make_optimization_iteration_table(database, first_eval, if_exists="extend"):
    """Generate a table for information that is generated with each function evaluation.

    Args:
        database (sqlalchemy.MetaData): Bound metadata object.
        first_eval (dict): The inputs and output of the first criterion evaluation. Has
            the entries "internal_params", "external_params" and "output".

    Returns:
        database (sqlalchemy.MetaData):Bound metadata object with added table.

    """
    table_name = "optimization_iterations"
    _handle_existing_table(database, "optimization_iterations", if_exists)

    columns = [
        Column("rowid", Integer, primary_key=True),
        Column("params", PickleType(pickler=RobustPickler)),
        Column("internal_derivative", PickleType(pickler=RobustPickler)),
        Column("timestamp", DATETIME),
        Column("exceptions", String),
        Column("valid", Boolean),
        Column("hash", String),
        Column("value", Float),
        Column("step", Integer),
    ]

    if isinstance(first_eval["output"], dict):
        extra_columns = {x for x in first_eval["output"] if x != "value"}
        if "root_contributions" in extra_columns:
            extra_columns |= {"contributions"}
        columns += [
            Column(key, PickleType(pickler=RobustPickler)) for key in extra_columns
        ]

    Table(
        table_name, database, *columns, sqlite_autoincrement=True, extend_existing=True
    )

    database.create_all(database.bind)


def make_steps_table(database, if_exists="extend"):
    table_name = "steps"
    _handle_existing_table(database, table_name, if_exists)
    columns = [
        Column("rowid", Integer, primary_key=True),
        Column("type", String),  # e.g. optimization
        Column("status", String),  # e.g. running
        Column("n_iterations", Integer),  # optional
        Column("name", String),  # e.g. "optimization-1", "exploration", not unique
    ]
    Table(
        table_name, database, *columns, extend_existing=True, sqlite_autoincrement=True
    )
    database.create_all(database.bind)


def make_optimization_problem_table(database, if_exists="extend"):
    table_name = "optimization_problem"
    _handle_existing_table(database, table_name, if_exists)

    columns = [
        Column("rowid", Integer, primary_key=True),
        Column("direction", String),
        Column("params", PickleType(pickler=RobustPickler)),
        Column("algorithm", PickleType(pickler=RobustPickler)),
        Column("algo_options", PickleType(pickler=RobustPickler)),
        Column("numdiff_options", PickleType(pickler=RobustPickler)),
        Column("log_options", PickleType(pickler=RobustPickler)),
        Column("error_handling", String),
        Column("error_penalty", PickleType(pickler=RobustPickler)),
        Column("cache_size", Integer),
        Column("constraints", PickleType(pickler=RobustPickler)),
    ]

    Table(
        table_name, database, *columns, extend_existing=True, sqlite_autoincrement=True
    )

    database.create_all(database.bind)


def _handle_existing_table(database, table_name, if_exists):
    assert if_exists in ["replace", "extend", "raise"]

    if table_name in database.tables:
        if if_exists == "replace":
            database.tables[table_name].drop(database.bind)
        elif if_exists == "raise":
            raise TableExistsError(f"The table {table_name} already exists.")


def update_row(data, rowid, table_name, database, path, fast_logging):
    database = load_database(database, path, fast_logging)

    table = database.tables[table_name]
    stmt = update(table).where(table.c.rowid == rowid).values(**data)

    _execute_write_statement(stmt, database, path, table_name, data)


def append_row(data, table_name, database, path, fast_logging):
    """

    Args:
        data (dict): The keys correspond to columns in the database table.
        table_name (str): Name of the database table to which the row is added.
        database (sqlalchemy.MetaData): Sqlachlemy metadata object of the database.
        path (str or pathlib.Path): Path to the database file. Using a path is much
            slower than a MetaData object and we advise to only use it as a fallback.
        fast_logging (bool)

    """
    # this is necessary because database.bind gets lost when the database is pickled.
    # it has no cost when database.bind is set.
    database = load_database(database, path, fast_logging)

    stmt = database.tables[table_name].insert().values(**data)

    _execute_write_statement(stmt, database, path, table_name, data)


def _execute_write_statement(statement, database, path, table_name, data):
    try:
        # this will automatically roll back the transaction if any exception is raised
        # and then raise the exception
        with database.bind.begin() as connection:
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
    path=None,
    fast_logging=False,
    limit=None,
    stride=1,
    step=None,
):
    """Read all iterations after last_retrieved up to a limit.

    Args:
        database (sqlalchemy.MetaData)
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
    database = load_database(database, path, fast_logging)
    last_retrieved = int(last_retrieved)
    limit = int(limit) if limit is not None else limit

    table = database.tables[table_name]
    stmt = table.select().where(table.c.rowid > last_retrieved).limit(limit)
    conditions = [table.c.rowid > last_retrieved]

    if stride != 1:
        conditions.append(table.c.rowid % stride == 0)

    if step is not None:
        conditions.append(table.c.step == int(step))

    stmt = table.select().where(and_(*conditions)).limit(limit)

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
    path=None,
    fast_logging=False,
    stride=1,
    step=None,
):
    """Read the last n_rows rows from a table.

    If a table has less than n_rows rows, the whole table is returned.

    Args:
        database (sqlalchemy.MetaData)
        table_name (str): name of the table to retrieve.
        n_rows (int): number of rows to retrieve.
        return_type (str): either "list_of_dicts" or "dict_of_lists".
        path (str or pathlib.Path): location of the database file. If the file does
            not exist, it will be created. Using a path is much slower than a
            MetaData object and we advise to only use it as a fallback.
        fast_logging (bool)
        stride (int): Only return every n-th row. Default is every row (stride=1).
        step (int): Only return rows that belong to step.

    Returns:
        result (return_type): the last rows of the `table_name` table as `return_type`.

    """
    database = load_database(database, path, fast_logging)
    n_rows = int(n_rows)

    table = database.tables[table_name]

    conditions = []

    if stride != 1:
        conditions.append(table.c.rowid % stride == 0)

    if step is not None:
        conditions.append(table.c.step == int(step))

    if conditions:
        stmt = (
            table.select()
            .order_by(table.c.rowid.desc())
            .where(and_(*conditions))
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


def read_specific_row(
    database, table_name, rowid, return_type, path=None, fast_logging=False
):
    """Read a specific row from a table.

    Args:
        database (sqlalchemy.MetaData)
        table_name (str): name of the table to retrieve.
        n_rows (int): number of rows to retrieve.
        return_type (str): either "list_of_dicts" or "dict_of_lists".
        path (str or pathlib.Path): location of the database file.
            Using a path is much slower than a MetaData object and we
            advise to only use it as a fallback.
        fast_logging (bool)

    Returns:
        dict or list: The requested row from the database.

    """
    database = load_database(database, path, fast_logging)
    rowid = int(rowid)
    table = database.tables[table_name]
    stmt = table.select().where(table.c.rowid == rowid)
    data = _execute_read_statement(database, table_name, stmt, return_type)
    return data


def read_table(database, table_name, return_type, path=None, fast_logging=False):
    database = load_database(database, path, fast_logging)
    table = database.tables[table_name]
    stmt = table.select()
    data = _execute_read_statement(database, table_name, stmt, return_type)
    return data


def _execute_read_statement(database, table_name, statement, return_type):

    try:
        with database.bind.begin() as connection:
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

    columns = database.tables[table_name].columns.keys()

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
            + f"not {return_type}."
        )

    return result


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


def _configure_engine(engine, fast_logging):
    """Configure the sqlite engine.

    The two functions that configure the emission of the begin statement are taken from
    the sqlalchemy documentation the documentation: https://tinyurl.com/u9xea5z and are
    the recommended way of working around a bug in the pysqlite driver.

    The other function speeds up the write process. If fast_logging is False, it does so
    using only completely safe optimizations. Of fast_logging is True, it also uses
    unsafe optimizations.

    """

    @event.listens_for(engine, "connect")
    def do_connect(dbapi_connection, connection_record):
        # disable pysqlite's emitting of the BEGIN statement entirely.
        # also stops it from emitting COMMIT before absolutely necessary.
        dbapi_connection.isolation_level = None

    @event.listens_for(engine, "begin")
    def do_begin(conn):
        # emit our own BEGIN
        conn.execute("BEGIN DEFERRED")

    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode = WAL")
        if fast_logging:
            cursor.execute("PRAGMA synchronous = OFF")
        else:
            cursor.execute("PRAGMA synchronous = NORMAL")
        cursor.close()


def _configure_reflect():
    """Mark all BLOB dtypes as PickleType with our custom pickle reader.

    Code ist taken from the documentation: https://tinyurl.com/y7q287jr

    """

    @event.listens_for(Table, "column_reflect")
    def _setup_pickletype(inspector, table, column_info):
        if isinstance(column_info["type"], BLOB):
            column_info["type"] = PickleType(pickler=RobustPickler)


class RobustPickler:
    @staticmethod
    def loads(data, fix_imports=True, encoding="ASCII", errors="strict", buffers=None):
        """Robust pickle loading

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
    def dumps(obj, protocol=None, *, fix_imports=True, buffer_callback=None):
        return cloudpickle.dumps(obj, protocol=protocol)
