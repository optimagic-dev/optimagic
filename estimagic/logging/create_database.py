"""Functions to create new databases or load existing ones.

Note: Most functions in this module change their arguments in place since this is the
recommended way of doing things in sqlalchemy and makes sense for database code.

"""
from pathlib import Path

from sqlalchemy import Column
from sqlalchemy import create_engine
from sqlalchemy import event
from sqlalchemy import Float
from sqlalchemy import Integer
from sqlalchemy import MetaData
from sqlalchemy import PickleType
from sqlalchemy import String
from sqlalchemy import Table
from sqlalchemy.dialects.sqlite import DATETIME

from estimagic.logging.update_database import append_rows


def load_database(path, replace=False):
    """Return database metadata object for the database stored in ``path``.

    This is the default way of loading a database for read-only purposes in estimagic.

    Args:
        path (str or pathlib.Path): location of the database file. If the file does
            not exist, it will be created.
        replace (bool): If true and the database exists, it will be overwritten.
            Otherwise, data will be appended.

    Returns:
        metadata (sqlalchemy.MetaData). The engine that connects to the database can be
        accessed via ``metadata.bind``.

    """
    if isinstance(path, str):
        path = Path(path)

    if path.exists() and replace:
        path.unlink()

    engine = create_engine(f"sqlite:///{path}")
    _make_engine_thread_safe(engine)
    metadata = MetaData()
    metadata.bind = engine
    metadata.reflect()
    return metadata


def _make_engine_thread_safe(engine):
    """Make the engine even more thread safe than by default.

    The code is taken from the documentation: https://tinyurl.com/u9xea5z

    The main purpose is to emit the begin statement of any connection
    as late as possible in order to keep the time in which the database
    is locked as short as possible.

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


def prepare_database(
    path,
    params,
    db_options=None,
    optimization_status="scheduled",
    gradient_status=0,
    replace=False,
):
    """Return database metadata object with all relevant tables.

    This should always be used to create entirely new databases or to create the
    tables needed during optimization in an existing database.

    The resulting database has the following tables:
    - params_history: the complete history of parameters from the optimization. The
      first column is "iteration", the remaining columns are parameter names taken
      from params["name"].
    - gradient_history: the complete history of gradient evaluations from the
      optimization. Same columns as params_history.
    - criterion_history: the complete history of criterion values from the optimization.
      The first column is "iteration", the second column is "value".
    - time_stamps: timestamps from the end of each criterion evaluation. Same columss as
      criterion_history.
    - convergence_history: the complete history of convergence criteria from the
      optimization. The first column is iteration, the other columns are "ftol", "gtol"
      and "xtol".
    - start_params: copy of user provided params.
    - optimization_status: table with one row and one column called "value" which takes
      the values "scheduled", "running", "success" or "failure". The initial value is
      "scheduled".
    - gradient_status: table with one row and one column called "value" which takes
      can be any float between 0 and 1 and indicates the progress of the gradient
      calculation. Initialized to 0.
    - db_options: table with one row and one column called "value". It contains
      a dictionary with the dashboard options. Internally this is a PickleType, so the
      dictionary must be pickle serializable.

    If the function is called with an existing database, only the tables that do not
    exist already are created. We assume that the existing tables have the correct
    columns but do not check it.

    start_params, db_options, optimization_status and gradient_status will be
    initialized (or overwritten if they already existed) with the corresponding
    argument. All other tables remain unchanged.

    Args:
        path (str or pathlib.Path): location of the database file. If the file does
            not exist, it will be created.
        params (pd.DataFrame): see :ref:`params`
        db_options (dict): Dashboard options.
        optimization_status (str): One of "scheduled", "running", "success", "failure".
        gradient_status (float): Progress of gradient calculation between 0 and 1.
        replace (bool): If true and the database exists, it will be overwritten.
            Otherwise, data will be appended.

    Returns:
        metadata (sqlalchemy.sql.schema.MetaData). The engine that connects
        to the database can be accessed via metadata.bind.

    """
    db_options = {} if db_options is None else db_options
    gradient_status = float(gradient_status)
    metadata = load_database(path, replace)

    _define_params_history_table(metadata, params)
    _define_gradient_history_table(metadata, params)
    _define_criterion_history_table(metadata)
    _define_time_stamps_table(metadata)
    _define_convergence_history_table(metadata)
    _define_start_params_table(metadata)
    _define_optimization_status_table(metadata)
    _define_gradient_status_table(metadata)
    _define_db_options_table(metadata)
    engine = metadata.bind
    metadata.create_all(engine)

    append_rows(metadata, "start_params", {"value": params})
    append_rows(metadata, "optimization_status", {"value": optimization_status})
    append_rows(metadata, "gradient_status", {"value": gradient_status})
    append_rows(metadata, "db_options", {"value": db_options})

    return metadata


def _define_params_history_table(metadata, params):
    names = params["name"].tolist()
    cols = [Column(name, Float) for name in names]
    parvals = Table(
        "params_history",
        metadata,
        Column("iteration", Integer, primary_key=True),
        *cols,
        sqlite_autoincrement=True,
    )
    return parvals


def _define_gradient_history_table(metadata, params):
    names = params["name"].tolist()
    cols = [Column(name, Float) for name in names]
    gradvals = Table(
        "gradient_history",
        metadata,
        Column("iteration", Integer, primary_key=True),
        *cols,
        sqlite_autoincrement=True,
    )
    return gradvals


def _define_criterion_history_table(metadata):
    critvals = Table(
        "criterion_history",
        metadata,
        Column("iteration", Integer, primary_key=True),
        Column("value", Float),
        sqlite_autoincrement=True,
    )
    return critvals


def _define_time_stamps_table(metadata):
    tstamps = Table(
        "timestamps",
        metadata,
        Column("iteration", Integer, primary_key=True),
        Column("value", DATETIME),
        sqlite_autoincrement=True,
    )
    return tstamps


def _define_convergence_history_table(metadata):
    names = ["ftol", "gtol", "xtol"]
    cols = [Column(name, Float) for name in names]
    term = Table(
        "convergence_history",
        metadata,
        Column("iteration", Integer, primary_key=True),
        *cols,
        sqlite_autoincrement=True,
    )
    return term


def _define_start_params_table(metadata):
    params_table = Table("start_params", metadata, Column("value", PickleType))
    return params_table


def _define_optimization_status_table(metadata):
    optstat = Table("optimization_status", metadata, Column("value", String),)
    return optstat


def _define_gradient_status_table(metadata):
    gradstat = Table("gradient_status", metadata, Column("value", Float),)
    return gradstat


def _define_db_options_table(metadata):
    db_options = Table("db_options", metadata, Column("value", PickleType),)
    return db_options
