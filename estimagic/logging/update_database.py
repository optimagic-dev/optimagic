"""Functions to update a DataBase in a thread safe way.

All write operation to a database in estimagic should be done via functions from this
module.

Public functions in this module should not require any knowledge of sqlalchemy or
sql in general. This is also the reason why _execute_write_statements is not a public
function.

"""
import pickle
import warnings
from datetime import datetime
from pathlib import Path

import sqlalchemy


def append_rows(metadata, tables, values):
    """As append values to tables in an atomic transaction.

    This ensures that the iteration counters stay correct in parallel optimizations.

    Args:
        metadata (sqlalchemy.MetaData):
        tables (list): list of strings
        values (list): list of dicts or pd.Series: values to append.

    """
    if isinstance(tables, str):
        tables = [tables]
        values = [values]

    values = [dict(val) for val in values]

    inserts = [
        metadata.tables[tab].insert().values(**val) for tab, val in zip(tables, values)
    ]

    _execute_write_statements(inserts, metadata)


def update_scalar_field(metadata, table, value):
    """Update the value of a table with one row and one column called "value".

    Args:
        metadata (sqlalchemy.MetaData)
        table (string): Name of a table.
        value: The new value of the scalar field.

    """
    value = {"value": value}
    upd = metadata.tables[table].update().values(**value)
    _execute_write_statements(upd, metadata)


def _execute_write_statements(statements, metadata):
    """Execute all statements in one atomic transaction.

    If any statement fails, the transaction is rolled back, and a warning is issued.

    If the statements contain inserts or updates, the values of that statement are
    pickled in the same directory as the database.

    Args:
        statements (list or sqlalchemy statement): List of sqlalchemy statements
            or single statement that entail a write operation. Examples are Insert,
            Update and Delete.
        metadata (sqlalchemy.sql.schema.MetaData): The bind argument must be set.

    """
    if not isinstance(statements, (list, tuple)):
        statements = [statements]

    engine = metadata.bind
    conn = engine.connect()
    # acquire lock
    trans = conn.begin()
    try:
        for stat in statements:
            conn.execute(stat)
        # release lock
        trans.commit()
        conn.close()
    except (KeyboardInterrupt, SystemExit):
        trans.rollback()
        conn.close()
        _handle_exception(statements, metadata)
        raise
    except Exception:
        trans.rollback()
        conn.close()
        _handle_exception(statements, metadata)


def _handle_exception(statements, metadata):
    directory = Path(str(metadata.bind.url)[10:])
    if not directory.is_dir():
        directory = Path(".")
    directory = directory.resolve()

    for stat in statements:
        if isinstance(stat, (sqlalchemy.sql.dml.Insert, sqlalchemy.sql.dml.Update)):
            values = stat.compile().params
            filename = f"{stat.table.name}_{datetime.now()}.pickle"
            with open(directory / filename, "wb") as p:
                pickle.dump(values, p)

    msg = "Unable to write to database. The data was saved in {} instead."
    warnings.warn(msg.format(directory))
