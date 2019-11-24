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

import pandas as pd
import sqlalchemy


def append_rows(database, tables, rows):
    """Append rows to one or several tables in one transaction.

    Using just one transaction ensures that the iteration counters stay correct in
    parallel optimizations. It is also faster than using several transactions.

    If anything fails, the complete operation is rolled back and the data is stored in
    pickle files instead.

    Args:
        database (sqlalchemy.MetaData):
        tables (str or list): A table name or list of table names.
        rows (dict, pd.Series or list): The data to append.

    """
    if isinstance(tables, str):
        tables = [tables]
    if isinstance(rows, (dict, pd.Series)):
        rows = [rows]

    assert len(tables) == len(rows), "There must be one value per table."

    rows = [dict(val) for val in rows]

    inserts = [
        database.tables[tab].insert().values(**row) for tab, row in zip(tables, rows)
    ]

    _execute_write_statements(inserts, database)


def update_scalar_field(database, table, value):
    """Update the value of a table with one row and one column called "value".

    Args:
        database (sqlalchemy.MetaData)
        table (string): Name of the table to be updated.
        value: The new value of the table.

    """
    value = {"value": value}
    upd = database.tables[table].update().values(**value)
    _execute_write_statements(upd, database)


def _execute_write_statements(statements, database):
    """Execute all statements in one atomic transaction.

    If any statement fails, the transaction is rolled back, and a warning is issued.

    If the statements contain inserts or updates, the values of that statement are
    pickled in the same directory as the database.

    Args:
        statements (list or sqlalchemy statement): List of sqlalchemy statements
            or single statement that entail a write operation. Examples are Insert,
            Update and Delete.
        database (sqlalchemy.sql.schema.MetaData): The bind argument must be set.

    """
    if not isinstance(statements, (list, tuple)):
        statements = [statements]

    engine = database.bind
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
        _handle_exception(statements, database)
        raise
    except Exception:
        trans.rollback()
        conn.close()
        _handle_exception(statements, database)


def _handle_exception(statements, database):
    directory = Path(str(database.bind.url)[10:])
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
