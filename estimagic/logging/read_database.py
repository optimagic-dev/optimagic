"""Functions to read from from database tables used to log an optimization."""
import io
import pickle
import traceback
import warnings

import pandas as pd
from sqlalchemy.sql.sqltypes import BLOB


def read_last_iterations(database, tables, n, return_type):
    """Read the last n iterations from all tables.

    If a table has less than n obervations, all observations are returned.

    Args:
        database (sqlalchemy.MetaData)
        tables (list): List of tables names.
        n (int): number of rows to retrieve
        return_type (str): one of "list", "pandas", "bokeh"
            - "list": A list of lists. The first sublist are the columns. The remaining
              sublists are retrieved rows.
            - "pandas": A dataframe.
            - "bokeh": A dictionary that can be used to stream to a ColumnDataSource.
              It has one key per column and the corresponding values are lists that
              contain the data of that column.

    Returns:
        result (dict or return_type):
            If ``tables`` has only one entry, return the last iterations of that table,
            converted to return_type. If ``tables`` has several entries, return a
            dictionary with one entry per table.

    """
    if isinstance(tables, (str, int)):
        tables = [tables]
    # sqlalchemy fails silently with many numpy integer types, e.g. np.int64.
    n = int(n)

    selects = []
    for table in tables:
        tab = database.tables[table]
        sel = tab.select().order_by(tab.c.iteration.desc()).limit(n)
        selects.append(sel)

    raw_results = _execute_select_statements(selects, database)
    ordered_results = [res[::-1] for res in raw_results]

    result = _process_selection_result(database, tables, ordered_results, return_type)
    return result


def read_new_iterations(database, tables, last_retrieved, return_type, limit=None):
    """Read all iterations after last_retrieved.

    Args:
        database (sqlalchemy.MetaData)
        tables (list): List of tables names.
        last_retrieved (int): The last iteration that was retrieved.
        return_type (str): one of "list", "pandas", "bokeh"
        limit (int): Only the first ``limit`` rows will be retrieved. Default None.

    Returns:
        result (dict or return_type):
            If ``tables`` has only one entry, return the last iterations of that table,
            converted to return_type. If ``tables`` has several entries, return a
            dictionary with one entry per table.
        int: The new last_retrieved value.

    """
    if isinstance(tables, (str, int)):
        tables = [tables]
    # sqlalchemy fails silently with many numpy integer types, e.g. np.int64.
    last_retrieved = int(last_retrieved)
    limit = int(limit)

    selects = []
    for table in tables:
        tab = database.tables[table]
        sel = tab.select().where(tab.c.iteration > last_retrieved).limit(limit)
        selects.append(sel)

    raw_results = _execute_select_statements(selects, database)
    if len(raw_results[0]) > 0:
        new_last = raw_results[0][-1][0]
    else:
        new_last = last_retrieved
    result = _process_selection_result(database, tables, raw_results, return_type)
    return result, new_last


def read_scalar_field(database, table):
    """Read the value of a table with one row and one column called "value".

    Args:
        database (sqlalchemy.MetaData)
        table (str): Name of the table.

    """
    sel = database.tables[table].select()
    res = _execute_select_statements(sel, database)[0][0][0]
    if isinstance(database.tables[table].c.value.type, BLOB):
        res = pickle.load(io.BytesIO(res))
    return res


def _execute_select_statements(statements, database):
    """Execute a list of select statements in one atomic transaction.

    If any statement fails, the transaction is rolled back, and a warning is issued.

    Args:
        statements (list or sqlalchemy statement): List of sqlalchemy select statements.
        database (sqlalchemy.MetaData): The bind argument must be set.


    Returns:
        result (list): List of selection results. A selection result is a list of
        tuples where each tuple is a selected row.

    """
    if not isinstance(statements, (list, tuple)):
        statements = [statements]

    results = []
    engine = database.bind
    conn = engine.connect()
    # acquire lock
    trans = conn.begin()
    try:
        for stat in statements:
            res = conn.execute(stat)
            results.append(list(res))
            res.close()
        # release lock
        trans.commit()
        conn.close()
    except (KeyboardInterrupt, SystemExit):
        trans.rollback()
        conn.close()
        raise
    except Exception:
        exception_info = traceback.format_exc()
        warnings.warn(
            "Unable to read from database. Try again later. The traceback was:\n\n"
            f"{exception_info}"
        )

        trans.rollback()
        conn.close()
        results = [[] for stat in statements]

    return results


def _transpose_nested_list(nested_list):
    """Transpose a list of lists."""
    return list(map(list, zip(*nested_list)))


def _process_selection_result(database, tables, raw_results, return_type):
    """Convert sqlalchemy selection results to desired return_type."""
    result = {}
    for table, raw_res in zip(tables, raw_results):
        columns = database.tables[table].columns.keys()
        if return_type == "list":
            res = [columns]
            for row in raw_res:
                res.append(list(row))
        elif return_type == "bokeh":
            res = dict(zip(columns, _transpose_nested_list(raw_res)))
            if res == {}:
                res = {col: [] for col in columns}
        elif return_type == "pandas":
            res = pd.DataFrame(data=raw_res, columns=columns).set_index("iteration")
        result[table] = res

    if len(tables) == 1:
        result = list(result.values())[0]
    return result
