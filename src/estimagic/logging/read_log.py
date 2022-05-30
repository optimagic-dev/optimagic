"""Functions to read data from the database used for logging.

The functions in the module are meant for end users of estimagic.
They do not require any knowledge of databases.

When using them internally (e.g. in the dashboard), make sure to supply a database to
path_or_database. Otherwise, the functions may be very slow.

"""
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from estimagic.logging.database_utilities import load_database
from estimagic.logging.database_utilities import read_last_rows
from estimagic.logging.database_utilities import read_new_rows
from estimagic.logging.database_utilities import read_specific_row
from estimagic.parameters.tree_registry import get_registry
from pybaum import tree_flatten
from pybaum import tree_unflatten
from sqlalchemy import MetaData


def read_start_params(path_or_database):
    """Load the start parameters DataFrame.

    Args:
        path_or_database (pathlib.Path, str or sqlalchemy.MetaData)

    Returns:
        params (pd.DataFrame): see :ref:`params`.

    """
    database = _load_database(path_or_database)
    optimization_problem = read_last_rows(
        database=database,
        table_name="optimization_problem",
        n_rows=1,
        return_type="dict_of_lists",
    )
    start_params = optimization_problem["params"][0]
    return start_params


def _load_database(path_or_database):
    """Get an sqlalchemy.MetaDate object from path or database."""

    res = {"path": None, "metadata": None, "fast_logging": False}
    if isinstance(path_or_database, MetaData):
        res = path_or_database
    elif isinstance(path_or_database, (Path, str)):
        path = Path(path_or_database)
        if not path.exists():
            raise FileNotFoundError(f"No such database file: {path}")
        res = load_database(path=path)
    else:
        raise ValueError(
            "path_or_database must be a path or sqlalchemy.MetaData object"
        )
    return res


def read_steps_table(path_or_database):
    """Load the steps table.

    Args:
        path_or_database (pathlib.Path, str or sqlalchemy.MetaData)

    Returns:
        steps_df (pandas.DataFrame)

    """
    database = _load_database(path_or_database)
    steps_table, _ = read_new_rows(
        database=database,
        table_name="steps",
        last_retrieved=0,
        return_type="list_of_dicts",
    )
    steps_df = pd.DataFrame(steps_table)

    return steps_df


def read_optimization_problem_table(path_or_database):
    """Load the start parameters DataFrame.

    Args:
        path_or_database (pathlib.Path, str or sqlalchemy.MetaData)

    Returns:
        params (pd.DataFrame): see :ref:`params`.

    """
    database = _load_database(path_or_database)
    steps_table, _ = read_new_rows(
        database=database,
        table_name="optimization_problem",
        last_retrieved=0,
        return_type="list_of_dicts",
    )
    steps_df = pd.DataFrame(steps_table)

    return steps_df


@dataclass
class OptimizeLogReader:
    """Read information about an optimization from a sqlite database."""

    path: Union[str, Path]

    def __post_init__(self):
        _database = _load_database(self.path)
        _start_params = read_start_params(_database)
        _registry = get_registry(extended=True)
        _, _treedef = tree_flatten(_start_params, registry=_registry)
        self._database = _database
        self._registry = _registry
        self._treedef = _treedef
        self._start_params = _start_params

    def read_iteration(self, iteration):
        out = _read_optimization_iteration(
            database=self._database,
            iteration=iteration,
            params_treedef=self._treedef,
            registry=self._registry,
        )
        return out

    def read_history(self):
        out = _read_optimization_history(
            database=self._database,
            params_treedef=self._treedef,
            registry=self._registry,
        )
        return out

    def read_start_params(self):
        return self._start_params


def _read_optimization_iteration(database, iteration, params_treedef, registry):
    """Get information about an optimization iteration."""
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

        # iteration is negative here!
        rowid = highest_rowid + iteration + 1

    data = read_specific_row(
        database,
        table_name="optimization_iterations",
        rowid=rowid,
        return_type="list_of_dicts",
    )

    if len(data) == 0:
        raise IndexError(f"Invalid iteration requested: {iteration}")
    else:
        data = data[0]

    params = tree_unflatten(params_treedef, data["params"], registry=registry)
    data["params"] = params

    return data


def _read_optimization_history(database, params_treedef, registry):
    """Read a histories out values, parameters and other information."""

    raw_res, _ = read_new_rows(
        database=database,
        table_name="optimization_iterations",
        last_retrieved=0,
        return_type="list_of_dicts",
    )

    history = {"params": [], "criterion": [], "runtime": []}
    for data in raw_res:
        if data["value"] is not None:
            params = tree_unflatten(params_treedef, data["params"], registry=registry)
            history["params"].append(params)
            history["criterion"].append(data["value"])
            history["runtime"].append(data["timestamp"])

    times = np.array(history["runtime"])
    times -= times[0]
    history["runtime"] = times

    return history
