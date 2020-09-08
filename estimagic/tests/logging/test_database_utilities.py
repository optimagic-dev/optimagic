import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import sqlalchemy
from numpy.testing import assert_array_equal
from sqlalchemy import Float
from sqlalchemy import PickleType

from estimagic.logging.database_utilities import append_row
from estimagic.logging.database_utilities import load_database
from estimagic.logging.database_utilities import make_optimization_iteration_table
from estimagic.logging.database_utilities import make_optimization_problem_table
from estimagic.logging.database_utilities import make_optimization_status_table
from estimagic.logging.database_utilities import read_last_rows
from estimagic.logging.database_utilities import read_new_rows


@pytest.fixture
def iteration_data():
    data = {
        "external_params": np.ones(1),
        "internal_params": np.ones(1),
        "timestamp": datetime(year=2020, month=4, day=9, hour=12, minute=41, second=1),
        "distance_origin": 1.0,
        "distance_ones": 0.0,
        "value": 5.0,
    }
    return data


@pytest.fixture
def problem_data():
    data = {
        "direction": "maximize",
        "criterion": np.sum,
        "params": np.arange(3),
        "algorithm": "bla",
        "constraints": [{"type": "bla"}],
        "algo_options": None,
        "derivative": None,
        "derivative_kwargs": None,
        "criterion_and_derivative": None,
        "criterion_and_derivative_kwargs": None,
        "numdiff_options": {},
        "log_options": {"fast_logging": False},
    }
    return data


def test_load_database_from_path(tmp_path):
    """Test that database is generated because it does not exist."""
    path = tmp_path / "test.db"
    database = load_database(path=path)
    assert isinstance(database, sqlalchemy.MetaData)
    assert database.bind is not None


def test_load_database_after_pickling(tmp_path):
    """Pickling unsets database.bind. Test that load_database sets it again."""
    path = tmp_path / "test.db"
    database = load_database(path=path)
    database = pickle.loads(pickle.dumps(database))
    database = load_database(metadata=database, path=path)
    assert database.bind is not None


def test_load_database_with_bound_metadata(tmp_path):
    """Test that nothing happens when load_database is called with bound MetaData."""
    path = tmp_path / "test.db"
    database = load_database(path=path)
    new_database = load_database(metadata=database)
    assert new_database is database


def test_optimization_iteration_table_scalar(tmp_path, iteration_data):
    path = tmp_path / "test.db"
    database = load_database(path=path)
    make_optimization_iteration_table(database, first_eval={"output": 0.5})
    append_row(iteration_data, "optimization_iterations", database, path, False)
    res = read_last_rows(database, "optimization_iterations", 1, "list_of_dicts")
    assert isinstance(res, list) and isinstance(res[0], dict)
    res = res[0]
    assert res["rowid"] == 1
    assert res["internal_derivative"] is None
    for key in ["internal_params", "external_params"]:
        assert_array_equal(res[key], iteration_data[key])

    for key in ["distance_ones", "distance_origin", "value", "timestamp"]:
        assert res[key] == iteration_data[key]


def test_optimization_iteration_table_vector_valued(tmp_path):
    path = tmp_path / "test.db"
    database = load_database(path=path)
    make_optimization_iteration_table(
        database, first_eval={"output": {"contributions": np.ones(3), "value": 0.5}}
    )
    assert isinstance(
        database.tables["optimization_iterations"].columns["contributions"].type,
        PickleType,
    )


def test_optimization_iteration_table_dict_valued(tmp_path):
    path = tmp_path / "test.db"
    database = load_database(path=path)
    first_eval = {
        "output": {"contributions": np.ones(3), "value": 5, "bla": pd.DataFrame()}
    }
    make_optimization_iteration_table(database, first_eval=first_eval)
    for col in ["contributions", "bla"]:
        assert isinstance(
            database.tables["optimization_iterations"].columns[col].type, PickleType
        )
    assert isinstance(
        database.tables["optimization_iterations"].columns["value"].type, Float
    )


def test_optimization_status_table(tmp_path):
    path = tmp_path / "test.db"
    database = load_database(path=path)
    make_optimization_status_table(database)
    for status in ["scheduled", "running", "success"]:
        append_row({"status": status}, "optimization_status", database, path, False)

    res, _ = read_new_rows(database, "optimization_status", 1, "dict_of_lists")

    expected = {"rowid": [2, 3], "status": ["running", "success"]}
    assert res == expected


def test_optimization_problem_table(tmp_path, problem_data):
    path = tmp_path / "test.db"
    database = load_database(path=path)
    make_optimization_problem_table(database)
    append_row(problem_data, "optimization_problem", database, path, False)
    res = read_last_rows(database, "optimization_problem", 1, "list_of_dicts")[0]
    assert res["rowid"] == 1
    for key, expected in problem_data.items():
        if key == "criterion":
            assert res[key](np.ones(3)) == 3
        elif isinstance(expected, np.ndarray):
            assert_array_equal(res[key], expected)
        else:
            assert res[key] == expected
