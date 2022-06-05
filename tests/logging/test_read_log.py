import numpy as np
import pandas as pd
import pytest
from estimagic.logging.read_log import OptimizeLogReader
from estimagic.logging.read_log import read_optimization_problem_table
from estimagic.logging.read_log import read_start_params
from estimagic.logging.read_log import read_steps_table
from estimagic.optimization.optimize import minimize


@pytest.fixture
def example_db(tmp_path):
    path = tmp_path / "test.db"

    def _crit(params):
        x = np.array(list(params.values()))
        return x @ x

    minimize(
        criterion=_crit,
        params={"a": 1, "b": 2, "c": 3},
        algorithm="scipy_lbfgsb",
        logging=path,
    )
    return path


def test_read_start_params(example_db):
    res = read_start_params(example_db)
    assert res == {"a": 1, "b": 2, "c": 3}


def test_log_reader_read_start_params(example_db):
    reader = OptimizeLogReader(example_db)
    res = reader.read_start_params()
    assert res == {"a": 1, "b": 2, "c": 3}


def test_log_reader_read_iteration(example_db):
    reader = OptimizeLogReader(example_db)
    first_row = reader.read_iteration(0)
    assert first_row["params"] == {"a": 1, "b": 2, "c": 3}
    assert first_row["rowid"] == 1
    assert first_row["value"] == 14

    last_row = reader.read_iteration(-1)
    assert list(last_row["params"]) == ["a", "b", "c"]
    assert np.allclose(last_row["value"], 0)


def test_log_reader_read_history(example_db):
    reader = OptimizeLogReader(example_db)
    res = reader.read_history()
    assert res["runtime"][0] == 0
    assert res["criterion"][0] == 14
    assert res["params"][0] == {"a": 1, "b": 2, "c": 3}


def test_read_steps_table(example_db):
    res = read_steps_table(example_db)
    assert isinstance(res, pd.DataFrame)
    assert res.loc[0, "rowid"] == 1
    assert res.loc[0, "type"] == "optimization"
    assert res.loc[0, "status"] == "complete"


def test_read_optimization_problem_table(example_db):
    res = read_optimization_problem_table(example_db)
    assert isinstance(res, pd.DataFrame)


def test_non_existing_database_raises_error():
    with pytest.raises(FileNotFoundError):
        read_start_params("i_do_not_exist.db")
