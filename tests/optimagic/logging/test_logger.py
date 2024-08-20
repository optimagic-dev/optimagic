from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest
from optimagic.logging.logger import (
    LogOptions,
    LogReader,
    LogStore,
    SQLiteLogOptions,
    SQLiteLogReader,
)
from optimagic.optimization.optimize import minimize
from optimagic.parameters.tree_registry import get_registry
from optimagic.typing import Direction
from pybaum import tree_equal, tree_just_flatten


@pytest.fixture()
def example_db(tmp_path):
    path = tmp_path / "test.db"

    def _crit(params):
        x = np.array(list(params.values()))
        return x @ x

    minimize(
        fun=_crit,
        params={"a": 1, "b": 2, "c": 3},
        algorithm="scipy_lbfgsb",
        logging=path,
    )
    return path


def test_read_start_params(example_db):
    res = LogReader.from_options(SQLiteLogOptions(example_db)).read_start_params()
    assert res == {"a": 1, "b": 2, "c": 3}


def test_log_reader_read_start_params(example_db):
    reader = LogReader.from_options(SQLiteLogOptions(example_db))
    res = reader.read_start_params()
    assert res == {"a": 1, "b": 2, "c": 3}


@pytest.mark.xfail(reason="Iteration logging is currently not implemented.")
def test_log_reader_read_iteration(example_db):
    reader = SQLiteLogReader(example_db)
    first_row = reader.read_iteration(0)
    assert first_row["params"] == {"a": 1, "b": 2, "c": 3}
    assert first_row["rowid"] == 1
    assert first_row["value"] == 14

    last_row = reader.read_iteration(-1)
    assert list(last_row["params"]) == ["a", "b", "c"]
    assert np.allclose(last_row["value"], 0)


def test_log_reader_index_exception(example_db):
    with pytest.raises(IndexError):
        SQLiteLogReader(example_db).read_iteration(10)

    with pytest.raises(IndexError):
        SQLiteLogReader(example_db).read_iteration(-4)


@pytest.mark.xfail(reason="Iteration logging is currently not implemented.")
def test_log_reader_read_history(example_db):
    reader = SQLiteLogReader(example_db)
    res = reader.read_history()
    assert res["runtime"][0] == 0
    assert res["criterion"][0] == 14
    assert res["params"][0] == {"a": 1, "b": 2, "c": 3}


@pytest.mark.xfail(reason="Iteration logging is currently not implemented.")
def test_log_reader_read_multistart_history(example_db):
    reader = SQLiteLogReader(example_db)
    history, local_history, exploration = reader.read_multistart_history(
        direction=Direction.MINIMIZE
    )
    assert local_history is None
    assert exploration is None

    registry = get_registry(extended=True)
    assert tree_equal(
        tree_just_flatten(asdict(history), registry=registry),
        tree_just_flatten(asdict(reader.read_history()), registry=registry),
    )


def test_read_steps_table(example_db):
    res = SQLiteLogReader(example_db)._step_store.to_df()
    assert isinstance(res, pd.DataFrame)
    assert res.loc[0, "rowid"] == 1
    assert res.loc[0, "type"] == "optimization"
    assert res.loc[0, "status"] == "complete"


def test_read_optimization_problem_table(example_db):
    res = SQLiteLogReader(example_db).problem_df
    assert isinstance(res, pd.DataFrame)


def test_non_existing_database_raises_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        SQLiteLogReader(tmp_path / "i_do_not_exist.db").read_start_params()


def test_available_log_options():
    available_types = LogOptions.available_option_types()
    assert len(available_types) == 1
    assert available_types[0] is SQLiteLogOptions


def test_no_registered():
    class DummyOptions(LogOptions):
        pass

    with pytest.raises(ValueError, match="DummyOptions"):
        LogReader.from_options(DummyOptions())

    with pytest.raises(ValueError, match="DummyOptions"):
        LogStore.from_options(DummyOptions())
