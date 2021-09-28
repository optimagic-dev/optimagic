from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from estimagic.logging.database_utilities import append_row
from estimagic.logging.database_utilities import load_database
from estimagic.logging.database_utilities import make_optimization_iteration_table
from estimagic.logging.database_utilities import make_optimization_problem_table
from estimagic.logging.read_log import read_optimization_iteration
from estimagic.logging.read_log import read_start_params


def test_read_start_params():
    this_folder = Path(__file__).resolve().parent
    db_path = this_folder.parent / "dashboard" / "db1.db"
    res = read_start_params(path_or_database=db_path)
    assert isinstance(res, pd.DataFrame)
    assert "value" in res.columns
    assert "group" in res.columns


def test_read_optimization_iteration(tmp_path):
    path = tmp_path / "test.db"
    database = load_database(path=path)

    # add the optimization_iterations table
    make_optimization_iteration_table(database, first_eval={"output": 0.5})
    iteration_data = [
        {"params": np.array([0])},
        {"params": np.array([1])},
        {"params": np.array([2])},
    ]

    for data in iteration_data:
        append_row(data, "optimization_iterations", database, path, False)

    # add the optimization_problem table
    make_optimization_problem_table(database)
    problem_data = {"params": pd.DataFrame(data=[10], columns=["value"])}
    append_row(problem_data, "optimization_problem", database, path, False)

    first_row_calc = read_optimization_iteration(path, 0)
    assert first_row_calc["rowid"] == 1
    calculated_params = first_row_calc["params"]
    expected_params = pd.DataFrame(data=[0], columns=["value"])
    assert_frame_equal(calculated_params, expected_params, check_dtype=False)

    last_row_calc = read_optimization_iteration(path, -1)
    assert last_row_calc["rowid"] == 3
    calculated_params = last_row_calc["params"]
    expected_params = pd.DataFrame(data=[2], columns=["value"])
    assert_frame_equal(calculated_params, expected_params, check_dtype=False)


def test_non_existing_database_raises_error():
    with pytest.raises(FileNotFoundError):
        read_optimization_iteration("i_do_not_exist.db", -1)
