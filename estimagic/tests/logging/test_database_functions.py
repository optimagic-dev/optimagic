import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from estimagic.logging.create_database import prepare_database
from estimagic.logging.read_database import read_last_iterations
from estimagic.logging.read_database import read_new_iterations
from estimagic.logging.read_database import read_scalar_field
from estimagic.logging.update_database import append_rows


@pytest.fixture
def database(tmp_path):
    params = pd.DataFrame()
    params["name"] = list("abc")
    database = prepare_database(
        path=tmp_path / "test.db",
        params=params,
        dash_options={"a": 3, "no_browser": True},
        constraints=[{"loc": "a", "type": "increasing"}],
        optimization_status="success",
    )

    tables = ["params_history", "criterion_history"]
    for i in range(10):
        params = pd.Series(index=list("abc"), data=i)
        critval = i ** 2
        append_rows(database, tables, [params, {"value": critval}])

    return database


def test_start_params_table(database):
    params = pd.DataFrame()
    params["name"] = list("abc")
    assert_frame_equal(read_scalar_field(database, "start_params"), params)


def test_optimization_status_table(database):
    assert read_scalar_field(database, "optimization_status") == "success"


def test_gradient_status_table(database):
    assert read_scalar_field(database, "gradient_status") == 0


def test_read_last_iterations_pandas(database):
    tables = ["params_history", "criterion_history"]
    res = read_last_iterations(database, tables, 3, "pandas")

    expected_params = pd.DataFrame(
        data=[[8, 7.0, 7.0, 7.0], [9, 8, 8, 8], [10, 9, 9, 9]],
        columns=["iteration", "a", "b", "c"],
    )
    expected_params.set_index("iteration", inplace=True)
    assert_frame_equal(res["params_history"], expected_params)

    expected_critvals = pd.Series(
        data=[49, 64, 81.0], index=[8, 9, 10], name="value"
    ).to_frame()
    expected_critvals.index.name = "iteration"
    assert_frame_equal(res["criterion_history"], expected_critvals)


def test_read_list_iterations_bokeh(database):
    res = read_last_iterations(database, "criterion_history", 3, "bokeh")
    assert res["value"] == [49, 64, 81]
    assert res["iteration"] == [8, 9, 10]


def test_read_new_iterations(database):
    tables = ["params_history", "criterion_history"]
    res, new_last = read_new_iterations(database, tables, 7, "pandas", 2)

    expected_params = pd.DataFrame(
        data=[[8, 7.0, 7.0, 7.0], [9, 8, 8, 8]], columns=["iteration", "a", "b", "c"]
    )
    expected_params.set_index("iteration", inplace=True)
    assert_frame_equal(res["params_history"], expected_params)

    expected_critvals = pd.Series(
        data=[49, 64.0], index=[8, 9], name="value"
    ).to_frame()
    expected_critvals.index.name = "iteration"
    assert_frame_equal(res["criterion_history"], expected_critvals)

    assert new_last == 9
