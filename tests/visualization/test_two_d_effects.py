import numpy as np
import pandas as pd
import pytest
from estimagic.visualization.two_d_effects import _create_linspace_with_start_value
from estimagic.visualization.two_d_effects import _create_points_to_evaluate


@pytest.fixture
def params():
    params = pd.DataFrame(
        data=[1.5, 4.5],
        columns=["value"],
        index=[f"x_{i}" for i in range(2)],
    )

    params["lower_bound"] = [-5, 0]
    params["upper_bound"] = [4, 9]
    return params


def test_create_linspace_with_start_values(params):
    res = _create_linspace_with_start_value(params, "x_1", 10)
    expected = np.array([0, 1, 2, 3, 4, 4.5, 5, 6, 7, 8, 9])
    np.testing.assert_array_almost_equal(res, expected)


def test_create_points_to_evaluate(params):
    res = _create_points_to_evaluate(params, 3)
    expected = pd.DataFrame(
        data=[
            [1.5, 4.5],
            [-5.0, 0.0],
            [-5.0, 4.5],
            [-5.0, 9.0],
            [-0.5, 0.0],
            [-0.5, 4.5],
            [-0.5, 9.0],
            [1.5, 0.0],
            [1.5, 9.0],
            [4.0, 0.0],
            [4.0, 4.5],
            [4.0, 9.0],
        ],
        columns=["x_0", "x_1"],
    )
    np.testing.assert_array_almost_equal(res, expected)
