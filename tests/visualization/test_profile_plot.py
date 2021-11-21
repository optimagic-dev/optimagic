import numpy as np
import pandas as pd
import pytest
from estimagic.visualization.profile_plot import _create_solution_times
from estimagic.visualization.profile_plot import _determine_alpha_grid
from estimagic.visualization.profile_plot import _find_switch_points


@pytest.fixture
def performance_ratios():
    df = pd.DataFrame(
        data={"algo1": [1.0, 1.0, 4.0], "algo2": [1.5, np.inf, 1.0]},
        index=["prob1", "prob2", "prob3"],
    )
    return df


def test_find_switch_points(performance_ratios):
    res = _find_switch_points(performance_ratios)
    expected = np.array([1.0, 1.5, 4.0])
    np.testing.assert_array_almost_equal(res, expected)


def test_determine_alpha_grid(performance_ratios):
    res = _determine_alpha_grid(performance_ratios)
    expected = np.array([1.0 + 1e-10, 1.25, 1.5, 2.75, 4.0, 4.0 * 1.025, 4.0 * 1.05])
    np.testing.assert_array_almost_equal(res, expected)


def test_create_solution_times():
    df = pd.DataFrame(
        columns=["problem", "algorithm", "n_evaluations"],
        data=[
            ["prob1", "algo1", 0],
            ["prob1", "algo1", 1],
            #
            ["prob1", "algo2", 2],
            ["prob1", "algo2", 3],
            #
            ["prob2", "algo1", 5],
            #
            ["prob2", "algo2", 0],
            ["prob2", "algo2", 1],
        ],
    )
    info = pd.DataFrame(
        {
            "algo1": [True, True],
            "algo2": [True, False],
        },
        index=["prob1", "prob2"],
    )
    expected = pd.DataFrame(
        {
            "algo1": [1, 5],
            "algo2": [3, np.inf],
        },
        index=pd.Index(["prob1", "prob2"], name="problem"),
    )
    expected.columns.name = "algorithm"

    res = _create_solution_times(
        df=df, runtime_measure="n_evaluations", converged_info=info
    )
    pd.testing.assert_frame_equal(res, expected)
