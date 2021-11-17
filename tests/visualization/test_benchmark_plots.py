import pandas as pd
from estimagic.visualization.benchmark_plots import (
    calculate_share_of_improvement_missing,
)
from estimagic.visualization.benchmark_plots import lowest_so_far


def test_lowest_so_far():
    sr = pd.Series([5, 3, 3, 4, 2, 4, 0])
    expected = pd.Series([5, 3, 3, 3, 2, 2, 0])
    res = lowest_so_far(sr)
    pd.testing.assert_series_equal(expected, res, check_dtype=False)


def test_calculate_share_of_improvement_missing():
    start_value = 20
    target_value = 5
    current = pd.Series([start_value, 10, target_value])
    expected = pd.Series([1, 1 / 3, 0])
    res = calculate_share_of_improvement_missing(
        current, start_value=start_value, target_value=target_value
    )
    pd.testing.assert_series_equal(expected, res)
