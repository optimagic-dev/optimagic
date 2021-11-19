import pandas as pd
import pytest
from estimagic.examples.process_benchmark_results import _find_first_converged
from estimagic.examples.process_benchmark_results import _normalize


@pytest.fixture
def problem_algo_eval_df():
    df = pd.DataFrame()
    df["problem"] = ["prob1"] * 8 + ["prob2"] * 6
    df["algorithm"] = ["algo1"] * 4 + ["algo2"] * 4 + ["algo1"] * 3 + ["algo2"] * 3
    df["n_evaluations"] = [0, 1, 2, 3] * 2 + [0, 1, 2] * 2
    return df


def test_find_first_converged(problem_algo_eval_df):
    # we can assume monotonicity, i.e. no switch back from True to False
    converged = pd.Series(
        [  # in the middle
            False,
            False,
            True,
            True,
        ]
        + [  # last entry
            False,
            False,
            False,
            True,
        ]
        + [  # first entry
            True,
            True,
            True,
        ]
        + [  # not converged
            False,
            False,
            False,
        ]
    )
    res = _find_first_converged(converged, problem_algo_eval_df)
    expected = pd.Series(
        [  # in the middle
            False,
            False,
            True,
            False,
        ]
        + [  # last entry
            False,
            False,
            False,
            True,
        ]
        + [  # first entry
            True,
            False,
            False,
        ]
        + [  # not converged
            False,
            False,
            False,
        ]
    )
    pd.testing.assert_series_equal(res, expected)


def test_normalize_maximize():
    start_values = pd.Series([1, 2, 3], index=["prob1", "prob2", "prob3"])
    target_values = pd.Series([5, 7, 10], index=["prob1", "prob2", "prob3"])

    df = pd.DataFrame()
    df["problem"] = ["prob1", "prob2", "prob3"] * 3
    df["criterion"] = start_values.tolist() + [2, 4, 9] + target_values.tolist()

    res = _normalize(
        df=df, col="criterion", start_values=start_values, target_values=target_values
    )

    # total improvements are [4, 5, 7]
    # missing improvements are [3, 3, 1] for the [2, 4, 9] part

    expected = pd.Series([1] * 3 + [3 / 4, 3 / 5, 1 / 7] + [0] * 3)
    pd.testing.assert_series_equal(res, expected)
