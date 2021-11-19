import pandas as pd
import pytest
from estimagic.examples.process_benchmark_results import _find_first_converged


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
