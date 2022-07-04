import numpy as np
import pandas as pd
import pytest
from estimagic.benchmarking.process_benchmark_results import _clip_histories
from estimagic.benchmarking.process_benchmark_results import _find_first_converged
from estimagic.benchmarking.process_benchmark_results import (
    _get_history_as_stacked_sr_from_results,
)
from estimagic.benchmarking.process_benchmark_results import (
    _get_history_of_the_parameter_distance,
)
from estimagic.benchmarking.process_benchmark_results import _make_history_monotone
from estimagic.benchmarking.process_benchmark_results import _normalize
from estimagic.benchmarking.process_benchmark_results import (
    create_convergence_histories,
)

PROBLEMS = ["prob1", "prob2", "prob3"]


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


def test_normalize_minimize():
    start_values = pd.Series([5, 4, 10], index=PROBLEMS)
    target_values = pd.Series([1, 0, 0], index=PROBLEMS)

    df = pd.DataFrame()
    df["problem"] = PROBLEMS * 3
    df["criterion"] = start_values.tolist() + [2, 3, 9] + target_values.tolist()

    res = _normalize(
        df=df, col="criterion", start_values=start_values, target_values=target_values
    )

    # total improvements are [4, 4, 10]
    # missing improvements are [1, 3, 9] for the [2, 3, 9] part

    expected = pd.Series([1] * 3 + [0.25, 0.75, 0.9] + [0] * 3)
    pd.testing.assert_series_equal(res, expected)


def test_normalize_maximize():
    start_values = pd.Series([1, 2, 3], index=PROBLEMS)
    target_values = pd.Series([5, 7, 10], index=PROBLEMS)

    df = pd.DataFrame()
    df["problem"] = PROBLEMS * 3
    df["criterion"] = start_values.tolist() + [2, 4, 9] + target_values.tolist()

    res = _normalize(
        df=df, col="criterion", start_values=start_values, target_values=target_values
    )

    # total improvements are [4, 5, 7]
    # missing improvements are [3, 3, 1] for the [2, 4, 9] part

    expected = pd.Series([1] * 3 + [3 / 4, 3 / 5, 1 / 7] + [0] * 3)
    pd.testing.assert_series_equal(res, expected)


@pytest.fixture
def df_for_clip_histories(problem_algo_eval_df):
    df = problem_algo_eval_df
    df["monotone_criterion_normalized"] = [
        # prob1, algo1: converged on 2nd
        1.8,  # keep, not converged
        0.05,  # keep, 1st converged
        0.03,  # clip
        0.0,  # clip
        # prob1, algo2: converged on last
        5.4,  # keep, not converged
        3.3,  # keep, not converged
        2.2,  # keep, not converged
        0.08,  # keep, 1st converged
        # prob2, algo1: converged on first
        0.08,  # keep, 1st converged
        0.04,  # drop
        0.01,  # drop
        # prob2, algo2: not converged
        3.3,  # keep, not converged
        2.2,  # keep, not converged
        1.1,  # keep, not converged
    ]
    df["monotone_parameter_distance_normalized"] = [
        # prob1, algo1: converged on 3rd -> 1 after y criterion
        1.8,  # keep, not converged
        0.5,  # keep, not converged
        0.05,  # keep, 1st converged
        0.01,  # clip
        # prob1, algo2: converged on 3rd -> 1 before y criterion
        2.2,  # keep, not converged
        1.1,  # keep, not converged
        0.04,  # keep, 1st converged
        0.03,  # clip
        # prob2, algo1: not converged (converged in y)
        3.0,  # keep, not converged
        3.0,  # keep, not converged
        3.0,  # keep, not converged
        # prob2, algo2: converged on 3rd (not converged in y)
        2.2,  # keep, not converged
        1.1,  # keep, not converged
        0.04,  # keep, 1st converged
    ]
    return df


def test_clip_histories_y(df_for_clip_histories):
    expected_shortened = df_for_clip_histories.loc[[0, 1, 4, 5, 6, 7, 8, 11, 12, 13]]
    expected_info = pd.DataFrame(
        {"algo1": [True, True], "algo2": [True, False]},
        index=["prob1", "prob2"],
    )
    res_shortened, res_info = _clip_histories(
        df=df_for_clip_histories,
        stopping_criterion="y",
        x_precision=0.1,
        y_precision=0.1,
    )
    pd.testing.assert_frame_equal(res_shortened, expected_shortened)
    pd.testing.assert_frame_equal(res_info, expected_info, check_names=False)


def test_clip_histories_x(df_for_clip_histories):
    expected_shortened = df_for_clip_histories.loc[
        [0, 1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13]
    ]
    expected_info = pd.DataFrame(
        {"algo1": [True, False], "algo2": [True, True]},
        index=["prob1", "prob2"],
    )
    res_shortened, res_info = _clip_histories(
        df=df_for_clip_histories,
        stopping_criterion="x",
        x_precision=0.1,
        y_precision=0.1,
    )
    pd.testing.assert_frame_equal(res_shortened, expected_shortened)
    pd.testing.assert_frame_equal(res_info, expected_info, check_names=False)


def test_clip_histories_x_and_y_with_nan(df_for_clip_histories):
    df = df_for_clip_histories
    df.loc[df["problem"] == "prob2", "monotone_parameter_distance_normalized"] = np.nan

    expected_shortened = df.loc[[0, 1, 2, 4, 5, 6, 7]]
    expected_info = pd.DataFrame(
        {"algo1": [True], "algo2": [True]},
        index=["prob1"],
    )

    res_shortened, res_info = _clip_histories(
        df=df,
        stopping_criterion="x_and_y",
        x_precision=0.1,
        y_precision=0.1,
    )
    pd.testing.assert_frame_equal(res_shortened, expected_shortened)
    pd.testing.assert_frame_equal(res_info, expected_info, check_names=False)


def test_clip_histories_x_or_y_no_nan(df_for_clip_histories):
    df = df_for_clip_histories

    expected_shortened = df.loc[[0, 1, 4, 5, 6, 8, 11, 12, 13]]
    expected_info = pd.DataFrame(
        {"algo1": [True, True], "algo2": [True, True]},
        index=["prob1", "prob2"],
    )

    res_shortened, res_info = _clip_histories(
        df=df,
        stopping_criterion="x_or_y",
        x_precision=0.1,
        y_precision=0.1,
    )
    pd.testing.assert_frame_equal(res_shortened, expected_shortened)
    pd.testing.assert_frame_equal(res_info, expected_info, check_names=False)


def test_make_history_monotone_minimize():
    sorted_df = pd.DataFrame(
        columns=["problem", "algorithm", "n_evaluations", "to_make_monotone"],
        data=[
            # already monotone
            ["prob1", "algo1", 0, 3.3],
            ["prob1", "algo1", 1, 2.2],
            ["prob1", "algo1", 2, 1.1],
            # 3rd & 4th entry must be changed
            ["prob1", "algo2", 0, 3.3],
            ["prob1", "algo2", 1, 1.1],
            ["prob1", "algo2", 2, 2.2],  # 1.1
            ["prob1", "algo2", 2, 5.0],  # 1.1
            # up, down, up, down
            ["prob2", "algo1", 0, 2.2],  # 2.2
            ["prob2", "algo1", 1, 3.3],  # 2.2
            ["prob2", "algo1", 2, 1.1],  # 1.1
            ["prob2", "algo1", 3, 2.5],  # 1.1
            ["prob2", "algo1", 4, 2.0],  # 1.1
        ],
    )
    shuffled = sorted_df.sample(frac=1, random_state=40954)

    res_shuffled = _make_history_monotone(
        df=shuffled, target_col="to_make_monotone", direction="minimize"
    )
    res_sorted = _make_history_monotone(
        df=sorted_df, target_col="to_make_monotone", direction="minimize"
    )

    expected = pd.Series(
        [  # prob1, algo1
            3.3,
            2.2,
            1.1,
            # prob1, algo2
            3.3,
            1.1,
            1.1,
            1.1,
            # prob2, algo1
            2.2,
            2.2,
            1.1,
            1.1,
            1.1,
        ],
        name="to_make_monotone",
    )
    pd.testing.assert_series_equal(res_sorted, expected)
    pd.testing.assert_series_equal(res_shuffled, expected)


@pytest.fixture
def benchmark_results():
    sec = pd.Timedelta(seconds=1)
    results = {
        ("prob1", "algo1"): {
            "criterion_history": pd.Series([1, 2, 3]),
            "time_history": pd.Series([sec, 2 * sec, 3 * sec]),
            "params_history": pd.DataFrame(
                [
                    [1, 2],
                    [1, 1],
                    [0.5, 0.5],
                ]
            ),
        },
        ("prob1", "algo2"): {
            "criterion_history": pd.Series([1, 2.5]),
            "time_history": pd.Series([0.5 * sec, 1.5 * sec]),
            "params_history": pd.DataFrame([[2, 3], [2, 2]]),
        },
        ("prob2", "algo1"): {
            "criterion_history": pd.Series([50, 40]),
            "time_history": pd.Series([3 * sec, 3.5 * sec]),
            "params_history": pd.DataFrame([[2], [4]]),
        },
        ("prob2", "algo2"): {
            "criterion_history": pd.Series([35]),
            "time_history": pd.Series([3.2 * sec]),
            "params_history": pd.DataFrame([[4.2]]),
        },
    }
    return results


def test_get_history_of_the_parameter_distance(benchmark_results):
    x_opt = {"prob1": np.array([1, 1]), "prob2": np.array([3])}
    res = _get_history_of_the_parameter_distance(results=benchmark_results, x_opt=x_opt)
    expected_df = pd.DataFrame(
        columns=["problem", "algorithm", "evaluation", "parameter_distance"],
        data=[
            ["prob1", "algo1", 0, 1],
            ["prob1", "algo1", 1, 0],
            ["prob1", "algo1", 2, np.sqrt(2 * 0.5**2)],
            ["prob1", "algo2", 0, np.sqrt(5)],
            ["prob1", "algo2", 1, np.sqrt(2)],
            ["prob2", "algo1", 0, 1],
            ["prob2", "algo1", 1, 1],
            ["prob2", "algo2", 0, 1.2],
        ],
    )
    expected = expected_df.set_index(["problem", "algorithm", "evaluation"])[
        "parameter_distance"
    ]
    pd.testing.assert_series_equal(res, expected)


def test_get_history_as_stacked_sr_from_results(benchmark_results):
    res = _get_history_as_stacked_sr_from_results(
        benchmark_results, key="criterion_history"
    )
    expected_df = pd.DataFrame(
        columns=["problem", "algorithm", "evaluation", "criterion"],
        data=[
            ["prob1", "algo1", 0, 1],
            ["prob1", "algo1", 1, 2],
            ["prob1", "algo1", 2, 3],
            ["prob1", "algo2", 0, 1],
            ["prob1", "algo2", 1, 2.5],
            ["prob2", "algo1", 0, 50],
            ["prob2", "algo1", 1, 40],
            ["prob2", "algo2", 0, 35],
        ],
    )
    expected = expected_df.set_index(["problem", "algorithm", "evaluation"])[
        "criterion"
    ]
    pd.testing.assert_series_equal(res, expected)


def test_create_convergence_histories(benchmark_results):
    problems = {
        "prob1": {
            "solution": {"value": 5, "params": pd.DataFrame(data={"value": [1, 1]})}
        },
        "prob2": {
            "solution": {"value": 1, "params": pd.DataFrame(data={"value": [3]})}
        },
    }
    res, _ = create_convergence_histories(
        problems=problems,
        results=benchmark_results,
        stopping_criterion=None,
        x_precision=None,
        y_precision=None,
    )

    expected_criterion = pd.DataFrame(
        columns=["problem", "algorithm", "evaluation", "criterion"],
        data=[
            ["prob1", "algo1", 0, 1],
            ["prob1", "algo1", 1, 2],
            ["prob1", "algo1", 2, 3],
            ["prob1", "algo2", 0, 1],
            ["prob1", "algo2", 1, 2.5],
            ["prob2", "algo1", 0, 50],
            ["prob2", "algo1", 1, 40],
            ["prob2", "algo2", 0, 35],
        ],
    )

    expected_x_distance = pd.DataFrame(
        columns=["problem", "algorithm", "evaluation", "parameter_distance"],
        data=[
            ["prob1", "algo1", 0, 1],
            ["prob1", "algo1", 1, 0],
            ["prob1", "algo1", 2, np.sqrt(2 * 0.5**2)],
            ["prob1", "algo2", 0, np.sqrt(5)],
            ["prob1", "algo2", 1, np.sqrt(2)],
            ["prob2", "algo1", 0, 1],
            ["prob2", "algo1", 1, 1],
            ["prob2", "algo2", 0, 1.2],
        ],
    )

    expected = pd.merge(
        expected_criterion,
        expected_x_distance,
        on=["problem", "algorithm", "evaluation"],
    ).rename(columns={"evaluation": "n_evaluations"})

    to_compare = [
        "problem",
        "algorithm",
        "n_evaluations",
        "criterion",
        "parameter_distance",
    ]
    # missing:
    # - criterion_normalized
    # - monotone_criterion
    # - monotone_criterion_normalized
    # - parameter_distance_normalized
    # - monotone_parameter_distance
    # - monotone_parameter_distance_normalized

    pd.testing.assert_frame_equal(res[to_compare], expected)
