import pandas as pd
from estimagic.visualization.benchmark_plots import (
    calculate_share_of_improvement_missing,
)
from estimagic.visualization.benchmark_plots import clip_histories
from estimagic.visualization.benchmark_plots import (
    get_history_as_stacked_sr_from_results,
)
from estimagic.visualization.benchmark_plots import get_lowest_so_far


def test_lowest_so_far_single_prob_and_algo():
    df = pd.DataFrame({"value": [5, 3, 3, 4, 2, 4, 0]})
    df.index.name = "evaluation"
    df = pd.concat({"prob1": df}, names=["problem"])
    df = pd.concat({"algo1": df}, names=["algorithm"])
    df = df.reorder_levels(["problem", "algorithm", "evaluation"])

    expected = pd.Series([5, 3, 3, 3, 2, 2, 0])
    expected.index.name = "evaluation"
    expected = pd.concat({"prob1": expected}, names=["problem"])
    expected = pd.concat({"algo1": expected}, names=["algorithm"])
    expected = expected.reorder_levels(["problem", "algorithm", "evaluation"])

    res = get_lowest_so_far(df, col="value")
    pd.testing.assert_series_equal(expected, res, check_dtype=False)


def test_lowest_so_far2():
    raise ValueError("In practice this does not create monotone Series!")


def test_get_history_as_stacked_sr_from_results():
    raise NotImplementedError()


def test_clip_histories():
    raise NotImplementedError()


def test_calculate_share_of_improvement_missing():
    probs = ["prob1", "prob2"]
    start_values = pd.Series([20, 10], index=pd.Index(probs, name="problem"))
    target_values = pd.Series([5, 0], index=pd.Index(probs, name="problem"))
    middle_values1 = pd.Series([10, 5], index=pd.Index(probs, name="problem"))
    middle_values2 = pd.Series([25, 2], index=pd.Index(probs, name="problem"))
    current1 = pd.concat(
        {0: start_values, 2: middle_values1, 5: target_values}, names=["evaluation"]
    )
    current2 = pd.concat({3: middle_values2}, names=["evaluation"])
    current = pd.concat({"algo1": current1, "algo2": current2}, names=["algorithm"])
    current = current.reorder_levels(["problem", "algorithm", "evaluation"])

    expected = pd.DataFrame(
        [
            ["prob1", "algo1", 0, 1.0],
            ["prob2", "algo1", 0, 1.0],
            ["prob1", "algo1", 2, 1 / 3],
            ["prob2", "algo1", 2, 0.5],
            ["prob1", "algo1", 5, 0.0],
            ["prob2", "algo1", 5, 0.0],
            ["prob1", "algo2", 3, (25 - 5) / 15],
            ["prob2", "algo2", 3, 0.2],
        ],
        columns=["problem", "algorithm", "evaluation", "value"],
    )
    expected = expected.set_index(["problem", "algorithm", "evaluation"])["value"]

    res = calculate_share_of_improvement_missing(
        current, start_values=start_values, target_values=target_values
    )
    pd.testing.assert_series_equal(expected, res, check_names=False)
