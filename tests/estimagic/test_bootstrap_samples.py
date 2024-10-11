import numpy as np
import pandas as pd
import pytest
from estimagic.bootstrap_samples import (
    _calculate_bootstrap_indices_weights,
    _convert_cluster_ids_to_indices,
    _get_bootstrap_samples_from_indices,
    get_bootstrap_indices,
    get_bootstrap_samples,
)
from numpy.testing import assert_array_equal as aae
from optimagic.utilities import get_rng
from pandas.testing import assert_frame_equal as afe
from pandas.testing import assert_series_equal as ase


@pytest.fixture()
def data():
    df = pd.DataFrame()
    df["id"] = np.arange(900)
    df["hh"] = [3, 1, 2, 0, 0, 2, 5, 4, 5] * 100
    df["weights"] = np.ones(900)
    return df


def test_get_bootstrap_indices_randomization_works_without_clustering(data):
    rng = get_rng(seed=12345)
    res = get_bootstrap_indices(data, n_draws=2, rng=rng)
    assert set(res[0]) != set(res[1])


def test_get_bootstrap_indices_radomization_works_with_clustering(data):
    rng = get_rng(seed=12345)
    res = get_bootstrap_indices(data, cluster_by="hh", n_draws=2, rng=rng)
    assert set(res[0]) != set(res[1])


def test_get_bootstrap_indices_randomization_works_with_weights(data):
    rng = get_rng(seed=12345)
    res = get_bootstrap_indices(data, weight_by="weights", n_draws=2, rng=rng)
    assert set(res[0]) != set(res[1])


def test_get_bootstrap_indices_randomization_works_with_weights_and_clustering(data):
    rng = get_rng(seed=12345)
    res = get_bootstrap_indices(
        data, weight_by="weights", cluster_by="hh", n_draws=2, rng=rng
    )
    assert set(res[0]) != set(res[1])


def test_get_bootstrap_indices_randomization_works_with_and_without_weights(data):
    rng1 = get_rng(seed=12345)
    rng2 = get_rng(seed=12345)
    res1 = get_bootstrap_indices(data, n_draws=1, rng=rng1)
    res2 = get_bootstrap_indices(data, weight_by="weights", n_draws=1, rng=rng2)
    assert not np.array_equal(res1, res2)


def test_get_boostrap_indices_randomization_works_with_extreme_case(data):
    rng = get_rng(seed=12345)
    weights = np.zeros(900)
    weights[0] = 1.0
    data["weights"] = weights
    res = get_bootstrap_indices(data, weight_by="weights", n_draws=1, rng=rng)
    assert len(np.unique(res)) == 1


def test_clustering_leaves_households_intact(data):
    rng = get_rng(seed=12345)
    indices = get_bootstrap_indices(data, cluster_by="hh", n_draws=1, rng=rng)[0]
    sampled = data.iloc[indices]
    sampled_households = sampled["hh"].unique()
    for household in sampled_households:
        expected_ids = set(data[data["hh"] == household]["id"].unique())
        actual_ids = set(sampled[sampled["hh"] == household]["id"].unique())
        assert expected_ids == actual_ids


def test_convert_cluster_ids_to_indices():
    cluster_col = pd.Series([2, 2, 0, 1, 0, 1])
    drawn_clusters = np.array([[1, 0]])
    expected = np.array([3, 5, 2, 4])
    calculated = _convert_cluster_ids_to_indices(cluster_col, drawn_clusters)[0]
    aae(calculated, expected)


def test_get_bootstrap_samples_from_indices():
    indices = [np.array([0, 1])]
    data = pd.DataFrame(np.arange(6).reshape(3, 2))
    expected = pd.DataFrame(np.arange(4).reshape(2, 2))
    calculated = _get_bootstrap_samples_from_indices(data, indices)[0]
    afe(calculated, expected)


def test_get_bootstrap_samples_runs(data):
    rng = get_rng(seed=12345)
    get_bootstrap_samples(data, n_draws=2, rng=rng)


@pytest.fixture
def sample_data():
    return pd.DataFrame({"weight": [1, 2, 3, 4], "cluster": ["A", "A", "B", "B"]})


def test_no_weights_no_clusters(sample_data):
    result = _calculate_bootstrap_indices_weights(sample_data, None, None)
    assert result is None


def test_weights_no_clusters(sample_data):
    result = _calculate_bootstrap_indices_weights(sample_data, "weight", None)
    expected = pd.Series([0.1, 0.2, 0.3, 0.4], index=sample_data.index, name="weight")
    pd.testing.assert_series_equal(result, expected)


def test_weights_and_clusters(sample_data):
    result = _calculate_bootstrap_indices_weights(sample_data, "weight", "cluster")
    expected = pd.Series(
        [0.3, 0.7], index=pd.Index(["A", "B"], name="cluster"), name="weight"
    )
    ase(result, expected)


def test_invalid_weight_column():
    data = pd.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(KeyError):
        _calculate_bootstrap_indices_weights(data, "weight", None)


def test_invalid_cluster_column(sample_data):
    with pytest.raises(KeyError):
        _calculate_bootstrap_indices_weights(sample_data, "weight", "invalid_cluster")


def test_empty_dataframe():
    empty_df = pd.DataFrame()
    result = _calculate_bootstrap_indices_weights(empty_df, None, None)
    assert result is None


def test_some_zero_weights_with_clusters():
    data = pd.DataFrame({"weight": [0, 1, 0, 2], "cluster": ["A", "A", "B", "B"]})
    result = _calculate_bootstrap_indices_weights(data, "weight", "cluster")
    expected = pd.Series(
        [1 / 3, 2 / 3], index=pd.Index(["A", "B"], name="cluster"), name="weight"
    )
    ase(result, expected)
