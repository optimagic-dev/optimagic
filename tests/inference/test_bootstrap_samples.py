import numpy as np
import pandas as pd
import pytest
from estimagic.config import DEFAULT_SEED
from estimagic.inference.bootstrap_samples import _convert_cluster_ids_to_indices
from estimagic.inference.bootstrap_samples import _get_bootstrap_samples_from_indices
from estimagic.inference.bootstrap_samples import get_bootstrap_indices
from estimagic.inference.bootstrap_samples import get_bootstrap_samples
from numpy.testing import assert_array_equal as aae
from pandas.testing import assert_frame_equal as afe


RNG = np.random.default_rng(DEFAULT_SEED)


@pytest.fixture
def data():
    df = pd.DataFrame()
    df["id"] = np.arange(900)
    df["hh"] = [3, 1, 2, 0, 0, 2, 5, 4, 5] * 100
    return df


def test_get_bootstrap_indices_randomization_works_without_clustering(data):
    res = get_bootstrap_indices(data, n_draws=2, rng=RNG)
    assert set(res[0]) != set(res[1])


def test_get_bootstrap_indices_radomization_works_with_clustering(data):
    res = get_bootstrap_indices(data, cluster_by="hh", n_draws=2, rng=RNG)
    assert set(res[0]) != set(res[1])


def test_clustering_leaves_households_intact(data):
    indices = get_bootstrap_indices(data, cluster_by="hh", n_draws=1, rng=RNG)[0]
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
    get_bootstrap_samples(data, n_draws=2, rng=RNG)
