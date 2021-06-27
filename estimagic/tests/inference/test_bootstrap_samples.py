import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal as aae
from pandas.testing import assert_frame_equal as afe

from estimagic.inference.bootstrap_samples import get_bootstrap_samples
from estimagic.inference.bootstrap_samples import get_cluster_index
from estimagic.inference.bootstrap_samples import get_seeds


@pytest.fixture
def setup():
    out = {}

    out["df"] = pd.DataFrame(
        np.array([[1, 10], [2, 7], [3, 6], [4, 5]]), columns=["x1", "x2"]
    )

    out["cluster_df"] = pd.DataFrame(
        np.array([[1, 10, 2], [2, 7, 2], [3, 6, 1], [4, 5, 2]]),
        columns=["x1", "x2", "stratum"],
    )

    out["seeds"] = [1, 2, 3, 4, 5]

    return out


@pytest.fixture
def expected():
    out = {}

    out["cluster_index"] = [np.array([0, 1, 3]), np.array([2])]

    return out


def test_get_seeds():

    seeds = get_seeds(15)

    assert len(seeds) == 15

    for i in range(15):
        np.random.seed(seeds[i])


def test_get_bootstrap_samples(setup, expected):
    sample_ids1 = get_bootstrap_samples(data=setup["df"], seeds=setup["seeds"])
    sample_ids2 = get_bootstrap_samples(data=setup["df"], seeds=setup["seeds"])

    samples1 = get_bootstrap_samples(
        data=setup["df"], seeds=setup["seeds"], return_samples=True
    )
    samples2 = get_bootstrap_samples(
        data=setup["df"], seeds=setup["seeds"], return_samples=True
    )

    for i in range(len(sample_ids1)):
        aae(sample_ids1[i], sample_ids2[i])

    for i in range(len(samples1)):

        afe(samples1[i], samples2[i])
        afe(samples1[i][samples1[i].isin(setup["df"])].dropna(), samples1[i])


def test_get_bootstrap_samples_cluster(setup, expected):
    sample_ids1 = get_bootstrap_samples(
        data=setup["cluster_df"], seeds=setup["seeds"], cluster_by="stratum"
    )
    sample_ids2 = get_bootstrap_samples(
        data=setup["cluster_df"], seeds=setup["seeds"], cluster_by="stratum"
    )
    samples1 = get_bootstrap_samples(
        data=setup["cluster_df"],
        seeds=setup["seeds"],
        cluster_by="stratum",
        return_samples=True,
    )
    samples2 = get_bootstrap_samples(
        data=setup["cluster_df"],
        seeds=setup["seeds"],
        cluster_by="stratum",
        return_samples=True,
    )

    for i in range(len(sample_ids1)):
        aae(sample_ids1[i], sample_ids2[i])

    for i in range(len(samples1)):
        afe(samples1[i], samples2[i])


def test_get_cluster_index(setup, expected):
    cluster_index = get_cluster_index(setup["cluster_df"], cluster_by="stratum")
    for i in range(len(cluster_index)):
        aae(cluster_index[i], expected["cluster_index"][i])
