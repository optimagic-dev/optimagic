import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal as afe

from estimagic.inference.bootstrap_estimates import _get_clustered_estimates
from estimagic.inference.bootstrap_estimates import _get_uniform_estimates
from estimagic.inference.bootstrap_estimates import get_bootstrap_estimates


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


def g(data):
    return data.mean(axis=0)


def test_get_bootstrap_estimates(setup):
    estimates1 = get_bootstrap_estimates(data=setup["df"], f=g, seeds=setup["seeds"])
    estimates2 = get_bootstrap_estimates(
        data=setup["df"], f=g, seeds=setup["seeds"], n_cores=-1
    )
    estimates3 = pd.DataFrame(
        _get_uniform_estimates(data=setup["df"], f=g, seeds=setup["seeds"])
    )
    estimates4 = pd.DataFrame(
        _get_uniform_estimates(data=setup["df"], f=g, seeds=setup["seeds"])
    )

    afe(estimates1, estimates2)
    afe(estimates2, estimates3)
    afe(estimates3, estimates4)


def test_get_bootstrap_estimates_cluster(setup):
    estimates1 = get_bootstrap_estimates(
        data=setup["cluster_df"], f=g, cluster_by="stratum", seeds=setup["seeds"]
    )
    estimates2 = get_bootstrap_estimates(
        data=setup["cluster_df"], f=g, cluster_by="stratum", seeds=setup["seeds"]
    )
    estimates3 = pd.DataFrame(
        _get_clustered_estimates(
            data=setup["cluster_df"], f=g, cluster_by="stratum", seeds=setup["seeds"]
        )
    )
    estimates4 = pd.DataFrame(
        _get_clustered_estimates(
            data=setup["cluster_df"], f=g, cluster_by="stratum", seeds=setup["seeds"]
        )
    )

    afe(estimates1, estimates2)
    afe(estimates2, estimates3)
    afe(estimates3, estimates4)
