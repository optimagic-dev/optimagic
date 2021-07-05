import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal as afe

from estimagic.inference.bootstrap import bootstrap_from_outcomes


@pytest.fixture
def setup():
    out = {}

    out["df"] = pd.DataFrame(
        np.array([[1, 10], [2, 7], [3, 6], [4, 5]]), columns=["x1", "x2"]
    )

    x = np.array([[2.0, 8.0], [2.0, 8.0], [2.5, 7.0], [3.0, 6.0], [3.25, 5.75]])
    out["estimates"] = pd.DataFrame(x, columns=["x1", "x2"])

    return out


@pytest.fixture
def expected():
    out = {}

    z = np.array([[2.55, 0.5701, 2, 3.225], [6.95, 1.0665, 5.775, 8]])

    out["results"] = pd.DataFrame(
        z, columns=["mean", "std", "lower_ci", "upper_ci"], index=["x1", "x2"]
    )

    return out


def g(data):
    return data.mean(axis=0)


def test_bootstrap_from_outcomes(setup, expected):

    results = bootstrap_from_outcomes(
        data=setup["df"],
        outcome=g,
        bootstrap_outcomes=setup["estimates"],
        ci_method="percentile",
    )

    # use rounding to adjust precision because there is no other way of handling this
    # such that it is compatible across all supported pandas versions.
    afe(results.round(2), expected["results"].round(2))
