"""Tests for the logit example."""
from pathlib import Path

import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal

from estimagic.examples.logit import logit_loglike
from estimagic.examples.logit import logit_loglikeobs


@pytest.fixture()
def statsmodels_fixtures():
    fix_path = Path(__file__).resolve().parent / "logit_fixtures.pickle"
    fix = pd.read_pickle(fix_path)
    fix["params"].name = "value"
    fix["params"] = fix["params"].to_frame()
    return fix


def test_loglike(statsmodels_fixtures):
    fix = statsmodels_fixtures
    assert_array_almost_equal(
        logit_loglike(fix["params"], fix["y"], fix["x"]), fix["loglike"]
    )


def test_loglikeobs(statsmodels_fixtures):
    fix = statsmodels_fixtures
    assert_array_almost_equal(
        logit_loglikeobs(fix["params"], fix["y"], fix["x"]), fix["loglikeobs"]
    )
