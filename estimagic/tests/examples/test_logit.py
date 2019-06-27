"""Tests for the logit example."""
import os
import pickle

import pytest
from numpy.testing import assert_array_almost_equal

from estimagic.examples.logit import logit_loglike
from estimagic.examples.logit import logit_loglikeobs


@pytest.fixture()
def statsmodels_fixtures():
    with open(
        os.path.join(os.path.dirname(__file__), "logit_fixtures.pickle"), "rb"
    ) as p:
        fix = pickle.load(p)
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
