import os
import pickle

import pytest
from pandas.testing import assert_frame_equal

from estimagic.differentiation.jacobian import jacobian
from estimagic.examples.logit import logit_loglikeobs


@pytest.fixture()
def statsmodels_fixtures():
    with open(
        os.path.join(os.path.dirname(__file__), "diff_fixtures.pickle"), "rb"
    ) as p:
        fix = pickle.load(p)
    return fix


def test_jacobian_central(statsmodels_fixtures):
    fix = statsmodels_fixtures
    assert_frame_equal(
        jacobian(
            logit_loglikeobs,
            fix["params"],
            method="central",
            func_args=[fix["y"], fix["x"]],
        ),
        fix["jacobian"],
    )


def test_jacobian_forward(statsmodels_fixtures):
    fix = statsmodels_fixtures
    assert_frame_equal(
        jacobian(
            logit_loglikeobs,
            fix["params"],
            method="forward",
            func_args=[fix["y"], fix["x"]],
        ),
        fix["jacobian"],
    )


def test_jacobian_backward(statsmodels_fixtures):
    fix = statsmodels_fixtures
    assert_frame_equal(
        jacobian(
            logit_loglikeobs,
            fix["params"],
            method="backward",
            func_args=[fix["y"], fix["x"]],
        ),
        fix["jacobian"],
    )
