"""Tests for the logit example."""
import os
import pickle

import pytest
from numpy.testing import assert_array_almost_equal
from pandas.testing import assert_frame_equal
from pandas.testing import assert_series_equal

from estimagic.differentiation.gradient import gradient
from estimagic.differentiation.hessian import hessian
from estimagic.differentiation.jacobian import jacobian
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


def test_gradient_forward(statsmodels_fixtures):
    fix = statsmodels_fixtures
    assert_series_equal(
        gradient(
            logit_loglike,
            fix["params"],
            method="forward",
            func_args=[fix["y"], fix["x"]],
        ),
        fix["gradient"],
    )


def test_gradient_backward(statsmodels_fixtures):
    fix = statsmodels_fixtures
    assert_series_equal(
        gradient(
            logit_loglike,
            fix["params"],
            method="backward",
            func_args=[fix["y"], fix["x"]],
        ),
        fix["gradient"],
    )


def test_gradient_central(statsmodels_fixtures):
    fix = statsmodels_fixtures
    assert_array_almost_equal(
        gradient(
            logit_loglike,
            fix["params"],
            method="central",
            func_args=[fix["y"], fix["x"]],
        ),
        fix["gradient"],
    )


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


def test_hessian_central(statsmodels_fixtures):
    fix = statsmodels_fixtures
    assert_frame_equal(
        hessian(
            logit_loglike,
            fix["params"],
            method="central",
            func_args=[fix["y"], fix["x"]],
        ),
        fix["hessian"],
    )
