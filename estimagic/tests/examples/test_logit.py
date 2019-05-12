"""Tests for the logit example."""
import os
import pickle
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from estimagic.examples.logit import logit_gradient
from estimagic.examples.logit import logit_hessian
from estimagic.examples.logit import logit_jacobian
from estimagic.examples.logit import logit_loglike
from estimagic.examples.logit import logit_loglikeobs
from estimagic.differentiation.gradient import gradient

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
    assert_array_almost_equal(
        gradient(logit_loglike, fix["params"], method='forward',
                 func_args=[fix["y"], fix["x"]]),
        fix["gradient"], decimal=4
    )


def test_gradient_forward_richardson(statsmodels_fixtures):
    fix = statsmodels_fixtures
    assert_array_almost_equal(
        gradient(logit_loglike, fix["params"], method='forward',
                 extrapolant='richardson',
                 func_args=[fix["y"], fix["x"]]),
        fix["gradient"]
    )


def test_gradient_backward(statsmodels_fixtures):
    fix = statsmodels_fixtures
    assert_array_almost_equal(
        gradient(logit_loglike, fix["params"], method='backward',
                 func_args=[fix["y"], fix["x"]]),
        fix["gradient"], decimal=4
    )


def test_gradient_backward_richardson(statsmodels_fixtures):
    fix = statsmodels_fixtures
    assert_array_almost_equal(
        gradient(logit_loglike, fix["params"], method='backward',
                 extrapolant='richardson',
                 func_args=[fix["y"], fix["x"]]),
        fix["gradient"], decimal=4
    )


def test_gradient_central(statsmodels_fixtures):
    fix = statsmodels_fixtures
    assert_array_almost_equal(
        gradient(logit_loglike, fix["params"], method='central',
                 func_args=[fix["y"], fix["x"]]),
        fix["gradient"]
    )


def test_gradient_central_richard(statsmodels_fixtures):
    fix = statsmodels_fixtures
    assert_array_almost_equal(
        gradient(logit_loglike, fix["params"], method='central',
                 extrapolant='richardson',
                 func_args=[fix["y"], fix["x"]]),
        fix["gradient"]
    )

def test_jacobian(statsmodels_fixtures):
    fix = statsmodels_fixtures
    assert_array_almost_equal(
        logit_jacobian(fix["params"], fix["y"], fix["x"]), fix["jacobian"]
    )


def test_hessian(statsmodels_fixtures):
    fix = statsmodels_fixtures
    assert_array_almost_equal(
        logit_hessian(fix["params"], fix["y"], fix["x"]), fix["hessian"]
    )
