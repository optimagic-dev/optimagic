"""Tests for the logit example."""
import os
import pickle
import pytest
from numpy.testing import assert_array_almost_equal
from estimagic.examples.logit import logit_loglike
from estimagic.examples.logit import logit_loglikeobs
from estimagic.differentiation.gradient import gradient
from estimagic.differentiation.jacobian import jacobian
from estimagic.differentiation.hessian import hessian


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
        gradient(
            logit_loglike,
            fix["params"],
            method="forward",
            func_args=[fix["y"], fix["x"]],
        ),
        fix["gradient"],
        decimal=4,
    )


def test_gradient_forward_richardson(statsmodels_fixtures):
    fix = statsmodels_fixtures
    assert_array_almost_equal(
        gradient(
            logit_loglike,
            fix["params"],
            method="forward",
            extrapolant="richardson",
            func_args=[fix["y"], fix["x"]],
        ),
        fix["gradient"],
    )


def test_gradient_backward(statsmodels_fixtures):
    fix = statsmodels_fixtures
    assert_array_almost_equal(
        gradient(
            logit_loglike,
            fix["params"],
            method="backward",
            func_args=[fix["y"], fix["x"]],
        ),
        fix["gradient"],
        decimal=4,
    )


def test_gradient_backward_richardson(statsmodels_fixtures):
    fix = statsmodels_fixtures
    assert_array_almost_equal(
        gradient(
            logit_loglike,
            fix["params"],
            method="backward",
            extrapolant="richardson",
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


def test_gradient_central_richardson(statsmodels_fixtures):
    fix = statsmodels_fixtures
    assert_array_almost_equal(
        gradient(
            logit_loglike,
            fix["params"],
            method="central",
            extrapolant="richardson",
            func_args=[fix["y"], fix["x"]],
        ),
        fix["gradient"],
    )


def test_jacobian_central(statsmodels_fixtures):
    fix = statsmodels_fixtures
    assert_array_almost_equal(
        jacobian(
            logit_loglikeobs,
            fix["params"],
            method="central",
            func_args=[fix["y"], fix["x"]],
        ),
        fix["jacobian"], decimal=8
    )


def test_jacobian_central_richardson(statsmodels_fixtures):
    fix = statsmodels_fixtures
    assert_array_almost_equal(
        jacobian(
            logit_loglikeobs,
            fix["params"],
            method="central",
            extrapolant="richardson",
            func_args=[fix["y"], fix["x"]],
        ),
        fix["jacobian"], decimal=7
    )


def test_jacobian_forward(statsmodels_fixtures):
    fix = statsmodels_fixtures
    assert_array_almost_equal(
        jacobian(
            logit_loglikeobs,
            fix["params"],
            method="forward",
            func_args=[fix["y"], fix["x"]],
        ),
        fix["jacobian"],

    )


def test_jacobian_forward_richardson(statsmodels_fixtures):
    fix = statsmodels_fixtures
    assert_array_almost_equal(
        jacobian(
            logit_loglikeobs,
            fix["params"],
            method="forward",
            extrapolant="richardson",
            func_args=[fix["y"], fix["x"]],
        ),
        fix["jacobian"], decimal=7
    )


def test_jacobian_backward(statsmodels_fixtures):
    fix = statsmodels_fixtures
    assert_array_almost_equal(
        jacobian(
            logit_loglikeobs,
            fix["params"],
            method="backward",
            func_args=[fix["y"], fix["x"]],
        ),
        fix["jacobian"],
        decimal=5,
    )


def test_jacobian_backward_richardson(statsmodels_fixtures):
    fix = statsmodels_fixtures
    assert_array_almost_equal(
        jacobian(
            logit_loglikeobs,
            fix["params"],
            method="backward",
            extrapolant="richardson",
            func_args=[fix["y"], fix["x"]],
        ),
        fix["jacobian"],
    )


def test_hessian_central(statsmodels_fixtures):
    fix = statsmodels_fixtures
    assert_array_almost_equal(
        hessian(
            logit_loglike,
            fix["params"],
            method="central",
            func_args=[fix["y"], fix["x"]],
        ),
        fix["hessian"],
        decimal=5,
    )


def test_hessian_forward(statsmodels_fixtures):
    fix = statsmodels_fixtures
    assert_array_almost_equal(
        hessian(
            logit_loglike,
            fix["params"],
            method="forward",
            func_args=[fix["y"], fix["x"]],
        ),
        fix["hessian"],
        decimal=1,
    )


def test_hessian_backward(statsmodels_fixtures):
    fix = statsmodels_fixtures
    assert_array_almost_equal(
        hessian(
            logit_loglike,
            fix["params"],
            method="backward",
            func_args=[fix["y"], fix["x"]],
        ),
        fix["hessian"],
        decimal=1,
    )


def test_hessian_central_richardson(statsmodels_fixtures):
    fix = statsmodels_fixtures
    assert_array_almost_equal(
        hessian(
            logit_loglike,
            fix["params"],
            method="central",
            extrapolant='richardson',
            func_args=[fix["y"], fix["x"]],
        ), fix["hessian"], decimal=4
    )


def test_hessian_backward_richardson(statsmodels_fixtures):
    fix = statsmodels_fixtures
    assert_array_almost_equal(
        hessian(
            logit_loglike,
            fix["params"],
            method="backward",
            extrapolant='richardson',
            func_args=[fix["y"], fix["x"]],
        ),
        fix["hessian"],
        decimal=3,
    )


def test_hessian_forward_richardson(statsmodels_fixtures):
    fix = statsmodels_fixtures
    assert_array_almost_equal(
        hessian(
            logit_loglike,
            fix["params"],
            method="forward",
            extrapolant='richardson',
            func_args=[fix["y"], fix["x"]],
        ),
        fix["hessian"],
        decimal=3,
    )
