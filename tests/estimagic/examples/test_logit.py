"""Tests for the logit example."""

from estimagic.examples.logit import logit_grad, logit_hess, logit_jac, logit_loglike
from numpy.testing import assert_array_almost_equal as aaae


def test_logit_loglikes(logit_inputs, logit_object):
    x = logit_inputs["params"]["value"].to_numpy()
    expected = logit_object.loglikeobs(x)
    got = logit_loglike(**logit_inputs)

    aaae(got, expected)


def test_logit_jac(logit_inputs, logit_object):
    x = logit_inputs["params"]["value"].to_numpy()
    expected = logit_object.score_obs(x)

    got = logit_jac(**logit_inputs)

    aaae(got, expected)


def test_logit_grad(logit_inputs, logit_object):
    x = logit_inputs["params"]["value"].to_numpy()
    expected = logit_object.score(x)
    calculated = logit_grad(**logit_inputs)
    aaae(calculated, expected)


def test_logit_hessian(logit_inputs, logit_object):
    x = logit_inputs["params"]["value"].to_numpy()
    expected = logit_object.hessian(x)
    got = logit_hess(**logit_inputs)
    aaae(got, expected)
