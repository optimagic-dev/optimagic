import os
import pickle

import pytest
from pandas.testing import assert_frame_equal

from estimagic.differentiation.differentiation import jacobian
from estimagic.examples.logit import logit_loglikeobs


@pytest.fixture()
def statsmodels_fixtures():
    with open(
        os.path.join(os.path.dirname(__file__), "diff_fixtures.pickle"), "rb"
    ) as p:
        fix = pickle.load(p)

    fix["params"].name = "value"
    fix["params"] = fix["params"].to_frame()
    return fix


def test_jacobian_central(statsmodels_fixtures):
    fix = statsmodels_fixtures
    assert_frame_equal(
        jacobian(
            logit_loglikeobs,
            fix["params"],
            method="central",
            extrapolation=False,
            func_args=[fix["y"], fix["x"]],
        ),
        fix["jacobian"],
    )


def test_jacobian_central_richardson(statsmodels_fixtures):
    fix = statsmodels_fixtures
    assert_frame_equal(
        jacobian(
            logit_loglikeobs,
            fix["params"],
            method="central",
            extrapolation=True,
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
            extrapolation=False,
            func_args=[fix["y"], fix["x"]],
        ),
        fix["jacobian"],
    )


def test_jacobian_forward_richardson(statsmodels_fixtures):
    fix = statsmodels_fixtures
    assert_frame_equal(
        jacobian(
            logit_loglikeobs,
            fix["params"],
            method="forward",
            extrapolation=True,
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
            extrapolation=False,
            func_args=[fix["y"], fix["x"]],
        ),
        fix["jacobian"],
    )


def test_jacobian_backward_richardson(statsmodels_fixtures):
    fix = statsmodels_fixtures
    assert_frame_equal(
        jacobian(
            logit_loglikeobs,
            fix["params"],
            method="backward",
            extrapolation=True,
            func_args=[fix["y"], fix["x"]],
        ),
        fix["jacobian"],
    )


def test_jacobian_backward_richardson_kwargs(statsmodels_fixtures):
    fix = statsmodels_fixtures
    assert_frame_equal(
        jacobian(
            logit_loglikeobs,
            fix["params"],
            method="backward",
            extrapolation=True,
            func_kwargs={"y": fix["y"], "x": fix["x"]},
        ),
        fix["jacobian"],
    )
