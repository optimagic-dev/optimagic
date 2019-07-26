import os
import pickle

import pytest
from pandas.testing import assert_series_equal

from estimagic.differentiation.differentiation import gradient
from estimagic.examples.logit import logit_loglike


@pytest.fixture()
def statsmodels_fixtures():
    with open(
        os.path.join(os.path.dirname(__file__), "diff_fixtures.pickle"), "rb"
    ) as p:
        fix = pickle.load(p)

    fix["params"].name = "value"
    fix["params"] = fix["params"].to_frame()
    fix["gradient"].name = "gradient"
    return fix


def test_gradient_forward(statsmodels_fixtures):
    fix = statsmodels_fixtures
    assert_series_equal(
        gradient(
            logit_loglike,
            fix["params"],
            method="forward",
            extrapolation=False,
            func_args=[fix["y"], fix["x"]],
        ),
        fix["gradient"],
    )


def test_gradient_forward_richardson(statsmodels_fixtures):
    fix = statsmodels_fixtures
    assert_series_equal(
        gradient(
            logit_loglike,
            fix["params"],
            method="forward",
            extrapolation=True,
            func_args=[fix["y"], fix["x"]],
        ),
        fix["gradient"],
    )


def test_gradient_forward_richardson_kwargs(statsmodels_fixtures):
    fix = statsmodels_fixtures
    assert_series_equal(
        gradient(
            logit_loglike,
            fix["params"],
            method="forward",
            extrapolation=True,
            func_kwargs={"y": fix["y"], "x": fix["x"]},
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
            extrapolation=False,
            func_args=[fix["y"], fix["x"]],
        ),
        fix["gradient"],
    )


def test_gradient_backward_richardson(statsmodels_fixtures):
    fix = statsmodels_fixtures
    assert_series_equal(
        gradient(
            logit_loglike,
            fix["params"],
            method="backward",
            extrapolation=True,
            func_args=[fix["y"], fix["x"]],
        ),
        fix["gradient"],
    )


def test_gradient_central(statsmodels_fixtures):
    fix = statsmodels_fixtures
    assert_series_equal(
        gradient(
            logit_loglike,
            fix["params"],
            method="central",
            extrapolation=False,
            func_args=[fix["y"], fix["x"]],
        ),
        fix["gradient"],
    )


def test_gradient_central_richardson(statsmodels_fixtures):
    fix = statsmodels_fixtures
    assert_series_equal(
        gradient(
            logit_loglike,
            fix["params"],
            method="central",
            extrapolation=True,
            func_args=[fix["y"], fix["x"]],
        ),
        fix["gradient"],
    )
