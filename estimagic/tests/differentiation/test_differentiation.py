import os
import pickle
from itertools import product

import pytest
from pandas.testing import assert_frame_equal
from pandas.testing import assert_series_equal

from estimagic.differentiation.differentiation import gradient
from estimagic.differentiation.differentiation import hessian
from estimagic.differentiation.differentiation import jacobian
from estimagic.examples.logit import logit_loglike
from estimagic.examples.logit import logit_loglikeobs


@pytest.fixture()
def statsmodels_fixtures():
    fix_path = os.path.join(os.path.dirname(__file__), "diff_fixtures.pickle")
    with open(fix_path, "rb") as p:
        fix = pickle.load(p)

    fix["params"] = fix["params"].to_frame()
    fix["gradient"].name = "gradient"
    return fix


to_test = list(product(["forward", "central", "backward"], [True, False]))


@pytest.mark.parametrize("method, extrapolation", to_test)
def test_gradient(statsmodels_fixtures, method, extrapolation):
    fix = statsmodels_fixtures
    func_args = [fix["y"], fix["x"]]
    calculated = gradient(
        logit_loglike,
        fix["params"],
        method="forward",
        extrapolation=False,
        func_args=func_args,
    )
    expected = fix["gradient"]
    assert_series_equal(calculated, expected)


@pytest.mark.parametrize("method, extrapolation", to_test)
def test_jacobian(statsmodels_fixtures, method, extrapolation):
    fix = statsmodels_fixtures
    func_kwargs = {"y": fix["y"], "x": fix["x"]}
    calculated = jacobian(
        logit_loglikeobs,
        params=fix["params"],
        method=method,
        extrapolation=extrapolation,
        func_kwargs=func_kwargs,
    )

    expected = fix["jacobian"]
    assert_frame_equal(calculated, expected)


to_test_hess = [("central", True), ("central", False)]


@pytest.mark.parametrize("method, extrapolation", to_test_hess)
def test_hessian(statsmodels_fixtures, method, extrapolation):
    fix = statsmodels_fixtures
    calculated = hessian(
        logit_loglike,
        fix["params"],
        method=method,
        extrapolation=extrapolation,
        func_kwargs={"y": fix["y"], "x": fix["x"]},
    )
    expected = fix["hessian"]
    assert_frame_equal(calculated, expected)
