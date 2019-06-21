import os
import pickle

import pytest
from pandas.testing import assert_frame_equal

from estimagic.differentiation.hessian import hessian
from estimagic.examples.logit import logit_loglike


@pytest.fixture()
def statsmodels_fixtures():
    with open(
        os.path.join(os.path.dirname(__file__), "diff_fixtures.pickle"), "rb"
    ) as p:
        fix = pickle.load(p)
    return fix


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
