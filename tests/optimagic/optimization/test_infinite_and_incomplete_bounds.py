import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from optimagic import mark
from optimagic.config import IS_NEVERGRAD_INSTALLED
from optimagic.optimization.optimize import minimize


@mark.least_squares
def sos(x):
    return x


@pytest.mark.skipif(
    not IS_NEVERGRAD_INSTALLED,
    reason="nevergrad not installed",
)
def test_no_bounds_with_nevergrad():
    res = minimize(
        fun=sos,
        params=np.arange(3),
        algorithm="nevergrad_cmaes",
        collect_history=True,
        skip_checks=True,
        algo_options={"seed": 12345, "stopping_maxfun": 10000},
    )
    aaae(res.x, np.zeros(3), 4)
