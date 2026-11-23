import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from optimagic import mark
from optimagic.optimization.optimize import minimize


@mark.least_squares
def sos(x):
    return x


@pytest.mark.nevergrad
def test_no_bounds_with_nevergrad():
    res = minimize(
        fun=sos,
        params=np.arange(3),
        algorithm="nevergrad_cmaes",
        collect_history=True,
        skip_checks=True,
        algo_options={"seed": 12345, "stopping_maxfun": 10000},
    )
    # The test checks that nevergrad works without bounds. CMA-ES ends between 2e-5
    # and 0.2 away from the optimum, depending on the seed and the platform.
    aaae(res.x, np.zeros(3), 2)
