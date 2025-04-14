import numpy as np
from numpy.testing import assert_array_almost_equal as aaae

import optimagic as om


def test_x0_works_in_minimize():
    res = om.minimize(
        fun=lambda x: x @ x,
        x0=np.arange(3),
        algorithm="nevergrad_oneplusone",
    )
    aaae(res.params, np.zeros(3))


def test_x0_works_in_maximize():
    res = om.maximize(
        fun=lambda x: -x @ x,
        x0=np.arange(3),
        algorithm="nevergrad_oneplusone",
    )
    aaae(res.params, np.zeros(3))
