import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

import optimagic as om
from optimagic.exceptions import IncompleteBoundsError


def test_differential_evolution_respects_equality_constraint():
    # The constraint is not binding if it is treated as x[0] + x[1] >= -1.
    res = om.minimize(
        fun=lambda x: x @ x,
        params=np.array([1.0, 2.0]),
        algorithm=om.algos.scipy_differential_evolution(seed=0),
        bounds=om.Bounds(lower=np.full(2, -5.0), upper=np.full(2, 5.0)),
        constraints=om.NonlinearConstraint(func=lambda x: x[0] + x[1], value=-1.0),
    )
    aaae(res.params, np.array([-0.5, -0.5]), decimal=4)


def test_shgo_without_bounds_raises():
    with pytest.raises(IncompleteBoundsError):
        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0]),
            algorithm="scipy_shgo",
        )
