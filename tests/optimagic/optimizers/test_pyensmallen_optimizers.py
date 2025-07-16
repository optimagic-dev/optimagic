"""Tests for pyensmallen optimizers."""

import numpy as np
import pytest

import optimagic as om
from optimagic.config import IS_PYENSMALLEN_INSTALLED
from optimagic.optimization.optimize import minimize


@pytest.mark.skipif(not IS_PYENSMALLEN_INSTALLED, reason="pyensmallen not installed.")
def test_stop_after_one_iteration():
    algo = om.algos.ensmallen_lbfgs(stopping_maxiter=1)
    expected = np.array([0, 0.81742581, 1.63485163, 2.45227744, 3.26970326])
    res = minimize(
        fun=lambda x: x @ x,
        fun_and_jac=lambda x: (x @ x, 2 * x),
        params=np.arange(5),
        algorithm=algo,
    )

    assert np.allclose(res.x, expected)
