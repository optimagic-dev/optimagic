"""Test suite for the iminuit migrad optimizer."""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from optimagic.config import IS_IMINUIT_INSTALLED
from optimagic.optimization.optimize import minimize
from optimagic.optimizers.iminuit_migrad import (
    IminuitMigrad,
    _convert_bounds_to_minuit_limits,
)


def sphere(x):
    return (x**2).sum()


def sphere_grad(x):
    return 2 * x


def test_convert_bounds_unbounded():
    """Test converting unbounded bounds."""
    lower = np.array([-np.inf, -np.inf])
    upper = np.array([np.inf, np.inf])
    limits = _convert_bounds_to_minuit_limits(lower, upper)

    assert len(limits) == 2
    assert limits[0] == (None, None)
    assert limits[1] == (None, None)


def test_convert_bounds_lower_only():
    """Test converting lower bounds only."""
    lower = np.array([1.0, 2.0])
    upper = np.array([np.inf, np.inf])
    limits = _convert_bounds_to_minuit_limits(lower, upper)

    assert len(limits) == 2
    assert limits[0] == (1.0, None)
    assert limits[1] == (2.0, None)


def test_convert_bounds_upper_only():
    """Test converting upper bounds only."""
    lower = np.array([-np.inf, -np.inf])
    upper = np.array([1.0, 2.0])
    limits = _convert_bounds_to_minuit_limits(lower, upper)

    assert len(limits) == 2
    assert limits[0] == (None, 1.0)
    assert limits[1] == (None, 2.0)


def test_convert_bounds_two_sided():
    """Test converting two-sided bounds."""
    lower = np.array([1.0, -2.0])
    upper = np.array([2.0, -1.0])
    limits = _convert_bounds_to_minuit_limits(lower, upper)

    assert len(limits) == 2
    assert limits[0] == (1.0, 2.0)
    assert limits[1] == (-2.0, -1.0)


def test_convert_bounds_mixed():
    """Test converting mixed bounds (some infinite, some finite)."""
    lower = np.array([-np.inf, 0.0, 1.0])
    upper = np.array([1.0, np.inf, 2.0])
    limits = _convert_bounds_to_minuit_limits(lower, upper)

    assert len(limits) == 3
    assert limits[0] == (None, 1.0)
    assert limits[1] == (0.0, None)
    assert limits[2] == (1.0, 2.0)


@pytest.mark.skipif(not IS_IMINUIT_INSTALLED, reason="iminuit not installed.")
def test_iminuit_migrad():
    """Test basic optimization with sphere function."""
    x0 = np.array([1.0, 2.0, 3.0])
    algorithm = IminuitMigrad()

    res = minimize(
        fun=sphere,
        jac=sphere_grad,
        algorithm=algorithm,
        x0=x0,
    )

    assert res.success
    aaae(res.x, np.zeros(3), decimal=6)
    assert res.n_fun_evals > 0
    assert res.n_jac_evals > 0
