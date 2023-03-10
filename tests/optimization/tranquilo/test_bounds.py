import numpy as np
import pytest
from estimagic.optimization.tranquilo.bounds import Bounds, _any_finite

CASES = [
    (np.array([1, 2]), np.array([5, 6]), True),
    (np.array([1, 2]), None, True),
    (None, np.array([5, 6]), True),
    (None, None, False),
    (np.array([np.inf, np.inf]), np.array([np.inf, np.inf]), False),
    (np.array([-np.inf, -np.inf]), np.array([np.inf, np.inf]), False),
    (np.array([1, 2]), np.array([np.inf, np.inf]), True),
]


@pytest.mark.parametrize("lb, ub, exp", CASES)
def test_any_finite_true(lb, ub, exp):
    out = _any_finite(lb, ub)
    assert out is exp


def test_bounds_none():
    bounds = Bounds(lower=None, upper=None)
    assert bounds.has_any is False


def test_bounds_inifinite():
    lb = np.array([np.inf, np.inf])
    ub = np.array([np.inf, np.inf])
    bounds = Bounds(lower=lb, upper=ub)
    assert bounds.has_any is False


def test_bounds_finite():
    lb = np.array([1, 2])
    ub = np.array([5, 6])
    bounds = Bounds(lower=lb, upper=ub)
    assert bounds.has_any is True
