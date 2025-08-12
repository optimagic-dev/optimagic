import numpy as np
import pytest

from optimagic.config import IS_GRADIENT_FREE_OPTIMIZERS_INSTALLED
from optimagic.optimization.internal_optimization_problem import (
    SphereExampleInternalOptimizationProblemWithConverter,
)
from optimagic.optimizers.gfo_optimizers import (
    GFOCommonOptions,
    _get_gfo_constraints,
    _get_initialize_gfo,
    _get_search_space_gfo,
    _gfo_internal,
    _value2para,
)
from optimagic.parameters.bounds import Bounds

problem = SphereExampleInternalOptimizationProblemWithConverter()


def test_get_gfo_constraints():
    got = _get_gfo_constraints()
    expected = []
    assert got == expected


def test_get_initialize_gfo():
    x0 = np.array([1, 0, 1])
    x1 = [
        {"x0": 1, "x1": 2, "x2": 3},
    ]
    n_init = 20
    got = _get_initialize_gfo(x0, n_init, x1, problem.converter)
    expected = {
        "warm_start": [
            {"x0": 1, "x1": 0, "x2": 1},  # x0
            {"x0": 1, "x1": 2, "x2": 3},
        ],  # x1
        "vertices": n_init,
    }
    assert got == expected


def test_get_search_space_gfo():
    bounds = Bounds(lower=np.array([-10, -10]), upper=np.array([10, 10]))
    n_grid_points = {
        "x0": 4,
        "x1": 4,
    }
    got = _get_search_space_gfo(bounds, n_grid_points, problem.converter)
    expected = {
        "x0": np.array([-10.0, -5.0, 0.0, 5.0]),
        "x1": np.array([-10.0, -5.0, 0.0, 5.0]),
    }
    assert len(got.keys()) == 2
    assert np.all(got["x0"] == expected["x0"])
    assert np.all(got["x1"] == expected["x1"])


def test_value2para():
    assert _value2para(np.array([0, 1, 2])) == {"x0": 0, "x1": 1, "x2": 2}


@pytest.mark.skipif(
    not IS_GRADIENT_FREE_OPTIMIZERS_INSTALLED, reason="gfo not installed"
)
def test_gfo_internal():
    from gradient_free_optimizers import DownhillSimplexOptimizer

    res = _gfo_internal(
        common_options=GFOCommonOptions(),
        problem=problem,
        x0=np.full(10, 2),
        optimizer=DownhillSimplexOptimizer,
    )

    assert np.all(res.x == np.full(10, 0))
