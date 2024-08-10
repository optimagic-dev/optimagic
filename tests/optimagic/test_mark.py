import functools
from dataclasses import dataclass

import optimagic as om
import pytest
from optimagic.optimization.algorithm import AlgoInfo, Algorithm
from optimagic.typing import AggregationLevel


def f(x):
    pass


@dataclass(frozen=True)
class ImmutableF:
    def __call__(self, x):
        pass


def _g(x, y):
    pass


g = functools.partial(_g, y=1)


CALLABLES = [f, ImmutableF(), g]


@pytest.mark.parametrize("func", CALLABLES)
def test_scalar(func):
    got = om.mark.scalar(func)

    assert got._problem_type == AggregationLevel.SCALAR


@pytest.mark.parametrize("func", CALLABLES)
def test_least_squares(func):
    got = om.mark.least_squares(func)

    assert got._problem_type == AggregationLevel.LEAST_SQUARES


@pytest.mark.parametrize("func", CALLABLES)
def test_likelihood(func):
    got = om.mark.likelihood(func)

    assert got._problem_type == AggregationLevel.LIKELIHOOD


def test_mark_minimizer():
    @om.mark.minimizer(
        name="test",
        solver_type=AggregationLevel.LEAST_SQUARES,
        is_available=True,
        is_global=True,
        needs_jac=True,
        needs_hess=True,
        supports_parallelism=True,
        supports_bounds=True,
        supports_linear_constraints=True,
        supports_nonlinear_constraints=True,
        disable_history=False,
    )
    @dataclass(frozen=True)
    class DummyAlgorithm(Algorithm):
        initial_radius: float = 1.0
        max_radius: float = 10.0
        convergence_ftol_rel: float = 1e-6
        stopping_maxiter: int = 1000

        def _solve_internal_problem(self, problem, x0):
            pass

    assert hasattr(DummyAlgorithm, "__algo_info__")
    assert isinstance(DummyAlgorithm.__algo_info__, AlgoInfo)
    assert DummyAlgorithm.__algo_info__.name == "test"
