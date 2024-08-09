import functools
from dataclasses import dataclass

import optimagic as om
import pytest
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
