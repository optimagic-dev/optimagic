import optimagic as om
from optimagic.typing import ProblemType


def test_scalar():
    @om.mark.scalar
    def f(x):
        pass

    assert f._problem_type == ProblemType.SCALAR


def test_least_squares():
    @om.mark.least_squares
    def f(x):
        pass

    assert f._problem_type == ProblemType.LEAST_SQUARES


def test_likelihood():
    @om.mark.likelihood
    def f(x):
        pass

    assert f._problem_type == ProblemType.LIKELIHOOD
