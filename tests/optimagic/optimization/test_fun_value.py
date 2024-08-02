import numpy as np
import pytest
from numpy.testing import assert_almost_equal as aae
from optimagic.exceptions import InvalidFunctionError
from optimagic.optimization.fun_value import (
    FunctionValue,
    LeastSquaresFunctionValue,
    LikelihoodFunctionValue,
    ScalarFunctionValue,
    enforce_least_squares,
    enforce_likelihood,
    enforce_scalar,
)


def test_enforce_scalar_with_scalar_return():
    @enforce_scalar
    def f(x):
        return 3

    got = f(np.ones(3))
    assert isinstance(got, ScalarFunctionValue)
    assert got.value == 3


def test_enforce_scalar_with_function_value_return():
    @enforce_scalar
    def f(x):
        return FunctionValue(3)

    got = f(np.ones(3))
    assert isinstance(got, ScalarFunctionValue)
    assert got.value == 3


def test_enforce_scalar_trivial_case():
    @enforce_scalar
    def f(x):
        return ScalarFunctionValue(3)

    got = f(3)
    assert isinstance(got, ScalarFunctionValue)
    assert got.value == 3


def test_enforce_scalar_invalid_return():
    @enforce_scalar
    def f(x):
        return x

    with pytest.raises(InvalidFunctionError):
        f(np.ones(3))


def test_enforce_least_squares_with_vector_return():
    @enforce_least_squares
    def f(x):
        return np.ones(3)

    got = f(np.ones(3))
    assert isinstance(got, LeastSquaresFunctionValue)
    aae(got.value, np.ones(3))


def test_enforce_least_squares_with_function_value_return():
    @enforce_least_squares
    def f(x):
        return FunctionValue(np.ones(3))

    got = f(np.ones(3))
    assert isinstance(got, LeastSquaresFunctionValue)
    aae(got.value, np.ones(3))


def test_enforce_least_squares_trivial_case():
    @enforce_least_squares
    def f(x):
        return LeastSquaresFunctionValue(np.ones(3))

    got = f(np.ones(3))
    assert isinstance(got, LeastSquaresFunctionValue)
    aae(got.value, np.ones(3))


def test_enforce_least_squares_invalid_return():
    @enforce_least_squares
    def f(x):
        return 3

    with pytest.raises(InvalidFunctionError):
        f(np.ones(3))


def test_enforce_likelihood_with_vector_return():
    @enforce_likelihood
    def f(x):
        return np.ones(3)

    got = f(np.ones(3))
    assert isinstance(got, LikelihoodFunctionValue)
    aae(got.value, np.ones(3))


def test_enforce_likelihood_with_function_value_return():
    @enforce_likelihood
    def f(x):
        return FunctionValue(np.ones(3))

    got = f(np.ones(3))
    assert isinstance(got, LikelihoodFunctionValue)
    aae(got.value, np.ones(3))


def test_enforce_likelihood_trivial_case():
    @enforce_likelihood
    def f(x):
        return LikelihoodFunctionValue(np.ones(3))

    got = f(np.ones(3))
    assert isinstance(got, LikelihoodFunctionValue)
    aae(got.value, np.ones(3))


def test_enforce_likelihood_invalid_return():
    @enforce_likelihood
    def f(x):
        return 3

    with pytest.raises(InvalidFunctionError):
        f(np.ones(3))
