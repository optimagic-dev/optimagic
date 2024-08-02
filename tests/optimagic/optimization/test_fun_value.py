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
from optimagic.typing import OptimizerType


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


SCALAR_VALUES = [
    ScalarFunctionValue(5),
]

LS_VALUES = [
    LeastSquaresFunctionValue(np.array([1, 2])),
    LeastSquaresFunctionValue({"a": 1, "b": 2}),
]

LIKELIHOOD_VALUES = [
    LikelihoodFunctionValue(np.array([1, 4])),
    LikelihoodFunctionValue({"a": 1, "b": 4}),
]


@pytest.mark.parametrize("value", SCALAR_VALUES + LS_VALUES + LIKELIHOOD_VALUES)
def test_values_for_scalar_optimizers(value):
    got = value.internal_value(OptimizerType.SCALAR)
    assert isinstance(got, float)
    assert got == 5.0


@pytest.mark.parametrize("value", LS_VALUES)
def test_values_for_least_squares_optimizers(value):
    got = value.internal_value(OptimizerType.LEAST_SQUARES)
    assert isinstance(got, np.ndarray)
    assert got.dtype == np.float64
    aae(got, np.array([1.0, 2]))


@pytest.mark.parametrize("value", LS_VALUES + LIKELIHOOD_VALUES)
def test_values_for_likelihood_optimizers(value):
    got = value.internal_value(OptimizerType.LIKELIHOOD)
    assert isinstance(got, np.ndarray)
    assert got.dtype == np.float64
    aae(got, np.array([1.0, 4]))


@pytest.mark.parametrize("value", SCALAR_VALUES + LIKELIHOOD_VALUES)
def test_invalid_values_for_least_squares_optimizers(value):
    with pytest.raises(InvalidFunctionError):
        SCALAR_VALUES[0].internal_value(OptimizerType.LEAST_SQUARES)


@pytest.mark.parametrize("value", SCALAR_VALUES)
def test_invalid_values_for_likelihood_optimizers(value):
    with pytest.raises(InvalidFunctionError):
        SCALAR_VALUES[0].internal_value(OptimizerType.LIKELIHOOD)
