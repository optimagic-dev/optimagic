import numpy as np
import pytest
from numpy.testing import assert_almost_equal as aae

from optimagic.exceptions import InvalidFunctionError
from optimagic.optimization.fun_value import (
    FunctionValue,
    LeastSquaresFunctionValue,
    LikelihoodFunctionValue,
    ScalarFunctionValue,
    enforce_return_type,
    enforce_return_type_with_jac,
)
from optimagic.typing import AggregationLevel

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
    got = value.internal_value(AggregationLevel.SCALAR)
    assert isinstance(got, float)
    assert got == 5.0


@pytest.mark.parametrize("value", LS_VALUES)
def test_values_for_least_squares_optimizers(value):
    got = value.internal_value(AggregationLevel.LEAST_SQUARES)
    assert isinstance(got, np.ndarray)
    assert got.dtype == np.float64
    aae(got, np.array([1.0, 2]))


@pytest.mark.parametrize("value", LS_VALUES + LIKELIHOOD_VALUES)
def test_values_for_likelihood_optimizers(value):
    got = value.internal_value(AggregationLevel.LIKELIHOOD)
    assert isinstance(got, np.ndarray)
    assert got.dtype == np.float64
    aae(got, np.array([1.0, 4]))


@pytest.mark.parametrize("value", SCALAR_VALUES + LIKELIHOOD_VALUES)
def test_invalid_values_for_least_squares_optimizers(value):
    with pytest.raises(InvalidFunctionError):
        SCALAR_VALUES[0].internal_value(AggregationLevel.LEAST_SQUARES)


@pytest.mark.parametrize("value", SCALAR_VALUES)
def test_invalid_values_for_likelihood_optimizers(value):
    with pytest.raises(InvalidFunctionError):
        SCALAR_VALUES[0].internal_value(AggregationLevel.LIKELIHOOD)


def test_enforce_scalar_with_scalar_return():
    @enforce_return_type(AggregationLevel.SCALAR)
    def f(x):
        return 3

    got = f(np.ones(3))
    assert isinstance(got, ScalarFunctionValue)
    assert got.value == 3


def test_enforce_scalar_with_function_value_return():
    @enforce_return_type(AggregationLevel.SCALAR)
    def f(x):
        return FunctionValue(3)

    got = f(np.ones(3))
    assert isinstance(got, ScalarFunctionValue)
    assert got.value == 3


def test_enforce_scalar_trivial_case():
    @enforce_return_type(AggregationLevel.SCALAR)
    def f(x):
        return ScalarFunctionValue(3)

    got = f(3)
    assert isinstance(got, ScalarFunctionValue)
    assert got.value == 3


def test_enforce_scalar_invalid_return():
    @enforce_return_type(AggregationLevel.SCALAR)
    def f(x):
        return x

    with pytest.raises(InvalidFunctionError):
        f(np.ones(3))


def test_enforce_least_squares_with_vector_return():
    @enforce_return_type(AggregationLevel.LEAST_SQUARES)
    def f(x):
        return np.ones(3)

    got = f(np.ones(3))
    assert isinstance(got, LeastSquaresFunctionValue)
    aae(got.value, np.ones(3))


def test_enforce_least_squares_with_function_value_return():
    @enforce_return_type(AggregationLevel.LEAST_SQUARES)
    def f(x):
        return FunctionValue(np.ones(3))

    got = f(np.ones(3))
    assert isinstance(got, LeastSquaresFunctionValue)
    aae(got.value, np.ones(3))


def test_enforce_least_squares_trivial_case():
    @enforce_return_type(AggregationLevel.LEAST_SQUARES)
    def f(x):
        return LeastSquaresFunctionValue(np.ones(3))

    got = f(np.ones(3))
    assert isinstance(got, LeastSquaresFunctionValue)
    aae(got.value, np.ones(3))


def test_enforce_least_squares_invalid_return():
    @enforce_return_type(AggregationLevel.LEAST_SQUARES)
    def f(x):
        return 3

    with pytest.raises(InvalidFunctionError):
        f(np.ones(3))


def test_enforce_likelihood_with_vector_return():
    @enforce_return_type(AggregationLevel.LIKELIHOOD)
    def f(x):
        return np.ones(3)

    got = f(np.ones(3))
    assert isinstance(got, LikelihoodFunctionValue)
    aae(got.value, np.ones(3))


def test_enforce_likelihood_with_function_value_return():
    @enforce_return_type(AggregationLevel.LIKELIHOOD)
    def f(x):
        return FunctionValue(np.ones(3))

    got = f(np.ones(3))
    assert isinstance(got, LikelihoodFunctionValue)
    aae(got.value, np.ones(3))


def test_enforce_likelihood_trivial_case():
    @enforce_return_type(AggregationLevel.LIKELIHOOD)
    def f(x):
        return LikelihoodFunctionValue(np.ones(3))

    got = f(np.ones(3))
    assert isinstance(got, LikelihoodFunctionValue)
    aae(got.value, np.ones(3))


def test_enforce_likelihood_invalid_return():
    @enforce_return_type(AggregationLevel.LIKELIHOOD)
    def f(x):
        return 3

    with pytest.raises(InvalidFunctionError):
        f(np.ones(3))


def test_enforce_scalar_with_jac_with_scalar_return():
    @enforce_return_type_with_jac(AggregationLevel.SCALAR)
    def f(x):
        return 3, np.zeros(3)

    got_value, got_jac = f(np.ones(3))
    assert isinstance(got_value, ScalarFunctionValue)
    assert got_value.value == 3
    aae(got_jac, np.zeros(3))


def test_enforce_scalar_with_jac_with_function_value_return():
    @enforce_return_type_with_jac(AggregationLevel.SCALAR)
    def f(x):
        return FunctionValue(3), np.zeros(3)

    got_value, got_jac = f(np.ones(3))
    assert isinstance(got_value, ScalarFunctionValue)
    assert got_value.value == 3
    aae(got_jac, np.zeros(3))


def test_enforce_scalar_with_jac_trivial_case():
    @enforce_return_type_with_jac(AggregationLevel.SCALAR)
    def f(x):
        return ScalarFunctionValue(3), np.zeros(3)

    got_value, got_jac = f(3)
    assert isinstance(got_value, ScalarFunctionValue)
    assert got_value.value == 3
    aae(got_jac, np.zeros(3))


def test_enforce_scalar_with_jac_invalid_return():
    @enforce_return_type_with_jac(AggregationLevel.SCALAR)
    def f(x):
        return x, np.zeros(3)

    with pytest.raises(InvalidFunctionError):
        f(np.ones(3))


def test_enforce_least_squares_with_jac_with_vector_return():
    @enforce_return_type_with_jac(AggregationLevel.LEAST_SQUARES)
    def f(x):
        return np.ones(3), np.zeros((3, 3))

    got_value, got_jac = f(np.ones(3))
    assert isinstance(got_value, LeastSquaresFunctionValue)
    aae(got_value.value, np.ones(3))
    aae(got_jac, np.zeros((3, 3)))


def test_enforce_least_squares_with_jac_with_function_value_return():
    @enforce_return_type_with_jac(AggregationLevel.LEAST_SQUARES)
    def f(x):
        return FunctionValue(np.ones(3)), np.zeros((3, 3))

    got_value, got_jac = f(np.ones(3))
    assert isinstance(got_value, LeastSquaresFunctionValue)
    aae(got_value.value, np.ones(3))
    aae(got_jac, np.zeros((3, 3)))


def test_enforce_least_squares_with_jac_trivial_case():
    @enforce_return_type_with_jac(AggregationLevel.LEAST_SQUARES)
    def f(x):
        return LeastSquaresFunctionValue(np.ones(3)), np.zeros((3, 3))

    got_value, got_jac = f(np.ones(3))
    assert isinstance(got_value, LeastSquaresFunctionValue)
    aae(got_value.value, np.ones(3))
    aae(got_jac, np.zeros((3, 3)))


def test_enforce_least_squares_with_jac_invalid_return():
    @enforce_return_type_with_jac(AggregationLevel.LEAST_SQUARES)
    def f(x):
        return 3, np.zeros((3, 3))

    with pytest.raises(InvalidFunctionError):
        f(np.ones(3))


def test_enforce_likelihood_with_jac_with_vector_return():
    @enforce_return_type_with_jac(AggregationLevel.LIKELIHOOD)
    def f(x):
        return np.ones(3), np.zeros((3, 3))

    got_value, got_jac = f(np.ones(3))
    assert isinstance(got_value, LikelihoodFunctionValue)
    aae(got_value.value, np.ones(3))
    aae(got_jac, np.zeros((3, 3)))


def test_enforce_likelihood_with_jac_with_function_value_return():
    @enforce_return_type_with_jac(AggregationLevel.LIKELIHOOD)
    def f(x):
        return FunctionValue(np.ones(3)), np.zeros((3, 3))

    got_value, got_jac = f(np.ones(3))
    assert isinstance(got_value, LikelihoodFunctionValue)
    aae(got_value.value, np.ones(3))
    aae(got_jac, np.zeros((3, 3)))


def test_enforce_likelihood_with_jac_trivial_case():
    @enforce_return_type_with_jac(AggregationLevel.LIKELIHOOD)
    def f(x):
        return LikelihoodFunctionValue(np.ones(3)), np.zeros((3, 3))

    got_value, got_jac = f(np.ones(3))
    assert isinstance(got_value, LikelihoodFunctionValue)
    aae(got_value.value, np.ones(3))
    aae(got_jac, np.zeros((3, 3)))


def test_enforce_likelihood_with_jac_invalid_return():
    @enforce_return_type_with_jac(AggregationLevel.LIKELIHOOD)
    def f(x):
        return 3, np.zeros((3, 3))

    with pytest.raises(InvalidFunctionError):
        f(np.ones(3))
