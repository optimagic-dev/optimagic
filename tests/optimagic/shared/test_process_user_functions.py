import numpy as np
import pytest
from numpy.typing import NDArray
from optimagic import mark
from optimagic.exceptions import InvalidKwargsError
from optimagic.optimization.fun_value import (
    LeastSquaresFunctionValue,
    LikelihoodFunctionValue,
    ScalarFunctionValue,
)
from optimagic.shared.process_user_function import (
    get_kwargs_from_args,
    infer_problem_type,
    partial_func_of_params,
)
from optimagic.typing import ProblemType


def test_partial_func_of_params():
    def f(params, b, c):
        return params + b + c

    func = partial_func_of_params(f, {"b": 2, "c": 3})

    assert func(1) == 6


def test_partial_func_of_params_too_many_kwargs():
    def f(params, b, c):
        return params + b + c

    with pytest.raises(InvalidKwargsError):
        partial_func_of_params(f, {"params": 1, "b": 2, "c": 3})


def test_partial_func_of_params_too_few_kwargs():
    def f(params, b, c):
        return params + b + c

    with pytest.raises(InvalidKwargsError):
        partial_func_of_params(f, {"c": 3})


def test_get_kwargs_from_args():
    def f(a, b, c=3, d=4):
        return a + b + c

    got = get_kwargs_from_args([1, 2], f, offset=1)
    expected = {"b": 1, "c": 2}

    assert got == expected


def test_infer_problem_type_no_decorator():
    def f(params):
        return 1

    assert infer_problem_type(f) == ProblemType.SCALAR


def test_infer_problem_type_scalar_decorator():
    @mark.scalar
    def f(params):
        return 1

    assert infer_problem_type(f) == ProblemType.SCALAR


def test_infer_problem_type_scalar_anotation():
    def f(params: NDArray[np.float64]) -> ScalarFunctionValue:
        return ScalarFunctionValue(1)

    assert infer_problem_type(f) == ProblemType.SCALAR


def test_infer_problem_type_least_squares_decorator():
    @mark.least_squares
    def f(params):
        return np.ones(3)

    assert infer_problem_type(f) == ProblemType.LEAST_SQUARES


def test_infer_problem_type_least_squares_anotation():
    def f(params: NDArray[np.float64]) -> LeastSquaresFunctionValue:
        return LeastSquaresFunctionValue(np.ones(3))

    assert infer_problem_type(f) == ProblemType.LEAST_SQUARES


def test_infer_problem_type_likelihood_decorator():
    @mark.likelihood
    def f(params):
        return np.ones(3)

    assert infer_problem_type(f) == ProblemType.LIKELIHOOD


def test_infer_problem_type_likelihood_anotation():
    def f(params: NDArray[np.float64]) -> LikelihoodFunctionValue:
        return LikelihoodFunctionValue(np.ones(3))

    assert infer_problem_type(f) == ProblemType.LIKELIHOOD
