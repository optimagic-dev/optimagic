"""Test different ways of specifying objective functions and their derivatives.

We also test that least-squares problems can be optimized with scalar optimizers.

"""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from optimagic import mark, maximize, minimize
from optimagic.exceptions import InvalidFunctionError
from optimagic.optimization.fun_value import (
    FunctionValue,
    LeastSquaresFunctionValue,
)

# ======================================================================================
# minimize cases with numpy params
# ======================================================================================


@mark.least_squares
def sos_ls(x):
    return x


def typed_sos_ls(x: np.ndarray) -> LeastSquaresFunctionValue:
    return LeastSquaresFunctionValue(x)


@mark.least_squares
def sos_ls_with_info(x):
    return FunctionValue(x, info={"x": x})


MIN_FUNS = [
    sos_ls,
    typed_sos_ls,
    sos_ls_with_info,
]


def jac(x):
    return 2 * x


@mark.least_squares
def jac_ls(x):
    return np.diag(2 * x)


MIN_JACS = [None, [jac, jac_ls]]


ALGORITHMS = ["scipy_lbfgsb", "scipy_ls_lm"]


@pytest.mark.parametrize("fun", MIN_FUNS)
@pytest.mark.parametrize("jac", MIN_JACS)
@pytest.mark.parametrize("use_fun_and_jac", [False, True])
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_least_squares_minimize(fun, jac, use_fun_and_jac, algorithm):
    start_params = np.array([1, 2, 3])
    if use_fun_and_jac and jac is not None:

        def fun_and_jac_scalar(x):
            return x @ x, 2 * x

        @mark.least_squares
        def fun_and_jac_ls(x):
            return x, np.diag(2 * x)

        fun_and_jac = [fun_and_jac_scalar, fun_and_jac_ls]
    else:
        fun_and_jac = None

    res = minimize(
        fun=fun,
        params=start_params,
        algorithm=algorithm,
        jac=jac,
        fun_and_jac=fun_and_jac,
    )
    aaae(res.params, np.zeros(3))


# ======================================================================================
# minimize cases with dict params
# ======================================================================================


def dict_jac(params):
    return {k: 2 * v for k, v in params.items()}


@mark.least_squares
def dict_jac_ls(params):
    out = {}
    for outer_key in params:
        row = {}
        for inner_key in params:
            if outer_key == inner_key:
                row[inner_key] = 2 * params[inner_key]
            else:
                row[inner_key] = 0
        out[outer_key] = row
    return out


MIN_JACS_DICT = [None, [dict_jac, dict_jac_ls]]


@pytest.mark.parametrize("fun", MIN_FUNS)
@pytest.mark.parametrize("jac", MIN_JACS_DICT)
@pytest.mark.parametrize("use_fun_and_jac", [False, True])
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_least_squares_minimize_dict(fun, jac, use_fun_and_jac, algorithm):
    start_params = {"a": 1, "b": 2, "c": 3}

    if use_fun_and_jac and jac is not None:

        def fun_and_jac_dict_scalar(params):
            x = np.array(list(params.values()))
            return x @ x, dict_jac(params)

        @mark.least_squares
        def fun_and_jac_dict_ls(params):
            return params, dict_jac_ls(params)

        fun_and_jac = [fun_and_jac_dict_scalar, fun_and_jac_dict_ls]
    else:
        fun_and_jac = None

    res = minimize(
        fun=fun,
        params=start_params,
        algorithm=algorithm,
        jac=jac,
        fun_and_jac=fun_and_jac,
    )

    for key in start_params:
        assert np.allclose(res.params[key], 0, atol=1e-5)


# ======================================================================================
# invalid cases
# ======================================================================================


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_maximize_with_ls_problems_raises_error(algorithm):
    with pytest.raises(InvalidFunctionError):
        maximize(
            fun=sos_ls,
            params=np.array([1, 2, 3]),
            algorithm=algorithm,
        )


@mark.least_squares
def invalid_sos_ls(x):
    return x @ x


@mark.least_squares
def invalid_sos_ls_with_info(x):
    return FunctionValue(x @ x, info={"x": x})


INVALID_FUNS = [
    invalid_sos_ls,
    invalid_sos_ls_with_info,
]


@pytest.mark.parametrize("fun", INVALID_FUNS)
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_invalid_least_squares_minimize(fun, algorithm):
    start_params = np.array([1, 2, 3])

    with pytest.raises(InvalidFunctionError):
        minimize(
            fun=fun,
            params=start_params,
            algorithm=algorithm,
        )


@mark.least_squares
def invalid_jac_ls(x):
    return 2 * x


@mark.least_squares
def invalid_jac_ls_2(x):
    return FunctionValue(2 * x)


INVALID_JACS = [invalid_jac_ls, invalid_jac_ls_2]


@pytest.mark.parametrize("jac", INVALID_JACS)
def test_least_squares_minimize_with_invalid_jac(jac):
    with pytest.raises(Exception):  # noqa: B017
        minimize(
            fun=sos_ls,
            params=np.array([1, 2, 3]),
            algorithm="scipy_ls_lm",
            jac=jac,
        )


@mark.least_squares
def invalid_fun_and_jac_value(x):
    return x @ x, np.diag(2 * x)


@mark.least_squares
def invalid_fun_and_jac_derivative(x):
    return x, 2 * x


INVALID_FUN_AND_JACS = [invalid_fun_and_jac_value, invalid_fun_and_jac_derivative]


@pytest.mark.parametrize("fun_and_jac", INVALID_FUN_AND_JACS)
def test_least_squares_minimize_with_invalid_fun_and_jac(fun_and_jac):
    with pytest.raises(InvalidFunctionError):
        minimize(
            fun=sos_ls,
            params=np.array([1, 2, 3]),
            algorithm="scipy_ls_lm",
            fun_and_jac=fun_and_jac,
        )
