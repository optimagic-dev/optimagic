"""Test different ways of specifying objective functions and their derivatives."""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae
from numpy.typing import NDArray
from optimagic import mark, maximize, minimize
from optimagic.exceptions import InvalidFunctionError
from optimagic.optimization.fun_value import FunctionValue, ScalarFunctionValue

# ======================================================================================
# minimize cases with numpy params
# ======================================================================================


def sos(x):
    return x @ x


@mark.scalar
def marked_sos(x):
    return x @ x


def typed_sos_float(x: np.ndarray) -> float:
    return x @ x


def typed_sos_value(x: np.ndarray) -> ScalarFunctionValue:
    return ScalarFunctionValue(x @ x)


def sos_with_info(x):
    return FunctionValue(x @ x, info={"x": x})


MIN_FUNS = [
    sos,
    marked_sos,
    typed_sos_float,
    typed_sos_value,
    sos_with_info,
]


def jac(x):
    return 2 * x


@mark.scalar
def marked_jac(x):
    return 2 * x


MIN_JACS = [None, jac, marked_jac]

FUN_AND_JAC_CASES = [None, "marked", "unmarked"]


@pytest.mark.parametrize("fun", MIN_FUNS)
@pytest.mark.parametrize("jac", MIN_JACS)
@pytest.mark.parametrize("fun_and_jac_case", FUN_AND_JAC_CASES)
def test_minimize_with_numpy_inputs(fun, jac, fun_and_jac_case):
    if fun_and_jac_case is None:
        fun_and_jac = None
    elif fun_and_jac_case == "marked":

        @mark.scalar
        def fun_and_jac(x):
            return fun(x), 2 * x
    else:

        def fun_and_jac(x):
            return fun(x), 2 * x

    res = minimize(
        fun=fun,
        params=np.array([1, 2, 3]),
        algorithm="scipy_lbfgsb",
        jac=jac,
        fun_and_jac=fun_and_jac,
    )
    aaae(res.params, np.zeros(3))


# ======================================================================================
# maximize cases with numpy params
# ======================================================================================


def neg_sos(x):
    return -x @ x


@mark.scalar
def marked_neg_sos(x):
    return -x @ x


def typed_neg_sos_float(x: np.ndarray) -> float:
    return -x @ x


def typed_neg_sos_value(x: np.ndarray) -> ScalarFunctionValue:
    return ScalarFunctionValue(-x @ x)


def neg_sos_with_info(x):
    return FunctionValue(-x @ x, info={"x": x})


MAX_FUNS = [
    neg_sos,
    marked_neg_sos,
    typed_neg_sos_float,
    typed_neg_sos_value,
    neg_sos_with_info,
]


def neg_jac(x):
    return -2 * x


@mark.scalar
def marked_neg_jac(x):
    return -2 * x


MAX_JACS = [None, neg_jac, marked_neg_jac]


@pytest.mark.parametrize("fun", MAX_FUNS)
@pytest.mark.parametrize("jac", MAX_JACS)
@pytest.mark.parametrize("fun_and_jac_case", FUN_AND_JAC_CASES)
def test_maximize_with_numpy_inputs(fun, jac, fun_and_jac_case):
    if fun_and_jac_case is None:
        fun_and_jac = None
    elif fun_and_jac_case == "marked":

        @mark.scalar
        def fun_and_jac(x):
            return fun(x), -2 * x
    else:

        def fun_and_jac(x):
            return fun(x), -2 * x

    res = maximize(
        fun=fun,
        params=np.array([1, 2, 3]),
        algorithm="scipy_lbfgsb",
        jac=jac,
        fun_and_jac=fun_and_jac,
    )
    aaae(res.params, np.zeros(3))


# ======================================================================================
# minimize cases with dict params
# ======================================================================================


def sos_dict(params):
    x = np.array(list(params.values()))
    return x @ x


@mark.scalar
def marked_sos_dict(params):
    x = np.array(list(params.values()))
    return x @ x


def typed_sos_dict_float(params: dict) -> float:
    x = np.array(list(params.values()))
    return x @ x


def typed_sos_dict_value(params: dict) -> ScalarFunctionValue:
    x = np.array(list(params.values()))
    return ScalarFunctionValue(x @ x)


def sos_dict_with_info(params):
    x = np.array(list(params.values()))
    return FunctionValue(x @ x, info={"x": x})


MIN_FUNS_DICT = [
    sos_dict,
    marked_sos_dict,
    typed_sos_dict_float,
    typed_sos_dict_value,
    sos_dict_with_info,
]


def jac_dict(params):
    return {k: 2 * v for k, v in params.items()}


@mark.scalar
def marked_jac_dict(params):
    return {k: 2 * v for k, v in params.items()}


MIN_JACS_DICT = [None, jac_dict, marked_jac_dict]


@pytest.mark.parametrize("fun", MIN_FUNS_DICT)
@pytest.mark.parametrize("jac", MIN_JACS_DICT)
@pytest.mark.parametrize("fun_and_jac_case", FUN_AND_JAC_CASES)
def test_minimize_with_dict_inputs(fun, jac, fun_and_jac_case):
    if fun_and_jac_case is None:
        fun_and_jac = None
    elif fun_and_jac_case == "marked":

        @mark.scalar
        def fun_and_jac(params):
            return fun(params), {k: 2 * v for k, v in params.items()}
    else:

        def fun_and_jac(params):
            return fun(params), {k: 2 * v for k, v in params.items()}

    res = minimize(
        fun=fun,
        params={"x": 1, "y": 2, "z": 3},
        algorithm="scipy_lbfgsb",
        jac=jac,
        fun_and_jac=fun_and_jac,
    )
    for number in res.params.values():
        assert np.allclose(number, 0, atol=1e-5)


# ======================================================================================
# maximize cases with dict params
# ======================================================================================


def neg_sos_dict(params):
    x = np.array(list(params.values()))
    return -x @ x


@mark.scalar
def marked_neg_sos_dict(params):
    x = np.array(list(params.values()))
    return -x @ x


def typed_neg_sos_dict_float(params: dict) -> float:
    x = np.array(list(params.values()))
    return -x @ x


def typed_neg_sos_dict_value(params: dict) -> ScalarFunctionValue:
    x = np.array(list(params.values()))
    return ScalarFunctionValue(-x @ x)


def neg_sos_dict_with_info(params):
    x = np.array(list(params.values()))
    return FunctionValue(-x @ x, info={"x": x})


MAX_FUNS_DICT = [
    neg_sos_dict,
    marked_neg_sos_dict,
    typed_neg_sos_dict_float,
    typed_neg_sos_dict_value,
    neg_sos_dict_with_info,
]


def neg_jac_dict(params):
    return {k: -2 * v for k, v in params.items()}


@mark.scalar
def marked_neg_jac_dict(params):
    return {k: -2 * v for k, v in params.items()}


MAX_JACS_DICT = [None, neg_jac_dict, marked_neg_jac_dict]


@pytest.mark.parametrize("fun", MAX_FUNS_DICT)
@pytest.mark.parametrize("jac", MAX_JACS_DICT)
@pytest.mark.parametrize("fun_and_jac_case", FUN_AND_JAC_CASES)
def test_maximize_with_dict_inputs(fun, jac, fun_and_jac_case):
    if fun_and_jac_case is None:
        fun_and_jac = None
    elif fun_and_jac_case == "marked":

        @mark.scalar
        def fun_and_jac(params):
            return fun(params), {k: -2 * v for k, v in params.items()}
    else:

        def fun_and_jac(params):
            return fun(params), {k: -2 * v for k, v in params.items()}

    res = maximize(
        fun=fun,
        params={"x": 1, "y": 2, "z": 3},
        algorithm="scipy_lbfgsb",
        jac=jac,
        fun_and_jac=fun_and_jac,
    )
    for number in res.params.values():
        assert np.allclose(number, 0, atol=1e-5)


# ======================================================================================
# invalid cases; Only test minimize for things that cannot plausibly depend on the
# direction of the optimization
# ======================================================================================


def test_invalid_marker_for_jac_in_minimize():
    @mark.least_squares
    def jac(x):
        return 2 * x

    with pytest.warns(UserWarning):
        minimize(
            fun=sos,
            params=np.array([1, 2, 3]),
            algorithm="scipy_lbfgsb",
            jac=jac,
        )


def test_invalid_marker_for_fun_and_jac_in_minimize():
    @mark.least_squares
    def fun_and_jac(x):
        return x @ x, 2 * x

    with pytest.warns(UserWarning):
        minimize(
            fun=sos,
            params=np.array([1, 2, 3]),
            algorithm="scipy_lbfgsb",
            fun_and_jac=fun_and_jac,
        )


def invalid_sos(x):
    return x


@mark.scalar
def invalid_marked_sos(x):
    return x


def invalid_typed_sos_array(x: np.ndarray) -> NDArray[np.float64]:
    return x


def invalid_typed_sos_value(x: np.ndarray) -> ScalarFunctionValue:
    return ScalarFunctionValue(x)


def invalid_sos_with_info(x):
    return FunctionValue(x, info={"x": x})


INVALID_FUNS = [
    invalid_sos,
    invalid_marked_sos,
    invalid_typed_sos_array,
    invalid_typed_sos_value,
    invalid_sos_with_info,
]


@pytest.mark.parametrize("fun", INVALID_FUNS)
def test_minimize_with_invalid_fun(fun):
    with pytest.raises(InvalidFunctionError):
        minimize(
            fun=fun,
            params=np.array([1, 2, 3]),
            algorithm="scipy_lbfgsb",
        )


def invalid_jac(x):
    return np.eye(len(x))


@mark.scalar
def invalid_marked_jac(x):
    return np.eye(len(x))


INVALID_JACS = [invalid_jac, invalid_marked_jac]


@pytest.mark.parametrize("jac", INVALID_JACS)
def test_minimize_with_invalid_jac(jac):
    with pytest.raises(Exception):  # noqa: B017
        minimize(
            fun=sos,
            params=np.array([1, 2, 3]),
            algorithm="scipy_lbfgsb",
            jac=jac,
        )


def invalid_fun_and_jac(x):
    return x, np.eye(len(x))


@mark.scalar
def invalid_marked_fun_and_jac(x):
    return x, np.eye(len(x))


INVALID_FUN_AND_JACS = [invalid_fun_and_jac, invalid_marked_fun_and_jac]


@pytest.mark.parametrize("fun_and_jac", INVALID_FUN_AND_JACS)
def test_minimize_with_invalid_fun_and_jac(fun_and_jac):
    with pytest.raises(Exception):  # noqa: B017
        minimize(
            fun=sos,
            params=np.array([1, 2, 3]),
            algorithm="scipy_lbfgsb",
            fun_and_jac=fun_and_jac,
        )
