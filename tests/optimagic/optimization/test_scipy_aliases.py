import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

import optimagic as om
from optimagic.exceptions import AliasError, InvalidFunctionError, InvalidKwargsError


def test_x0_works_in_minimize():
    res = om.minimize(
        fun=lambda x: x @ x,
        x0=np.arange(3),
        algorithm="scipy_lbfgsb",
    )
    aaae(res.params, np.zeros(3))


def test_x0_works_in_maximize():
    res = om.maximize(
        fun=lambda x: -x @ x,
        x0=np.arange(3),
        algorithm="scipy_lbfgsb",
    )
    aaae(res.params, np.zeros(3))


def test_x0_and_params_do_not_work_together_in_minimize():
    with pytest.raises(AliasError, match="x0 is an alias"):
        om.minimize(
            fun=lambda x: x @ x,
            x0=np.arange(3),
            params=np.arange(3),
            algorithm="scipy_lbfgsb",
        )


def test_x0_and_params_do_not_work_together_in_maximize():
    with pytest.raises(AliasError, match="x0 is an alias"):
        om.maximize(
            fun=lambda x: -x @ x,
            x0=np.arange(3),
            params=np.arange(3),
            algorithm="scipy_lbfgsb",
        )


METHODS = [
    "Nelder-Mead",
    "Powell",
    "CG",
    "BFGS",
    "Newton-CG",
    "L-BFGS-B",
    "TNC",
    "COBYLA",
    "SLSQP",
    "trust-constr",
]


@pytest.mark.parametrize("method", METHODS)
def test_method_works_in_minimize(method):
    res = om.minimize(
        fun=lambda x: x @ x,
        x0=np.arange(3),
        method="L-BFGS-B",
    )
    aaae(res.params, np.zeros(3))


@pytest.mark.parametrize("method", METHODS)
def test_method_works_in_maximize(method):
    res = om.maximize(
        fun=lambda x: -x @ x,
        x0=np.arange(3),
        method="L-BFGS-B",
    )
    aaae(res.params, np.zeros(3))


def test_method_and_algorithm_do_not_work_together_in_minimize():
    with pytest.raises(AliasError, match="method is an alias"):
        om.minimize(
            fun=lambda x: x @ x,
            x0=np.arange(3),
            algorithm="scipy_lbfgsb",
            method="L-BFGS-B",
        )


def test_method_and_algorithm_do_not_work_together_in_maximize():
    with pytest.raises(AliasError, match="method is an alias"):
        om.maximize(
            fun=lambda x: -x @ x,
            x0=np.arange(3),
            algorithm="scipy_lbfgsb",
            method="L-BFGS-B",
        )


def test_exception_for_hess():
    msg = "The hess argument is not yet supported"
    with pytest.raises(NotImplementedError, match=msg):
        om.minimize(
            fun=lambda x: x @ x,
            x0=np.arange(3),
            algorithm="scipy_lbfgsb",
            hess=lambda x: np.eye(len(x)),
        )


def test_exception_for_hessp():
    msg = "The hessp argument is not yet supported"
    with pytest.raises(NotImplementedError, match=msg):
        om.minimize(
            fun=lambda x: x @ x,
            x0=np.arange(3),
            algorithm="scipy_lbfgsb",
            hessp=lambda x, p: np.eye(len(x)) @ p,
        )


def test_callback_xk_is_called():
    """SciPy-style callback(xk) is invoked on objective evaluations."""
    xs = []

    def callback(xk):
        xs.append(np.asarray(xk).copy())

    res = om.minimize(
        fun=lambda x: x @ x,
        x0=np.arange(3, dtype=float),
        algorithm="scipy_neldermead",
        callback=callback,
    )

    assert len(xs) >= 1
    assert xs[0].shape == (3,)
    aaae(res.x, np.zeros(3), decimal=5)


def test_callback_receives_external_params():
    """Callback gets external PyTree params, not the internal flat vector."""
    received = []

    def callback(xk):
        received.append(xk)

    params = {"a": np.array([1.0, 2.0]), "b": np.array([3.0])}

    def fun(p):
        return p["a"] @ p["a"] + p["b"] @ p["b"]

    om.minimize(
        fun=fun,
        params=params,
        algorithm="scipy_neldermead",
        callback=callback,
    )

    assert len(received) >= 1
    assert isinstance(received[0], dict)
    assert set(received[0]) == {"a", "b"}
    assert received[0]["a"].shape == (2,)
    assert received[0]["b"].shape == (1,)


def test_callback_not_called_on_jac():
    """Callback runs next to history on fun, not on jac-only evaluations."""
    from optimagic.optimization.internal_optimization_problem import (
        SphereExampleInternalOptimizationProblem,
    )

    problem = SphereExampleInternalOptimizationProblem()
    calls = []
    problem._callback = lambda p: calls.append(np.asarray(p).copy())

    x = np.ones(10)
    problem.jac(x)
    assert calls == []

    problem.fun(x)
    assert len(calls) == 1
    aaae(calls[0], x)


def test_invalid_callback_too_few_arguments():
    msg = "callback must have at least one free argument"

    def bad_callback():
        return None

    with pytest.raises(InvalidFunctionError, match=msg):
        om.minimize(
            fun=lambda x: x @ x,
            x0=np.arange(3, dtype=float),
            algorithm="scipy_neldermead",
            callback=bad_callback,
        )


def test_invalid_callback_too_many_required_arguments():
    msg = "Too few keyword arguments for callback"

    def bad_callback(xk, extra):
        return None

    with pytest.raises(InvalidKwargsError, match=msg):
        om.minimize(
            fun=lambda x: x @ x,
            x0=np.arange(3, dtype=float),
            algorithm="scipy_neldermead",
            callback=bad_callback,
        )


def test_invalid_callback_not_callable():
    with pytest.raises(InvalidFunctionError, match="callback must be a callable"):
        om.minimize(
            fun=lambda x: x @ x,
            x0=np.arange(3, dtype=float),
            algorithm="scipy_neldermead",
            callback="not-a-callable",
        )


def test_exception_for_options():
    msg = "The options argument is not supported"
    with pytest.raises(NotImplementedError, match=msg):
        om.minimize(
            fun=lambda x: x @ x,
            x0=np.arange(3),
            algorithm="scipy_lbfgsb",
            options={"maxiter": 100},
        )


def test_exception_for_tol():
    msg = "The tol argument is not supported"
    with pytest.raises(NotImplementedError, match=msg):
        om.minimize(
            fun=lambda x: x @ x,
            x0=np.arange(3),
            algorithm="scipy_lbfgsb",
            tol=1e-6,
        )


def test_args_works_in_minimize():
    res = om.minimize(
        fun=lambda x, a: ((x - a) ** 2).sum(),
        x0=np.arange(3),
        args=(1,),
        algorithm="scipy_lbfgsb",
    )
    aaae(res.params, np.ones(3))


def test_args_works_in_maximize():
    res = om.maximize(
        fun=lambda x, a: -((x - a) ** 2).sum(),
        x0=np.arange(3),
        args=(1,),
        algorithm="scipy_lbfgsb",
    )
    aaae(res.params, np.ones(3))


def test_args_does_not_work_with_together_with_any_kwargs():
    with pytest.raises(AliasError, match="args is an alternative"):
        om.minimize(
            fun=lambda x, a: ((x - a) ** 2).sum(),
            params=np.arange(3),
            algorithm="scipy_lbfgsb",
            args=(1,),
            fun_kwargs={"a": 1},
        )


def test_jac_equal_true_works_in_minimize():
    res = om.minimize(
        fun=lambda x: (x @ x, 2 * x),
        params=np.arange(3),
        algorithm="scipy_lbfgsb",
        jac=True,
    )
    aaae(res.params, np.zeros(3))


def test_jac_equal_true_works_in_maximize():
    res = om.maximize(
        fun=lambda x: (-x @ x, -2 * x),
        params=np.arange(3),
        algorithm="scipy_lbfgsb",
        jac=True,
    )
    aaae(res.params, np.zeros(3))
