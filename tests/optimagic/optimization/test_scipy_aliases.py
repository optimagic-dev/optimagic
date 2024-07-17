import optimagic as om
import numpy as np
from numpy.testing import assert_array_almost_equal as aaae
from optimagic.exceptions import AliasError
import pytest


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
