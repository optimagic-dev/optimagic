"""Test helper functions for nevergrad optimizers."""

import time
from typing import get_args, get_type_hints

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from optimagic import algorithms, mark
from optimagic.config import IS_NEVERGRAD_INSTALLED
from optimagic.optimization.optimize import minimize
from optimagic.parameters.bounds import Bounds

pytestmark = pytest.mark.nevergrad

if IS_NEVERGRAD_INSTALLED:
    import nevergrad as ng


@mark.least_squares
def sos(x):
    return x


### Nonlinear constraints on hold until improved handling.
# def dummy_func():
#     return lambda x: x


# vec_constr = [
#     {
#         "type": "ineq",
#         "fun": lambda x: [np.prod(x) + 1.0, 2.0 - np.prod(x)],
#         "jac": dummy_func,
#         "n_constr": 2,
#     }
# ]

# constrs = [
#     {
#         "type": "ineq",
#         "fun": lambda x: np.prod(x) + 1.0,
#         "jac": dummy_func,
#         "n_constr": 1,
#     },
#     {
#         "type": "ineq",
#         "fun": lambda x: 2.0 - np.prod(x),
#         "jac": dummy_func,
#         "n_constr": 1,
#     },
# ]


# def test_process_nonlinear_constraints():
#     got = _process_nonlinear_constraints(vec_constr)
#     assert len(got) == 2


# def test_get_constraint_evaluations():
#     x = np.array([1, 1])
#     got = _get_constraint_evaluations(constrs, x)
#     expected = [np.array([-2.0]), np.array([-1.0])]
#     assert got == expected


# def test_batch_constraint_evaluations():
#     x = np.array([1, 1])
#     x_list = [x] * 2
#     got = _batch_constraint_evaluations(constrs, x_list, 2)
#     expected = [[np.array([-2.0]), np.array([-1.0])]] * 2
#     assert got == expected
###


def _get_optimizer_names(algorithm):
    """Return the values of the Literal type hint of the option ``optimizer``.

    The annotations of the nevergrad module are strings, so they have to be evaluated
    with ``get_type_hints``. ``get_args`` of the raw string would return ``()``.

    """
    return get_args(get_type_hints(algorithm)["optimizer"])


NEVERGRAD_META = _get_optimizer_names(algorithms.NevergradMeta)
NEVERGRAD_NGOPT = _get_optimizer_names(algorithms.NevergradNGOpt)


@pytest.mark.parametrize("optimizer", NEVERGRAD_META + NEVERGRAD_NGOPT)
def test_optimizers_in_literal_type_hints_exist_in_nevergrad(optimizer):
    assert hasattr(ng.optimizers, optimizer)


# test stochastic_global_algorithm_on_sum_of_squares
IMPRECISE_META_OPTIMIZERS = {"NoisyBandit"}


@pytest.mark.slow
@pytest.mark.parametrize(
    "algorithm",
    [
        pytest.param(
            name,
            marks=pytest.mark.xfail(
                raises=AssertionError,
                strict=True,
                reason="Does not reach the required precision on this problem.",
            ),
        )
        if name in IMPRECISE_META_OPTIMIZERS
        else name
        for name in NEVERGRAD_META
    ],
)
def test_meta_optimizers_with_stochastic_global_algorithm_on_sum_of_squares(algorithm):
    res = minimize(
        fun=sos,
        params=np.array([0.35, 0.35]),
        bounds=Bounds(lower=np.array([0.2, -0.5]), upper=np.array([1, 0.5])),
        algorithm=algorithms.NevergradMeta(algorithm),
        collect_history=False,
        skip_checks=True,
        algo_options={"seed": 12345},
    )
    assert res.success in [True, None]
    aaae(res.params, np.array([0.2, 0]), decimal=1)


@pytest.mark.slow
@pytest.mark.parametrize("algorithm", NEVERGRAD_NGOPT)
def test_ngopt_optimizers_with_stochastic_global_algorithm_on_sum_of_squares(algorithm):
    res = minimize(
        fun=sos,
        params=np.array([0.35, 0.35]),
        bounds=Bounds(lower=np.array([0.2, -0.5]), upper=np.array([1, 0.5])),
        algorithm=algorithms.NevergradNGOpt(algorithm),
        collect_history=False,
        skip_checks=True,
        algo_options={"seed": 12345},
    )
    assert res.success in [True, None]
    aaae(res.params, np.array([0.2, 0]), decimal=1)


def test_cmaes_returns_best_point_of_all_restarts():
    # stopping_maxiter=5 makes nevergrad restart CMA-ES every 5 iterations, and
    # nevergrad only recommends the best point of the last restart.
    res = minimize(
        fun=lambda x: x @ x,
        params=np.arange(3.0),
        algorithm=algorithms.NevergradCMAES(
            stopping_maxiter=5, stopping_maxfun=500, seed=0
        ),
    )
    assert res.history is not None
    assert res.fun == min(res.history.fun)


def test_cmaes_stops_after_stopping_maxtime():
    def slow_sphere(x):
        time.sleep(0.01)
        return x @ x

    res = minimize(
        fun=slow_sphere,
        params=np.arange(3.0),
        algorithm=algorithms.NevergradCMAES(
            stopping_maxtime=0.1, stopping_maxfun=1_000, seed=0
        ),
    )
    assert res.n_fun_evals is not None
    assert res.n_fun_evals < 100
