"""Test helper functions for nevergrad optimizers."""

from typing import get_args

import numpy as np
import pytest

from optimagic import algorithms, mark
from optimagic.config import IS_NEVERGRAD_INSTALLED
from optimagic.optimizers.nevergrad_optimizers import (
    _batch_constraint_evaluations,
    _get_constraint_evaluations,
    _process_nonlinear_constraints,
)

if IS_NEVERGRAD_INSTALLED:
    import nevergrad as ng


@mark.least_squares
def sos(x):
    return x


def dummy_func():
    return lambda x: x


vec_constr = [
    {
        "type": "ineq",
        "fun": lambda x: [np.prod(x) + 1.0, 2.0 - np.prod(x)],
        "jac": dummy_func,
        "n_constr": 2,
    }
]

constrs = [
    {
        "type": "ineq",
        "fun": lambda x: np.prod(x) + 1.0,
        "jac": dummy_func,
        "n_constr": 1,
    },
    {
        "type": "ineq",
        "fun": lambda x: 2.0 - np.prod(x),
        "jac": dummy_func,
        "n_constr": 1,
    },
]


def test_process_nonlinear_constraints():
    got = _process_nonlinear_constraints(vec_constr)
    assert len(got) == 2


def test_get_constraint_evaluations():
    x = np.array([1, 1])
    got = _get_constraint_evaluations(constrs, x)
    expected = [np.array([-2.0]), np.array([-1.0])]
    assert got == expected


def test_batch_constraint_evaluations():
    x = np.array([1, 1])
    x_list = [x] * 2
    got = _batch_constraint_evaluations(constrs, x_list, 2)
    expected = [[np.array([-2.0]), np.array([-1.0])]] * 2
    assert got == expected


# def test_nevergrad_internal():
#     pass


# test all optimizers listed in Literal type hint are valid attributes
@pytest.mark.skipif(not IS_NEVERGRAD_INSTALLED, reason="nevergrad not installed")
def test_meta_optimizers_are_valid():
    opt = algorithms.NevergradMeta
    optimizers = get_args(opt.__annotations__["optimizer"])
    for optimizer in optimizers:
        try:
            getattr(ng.optimizers, optimizer)
        except AttributeError:
            pytest.fail(f"Optimizer '{optimizer}' not found in Nevergrad")


@pytest.mark.skipif(not IS_NEVERGRAD_INSTALLED, reason="nevergrad not installed")
def test_ngopt_optimizers_are_valid():
    opt = algorithms.NevergradNGOpt
    optimizers = get_args(opt.__annotations__["optimizer"])
    for optimizer in optimizers:
        try:
            getattr(ng.optimizers, optimizer)
        except AttributeError:
            pytest.fail(f"Optimizer '{optimizer}' not found in Nevergrad")
