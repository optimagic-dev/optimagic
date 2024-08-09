import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal as aaae
from optimagic.examples.criterion_functions import (
    sos_gradient,
    sos_ls,
    sos_ls_jacobian,
    sos_scalar,
)
from optimagic.optimization.optimize import minimize
from optimagic.parameters.tree_registry import get_registry
from pybaum import tree_just_flatten

REGISTRY = get_registry(extended=True)

PARAMS = [
    {"a": 1.0, "b": 2, "c": 3, "d": 4, "e": 5},
    np.arange(5),
    list(range(5)),
    tuple(range(5)),
    pd.Series(np.arange(5)),
    {"a": 1, "b": np.array([2, 3]), "c": [pd.Series([4, 5])]},
]

SCALAR_PARAMS = [6, 6.2, np.array([4]), np.array([4.5])]


@pytest.mark.parametrize("params", PARAMS + SCALAR_PARAMS)
def test_tree_params_numerical_derivative_scalar_criterion(params):
    flat = np.array(tree_just_flatten(params, registry=REGISTRY))
    expected = np.zeros_like(flat)

    res = minimize(
        fun=sos_scalar,
        params=params,
        algorithm="scipy_lbfgsb",
    )
    calculated = np.array(tree_just_flatten(res.params, registry=REGISTRY))
    aaae(calculated, expected)


@pytest.mark.parametrize("params", PARAMS + SCALAR_PARAMS)
def test_tree_params_scalar_criterion(params):
    flat = np.array(tree_just_flatten(params, registry=REGISTRY))
    expected = np.zeros_like(flat)

    res = minimize(
        fun=sos_scalar,
        jac=sos_gradient,
        params=params,
        algorithm="scipy_lbfgsb",
    )
    calculated = np.array(tree_just_flatten(res.params, registry=REGISTRY))
    aaae(calculated, expected)


TEST_CASES_SOS_LS = []
for p in PARAMS:
    for algo in ["scipy_lbfgsb", "scipy_ls_lm"]:
        TEST_CASES_SOS_LS.append((p, algo))


@pytest.mark.parametrize("params, algorithm", TEST_CASES_SOS_LS)
def test_tree_params_numerical_derivative_sos_ls(params, algorithm):
    flat = np.array(tree_just_flatten(params, registry=REGISTRY))
    expected = np.zeros_like(flat)

    res = minimize(
        fun=sos_ls,
        params=params,
        algorithm=algorithm,
    )
    calculated = np.array(tree_just_flatten(res.params, registry=REGISTRY))
    aaae(calculated, expected)


@pytest.mark.parametrize("params, algorithm", TEST_CASES_SOS_LS)
def test_tree_params_sos_ls(params, algorithm):
    flat = np.array(tree_just_flatten(params, registry=REGISTRY))
    expected = np.zeros_like(flat)

    derivatives = [sos_gradient, sos_ls_jacobian]
    res = minimize(
        fun=sos_ls,
        jac=derivatives,
        params=params,
        algorithm=algorithm,
    )
    calculated = np.array(tree_just_flatten(res.params, registry=REGISTRY))
    aaae(calculated, expected)
