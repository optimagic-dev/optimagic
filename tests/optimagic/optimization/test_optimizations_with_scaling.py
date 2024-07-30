"""Test a subset of algorithms with different scaling methods on sum of squares."""

import itertools

import numpy as np
import pandas as pd
from optimagic.options import ScalingOptions
import pytest
from optimagic.config import IS_PYBOBYQA_INSTALLED
from optimagic.optimization.optimize import minimize
from numpy.testing import assert_array_almost_equal as aaae

ALGORITHMS = ["scipy_lbfgsb"]

if IS_PYBOBYQA_INSTALLED:
    ALGORITHMS.append("nag_pybobyqa")


def sos_scalar_criterion(params):
    return (params["value"].to_numpy() ** 2).sum()


def sos_gradient(params):
    return 2 * params["value"].to_numpy()


SCALING_OPTIONS = [
    {"method": "start_values"},
    {"method": "bounds"},
]

PARAMETRIZATION = list(itertools.product(ALGORITHMS, SCALING_OPTIONS))


@pytest.mark.parametrize("algorithm, scaling_options", PARAMETRIZATION)
def test_optimizations_with_scaling_via_dict_options(algorithm, scaling_options):
    params = pd.DataFrame()
    params["value"] = np.arange(5)
    params["lower_bound"] = [-1, 0, 0, 0, 0]
    params["upper_bound"] = np.full(5, 10)

    constraints = [{"loc": [3, 4], "type": "fixed"}]

    res = minimize(
        fun=sos_scalar_criterion,
        params=params,
        constraints=constraints,
        algorithm=algorithm,
        scaling=True,
        scaling_options=scaling_options,
        jac=sos_gradient,
    )

    expected_solution = np.array([0, 0, 0, 3, 4])
    aaae(res.params["value"].to_numpy(), expected_solution)


@pytest.mark.parametrize("algorithm, scaling_options", PARAMETRIZATION)
def test_optimizations_with_scaling(algorithm, scaling_options):
    params = pd.DataFrame()
    params["value"] = np.arange(5)
    params["lower_bound"] = [-1, 0, 0, 0, 0]
    params["upper_bound"] = np.full(5, 10)

    constraints = [{"loc": [3, 4], "type": "fixed"}]

    scaling = ScalingOptions(**scaling_options)

    res = minimize(
        fun=sos_scalar_criterion,
        params=params,
        constraints=constraints,
        algorithm=algorithm,
        scaling=scaling,
        jac=sos_gradient,
    )

    expected_solution = np.array([0, 0, 0, 3, 4])
    aaae(res.params["value"].to_numpy(), expected_solution)
