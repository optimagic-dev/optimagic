"""Test the external interface for optimization."""
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from estimagic.config import ALL_ALGORITHMS
from estimagic.optimization.optimize import minimize


ALL_ALGORITHMS = list(ALL_ALGORITHMS)

LEAST_SQUARES_ALOGORITHMS = ["tao_pounders"]

SUM_ALGORITHMS = ["bhhh"]

SCALAR_ALGORITHMS = [
    alg
    for alg in ALL_ALGORITHMS
    if alg not in LEAST_SQUARES_ALOGORITHMS + SUM_ALGORITHMS
]


def sum_of_squares_dict_criterion(params):
    out = {
        "value": (params["value"] ** 2).sum(),
        "contributions": params["value"].to_numpy() ** 2,
        "root_contributions": params["value"].to_numpy(),
    }
    return out


def sum_of_squares_dict_criterion_with_pd_objects(params):
    out = {
        "value": (params["value"] ** 2).sum(),
        "contributions": params["value"] ** 2,
        "root_contributions": params["value"],
    }
    return out


def sum_of_squares_scalar_criterion(params):
    return (params["value"].to_numpy() ** 2).sum()


def sum_of_squares_gradient(params):
    return 2 * params["value"].to_numpy()


def sum_of_squares_jacobian(params):
    return np.diag(2 * params["value"])


def sum_of_squares_pandas_gradient(params):
    return 2 * params["value"]


def sum_of_squares_pandas_jacobian(params):
    return pd.DataFrame(np.diag(3 * params["value"]))


def sum_of_squares_criterion_and_derivative(params):
    x = params["value"].to_numpy()
    return (x ** 2).sum(), 2 * x


@pytest.mark.parametrize("algorithm", SCALAR_ALGORITHMS)
def test_minimization_no_derivative(algorithm):
    params = pd.DataFrame(data=np.ones((10, 1)), columns=["value"])
    params["lower"] = -10
    params["upper"] = 10
    batch_options = {"error_handling": "raise", "n_cores": 1}
    res = minimize(
        sum_of_squares_scalar_criterion,
        params,
        algorithm,
        batch_evaluator_options=batch_options,
    )
    aaae(res["solution_params"]["value"].to_numpy(), np.zeros(10))


@pytest.mark.parametrize("algorithm", ALL_ALGORITHMS)
def test_minimization_with_dict_output_no_derivative(algorithm):
    params = pd.DataFrame(data=np.ones((10, 1)), columns=["value"])
    batch_options = {"error_handling": "raise", "n_cores": 1}
    res = minimize(
        sum_of_squares_dict_criterion,
        params,
        algorithm,
        batch_evaluator_options=batch_options,
    )
    aaae(res["solution_params"]["value"].to_numpy(), np.zeros(10))


@pytest.mark.parametrize("algorithm", SCALAR_ALGORITHMS)
def test_minimization_scalar_output_with_derivative(algorithm):
    params = pd.DataFrame(data=np.ones((10, 1)), columns=["value"])
    batch_options = {"error_handling": "raise", "n_cores": 1}
    res = minimize(
        sum_of_squares_dict_criterion,
        params,
        algorithm,
        derivative=sum_of_squares_gradient,
        batch_evaluator_options=batch_options,
    )
    aaae(res["solution_params"]["value"].to_numpy(), np.zeros(10))


@pytest.mark.parametrize("algorithm", SCALAR_ALGORITHMS)
def test_minimization_scalar_output_with_criterion_and_derivative(algorithm):
    params = pd.DataFrame(data=np.ones((10, 1)), columns=["value"])
    batch_options = {"error_handling": "raise", "n_cores": 1}
    res = minimize(
        sum_of_squares_dict_criterion,
        params,
        algorithm,
        criterion_and_derivative=sum_of_squares_criterion_and_derivative,
        batch_evaluator_options=batch_options,
    )
    aaae(res["solution_params"]["value"].to_numpy(), np.zeros(10))
