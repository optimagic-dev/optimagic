from itertools import product
from pathlib import Path

import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from estimagic.inference import ml_covs

FIX_PATH = Path(__file__).resolve().parent / "fixtures"


def get_expected_covariance(model, cov_method):
    """Load expected covariance matrix.

    Args:
        model (str): one of ['logit', 'probit']
        cov_method (str): one of ['jacobian', 'hessian', 'robust']

    Returns:
        expected_covariance

    """
    _name = cov_method if cov_method != "robust" else "sandwich"
    fix_name = f"{model}_{_name}.pickle"
    expected_cov = pd.read_pickle(FIX_PATH / fix_name)
    return expected_cov


def get_input(model, input_types):
    """Load the inputs.

    Args:
        model (str): one of ['logit', 'probit']
        input_types (list): can contain the elements 'jacobian' and 'hessian'

    Returns:
        inputs (dict): The inputs for the covariance function

    """
    inputs = {}
    for typ in input_types:
        fix_name = "{}_{}_matrix.pickle".format(model, typ)
        input_matrix = pd.read_pickle(FIX_PATH / fix_name)
        inputs[typ] = input_matrix

    short_names = {"jacobian": "jac", "hessian": "hess"}
    inputs = {short_names[key]: val for key, val in inputs.items()}
    return inputs


models = ["probit", "logit"]
methods = ["jacobian", "hessian", "robust"]
test_cases = list(product(models, methods))


@pytest.mark.parametrize("model, method", test_cases)
def test_cov_function_against_statsmodels(model, method):
    expected = get_expected_covariance(model, method)

    if method in ["jacobian", "hessian"]:
        input_types = [method]
    elif method == "robust":
        input_types = ["jacobian", "hessian"]

    inputs = get_input(model, input_types)

    calculated = getattr(ml_covs, "cov_{}".format(method))(**inputs)

    aaae(calculated, expected)
