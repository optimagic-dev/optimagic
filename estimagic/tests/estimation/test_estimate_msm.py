"""Most test exploit the special case where simulate_moments just returns parameters."""
import itertools
import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from estimagic.estimation.estimate_msm import _check_and_process_derivative_options
from estimagic.estimation.estimate_msm import _check_and_process_minimize_options
from estimagic.estimation.estimate_msm import estimate_msm


def _sim_pd(params):
    return params["value"]


def _sim_np(params):
    return params["value"].to_numpy()


def _sim_dict_pd(params):
    return {"simulated_moments": params["value"], "other": "bla"}


def _sim_dict_np(params):
    return {"simulated_moments": params["value"].to_numpy(), "other": "bla"}


cov_np = np.diag([1, 2, 3.0])
cov_pd = pd.DataFrame(cov_np)

funcs = [_sim_pd, _sim_np, _sim_dict_pd, _sim_dict_np]
covs = [cov_np, cov_pd]


test_cases = list(itertools.product(funcs, covs))


@pytest.mark.parametrize("simulate_moments, moments_cov", test_cases)
def test_estimate_msm(simulate_moments, moments_cov):
    start_params = pd.DataFrame()
    start_params["value"] = [3, 2, 1]

    expected_params = pd.DataFrame()
    expected_params["value"] = np.zeros(3)

    # abuse simulate_moments to get empirical moments in correct format
    empirical_moments = simulate_moments(expected_params)
    if isinstance(empirical_moments, dict):
        empirical_moments = empirical_moments["simulated_moments"]

    minimize_options = {"algorithm": "scipy_lbfgsb"}

    # catching warnings is necessary because the very special case with diagonal
    # weighting and diagonal jacobian leads to singular matrices while calculating
    # sensitivity to removal of moments.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Standard matrix inversion failed")
        calculated = estimate_msm(
            simulate_moments=simulate_moments,
            empirical_moments=empirical_moments,
            moments_cov=moments_cov,
            params=start_params,
            minimize_options=minimize_options,
        )

    calculated_params = calculated["minimize_res"]["solution_params"][["value"]]
    # check that minimization works
    aaae(calculated_params["value"].to_numpy(), expected_params["value"].to_numpy())

    # check that cov works
    calculated_cov = calculated["cov"]
    if isinstance(calculated_cov, pd.DataFrame):
        calculated_cov = calculated_cov.to_numpy()

    # this works only in the very special case with diagonal moments cov and
    # jac = identity matrix
    expected_cov = np.diag([1, 2, 3])
    aaae(calculated_cov, expected_cov)


def test_check_and_process_numdiff_options_differentiated_but_not_minimized():
    with pytest.raises(ValueError):
        _check_and_process_derivative_options({}, pd.DataFrame(), False)


def test_check_and_process_numdiff_options_with_invalid_entries():
    with pytest.raises(ValueError):
        _check_and_process_derivative_options({"func": lambda x: x}, None, False)


def test_check_and_process_minimize_options_with_invalid_entries():
    with pytest.raises(ValueError):
        _check_and_process_minimize_options({"criterion": lambda x: x})
