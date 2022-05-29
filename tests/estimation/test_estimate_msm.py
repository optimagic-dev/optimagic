"""Most test exploit the special case where simulate_moments just returns parameters."""
import itertools
import warnings

import numpy as np
import pandas as pd
import pytest
from estimagic.estimation.estimate_msm import estimate_msm
from estimagic.shared.check_option_dicts import check_numdiff_options
from estimagic.shared.check_option_dicts import check_optimization_options
from numpy.testing import assert_array_almost_equal as aaae


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

test_cases = itertools.product(
    [_sim_pd, _sim_np, _sim_dict_pd, _sim_dict_np],  # simulate_moments
    [cov_np, cov_pd],  # moments_cov
    [{"algorithm": "scipy_lbfgsb"}, "scipy_lbfgsb"],  # optimize_options
)


@pytest.mark.parametrize("simulate_moments, moments_cov, optimize_options", test_cases)
def test_estimate_msm(simulate_moments, moments_cov, optimize_options):
    start_params = pd.DataFrame()
    start_params["value"] = [3, 2, 1]

    expected_params = pd.DataFrame()
    expected_params["value"] = np.zeros(3)

    # abuse simulate_moments to get empirical moments in correct format
    empirical_moments = simulate_moments(expected_params)
    if isinstance(empirical_moments, dict):
        empirical_moments = empirical_moments["simulated_moments"]

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
            optimize_options=optimize_options,
        )

    calculated_params = calculated["optimize_res"].params[["value"]]
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


def test_check_and_process_numdiff_options_with_invalid_entries():
    with pytest.raises(ValueError):
        check_numdiff_options({"func": lambda x: x}, "estimate_msm")


def test_check_and_process_optimize_options_with_invalid_entries():
    with pytest.raises(ValueError):
        check_optimization_options({"criterion": lambda x: x}, "estimate_msm")
