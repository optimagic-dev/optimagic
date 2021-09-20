"""Test the different options of ipopt."""
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from estimagic.examples.criterion_functions import sos_dict_criterion
from estimagic.optimization.optimize import minimize

test_cases = [
    ({}, None),
    ({"convergence.relative_criterion_tolerance": 1e-7}, None),
    ({"stopping.max_iterations": 1_100_000}, None),
    ({"mu_strategy": "adaptive"}, None),
    ({"s_max": 200}, None),
    ({"stopping.max_wall_time_seconds": 20}, None),
    ({"stopping.max_cpu_time": 1e10}, None),
    ({"dual_inf_tol": 2.5}, None),
    ({"dual_inf_tol": -2.5}, TypeError),
    ({"constr_viol_tol": 1e-7}, None),
    ({"compl_inf_tol": 1e-7}, None),
    #
    ({"acceptable_iter": 15}, None),
    ({"acceptable_tol": 1e-10}, ValueError),
    ({"acceptable_tol": 1e-5}, None),
    ({"acceptable_dual_inf_tol": 1e-5}, None),
    ({"acceptable_constr_viol_tol": 1e-5}, None),
    ({"acceptable_compl_inf_tol": 1e-5}, None),
    ({"acceptable_obj_change_tol": 1e5}, None),
]


@pytest.mark.parametrize("algo_options, expected", test_cases)
def test_ipopt_algo_options(algo_options, expected):
    start_params = pd.DataFrame()
    start_params["value"] = [1, 2, 3]

    if expected is None:
        res = minimize(
            criterion=sos_dict_criterion,
            params=start_params,
            algorithm="ipopt",
            algo_options=algo_options,
        )
        res_values = res["solution_params"]["value"].to_numpy()
        aaae(res_values, np.zeros(3), decimal=7)

    else:
        with pytest.raises(expected):
            res = minimize(
                criterion=sos_dict_criterion,
                params=start_params,
                algorithm="ipopt",
                algo_options=algo_options,
            )
