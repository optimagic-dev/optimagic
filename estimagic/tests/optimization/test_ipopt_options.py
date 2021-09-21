"""Test the different options of ipopt."""
from itertools import product

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from estimagic.examples.criterion_functions import sos_dict_criterion
from estimagic.optimization.optimize import minimize

options_and_expected = [
    ({}, None),
    ({"convergence.relative_criterion_tolerance": 1e-7}, None),
    ({"stopping.max_iterations": 1_100_000}, None),
    ({"mu_target": 1e-8}, None),
    ({"s_max": 200}, None),
    ({"stopping.max_wall_time_seconds": 200}, None),
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
    ({"diverging_iterates_tol": 1e5}, None),
    ({"nlp_lower_bound_inf": 1e5}, ValueError),
    ({"nlp_lower_bound_inf": -1e5}, None),
    ({"nlp_upper_bound_inf": -1e5}, ValueError),
    ({"nlp_upper_bound_inf": 1e10}, None),
    ({"fixed_variable_treatment": "relax_bounds"}, None),
    ({"fixed_variable_treatment": "non_existant"}, TypeError),
    ({"dependency_detector": "mumps"}, None),
    ({"dependency_detector": "non_existent"}, ValueError),
    ({"dependency_detection_with_rhs": "no"}, None),
    ({"dependency_detection_with_rhs": False}, None),
    ({"dependency_detection_with_rhs": "non_existent"}, ValueError),
    ({"kappa_d": 1e-7}, None),
    ({"bound_relax_factor": 1e-12}, None),
    ({"honor_original_bounds": "yes"}, None),
    ({"check_derivatives_for_naninf": True}, None),
    ({"jac_c_constant": True}, None),
    ({"jac_d_constant": True}, None),
    ({"hessian_constant": True}, None),
    ({"bound_push": 0.02}, None),
    ({"bound_frac": 0.02}, None),
    ({"slack_bound_push": 0.001}, None),
    ({"slack_bound_frac": 0.001}, None),
    ({"constr_mult_init_max": 5000}, None),
    ({"bound_mult_init_val": 1.2}, None),
    ({"bound_mult_init_method": "mu-based"}, None),
    ({"least_square_init_primal": "yes"}, None),
    ({"least_square_init_duals": "yes"}, None),
    ({"warm_start_init_point": "yes"}, None),
    ({"warm_start_same_structure": False}, None),
    ({"warm_start_bound_push": 0.002}, None),
    ({"warm_start_bound_frac": 0.002}, None),
    ({"warm_start_slack_bound_push": 0.0001}, None),
    ({"warm_start_slack_bound_frac": 0.002}, None),
    ({"warm_start_mult_bound_push": 0.002}, None),
    ({"warm_start_mult_init_max": 1e8}, None),
    ({"warm_start_entire_iterate": "yes"}, None),
    ({"replace_bounds": "yes"}, None),
    ({"skip_finalize_solution_call": "no"}, None),
    ({"timing_statistics": "yes"}, None),
    ({"mu_max_fact": 1500}, None),
    ({"mu_max": 100_500}, None),
    ({"mu_min": 1e-09}, None),
    ({"adaptive_mu_globalization": "kkt-error"}, None),
    ({"adaptive_mu_globalization": "not-implemented"}, TypeError),
    ({"adaptive_mu_kkterror_red_iters": 5}, None),
    ({"adaptive_mu_kkterror_red_fact": 0.9}, None),
    ({"adaptive_mu_kkterror_red_fact": 1.1}, TypeError),
    ({"filter_margin_fact": 1e-4}, None),
    ({"filter_max_margin": 0.5}, None),
    ({"adaptive_mu_restore_previous_iterate": False}, None),
    ({"adaptive_mu_monotone_init_factor": 0.9}, None),
    ({"adaptive_mu_kkt_norm_type": "max-norm"}, None),
    ({"mu_strategy": "adaptive"}, None),
    ({"mu_oracle": "probing"}, None),
    ({"mu_oracle": "loqo"}, None),
    ({"fixed_mu_oracle": "loqo"}, None),
    ({"mu_init": 0.2}, None),
    ({"barrier_tol_factor": 10.5}, None),
    ({"mu_linear_decrease_factor": 0.01}, None),
    ({"mu_superlinear_decrease_power": 1.2}, None),
    ({"mu_allow_fast_monotone_decrease": False}, None),
    ({"tau_min": 0.75}, None),
    ({"sigma_max": 200}, None),
    ({"sigma_min": 1e-8}, None),
    ({"quality_function_norm_type": "2-norm"}, None),
    ({"quality_function_centrality": "log"}, None),
    ({"quality_function_balancing_term": "cubic"}, None),
    ({"quality_function_max_section_steps": 10}, None),
    ({"quality_function_max_section_steps": 5.5}, None),
    ({"quality_function_section_sigma_tol": 0.02}, None),
    ({"quality_function_section_qf_tol": 0.5}, None),
    ({"quality_function_section_qf_tol": 1.1}, TypeError),
]

test_cases = product([sos_dict_criterion], options_and_expected)


@pytest.mark.parametrize("criterion, options_and_expected", test_cases)
def test_ipopt_algo_options(criterion, options_and_expected):
    algo_options, expected = options_and_expected
    start_params = pd.DataFrame()
    start_params["value"] = [1, 2, 3]

    if expected is None:
        res = minimize(
            criterion=criterion,
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
