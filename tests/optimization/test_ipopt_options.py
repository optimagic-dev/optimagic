"""Test the different options of ipopt."""
import numpy as np
import pytest
from estimagic.config import IS_CYIPOPT_INSTALLED
from estimagic.optimization.cyipopt_optimizers import ipopt
from numpy.testing import assert_array_almost_equal as aaae

test_cases = [
    {},
    {"convergence_relative_criterion_tolerance": 1e-7},
    {"stopping_max_iterations": 1_100_000},
    {"mu_target": 1e-8},
    {"s_max": 200},
    {"stopping_max_wall_time_seconds": 200},
    {"stopping_max_cpu_time": 1e10},
    {"dual_inf_tol": 2.5},
    {"constr_viol_tol": 1e-7},
    {"compl_inf_tol": 1e-7},
    #
    {"acceptable_iter": 15},
    {"acceptable_tol": 1e-5},
    {"acceptable_dual_inf_tol": 1e-5},
    {"acceptable_constr_viol_tol": 1e-5},
    {"acceptable_compl_inf_tol": 1e-5},
    {"acceptable_obj_change_tol": 1e5},
    {"diverging_iterates_tol": 1e5},
    {"nlp_lower_bound_inf": -1e5},
    {"nlp_upper_bound_inf": 1e10},
    {"fixed_variable_treatment": "relax_bounds"},
    {"dependency_detector": "mumps"},
    {"dependency_detection_with_rhs": "no"},
    {"dependency_detection_with_rhs": False},
    {"kappa_d": 1e-7},
    {"bound_relax_factor": 1e-12},
    {"honor_original_bounds": "yes"},
    {"check_derivatives_for_naninf": True},
    {"jac_c_constant": True},
    {"jac_d_constant": True},
    {"hessian_constant": True},
    # scaling
    {"nlp_scaling_method": None},
    {"obj_scaling_factor": 1.1},
    {"nlp_scaling_max_gradient": 200},
    {"nlp_scaling_obj_target_gradient": 0.2},
    {"nlp_scaling_constr_target_gradient": 0},
    {"nlp_scaling_constr_target_gradient": 2e-9},
    {"nlp_scaling_min_value": 1e-9},
    #
    {"bound_push": 0.02},
    {"bound_frac": 0.02},
    {"slack_bound_push": 0.001},
    {"slack_bound_frac": 0.001},
    {"constr_mult_init_max": 5000},
    {"bound_mult_init_val": 1.2},
    {"bound_mult_init_method": "mu-based"},
    {"least_square_init_primal": "yes"},
    {"least_square_init_duals": "yes"},
    {"warm_start_init_point": "yes"},
    {"warm_start_same_structure": False},
    {"warm_start_bound_push": 0.002},
    {"warm_start_bound_frac": 0.002},
    {"warm_start_slack_bound_push": 0.0001},
    {"warm_start_slack_bound_frac": 0.002},
    {"warm_start_mult_bound_push": 0.002},
    {"warm_start_mult_init_max": 1e8},
    {"warm_start_entire_iterate": "yes"},
    {"replace_bounds": "yes"},
    {"skip_finalize_solution_call": "no"},
    {"timing_statistics": "yes"},
    {"mu_max_fact": 1500},
    {"mu_max": 100_500},
    {"mu_min": 1e-09},
    {"adaptive_mu_globalization": "kkt-error"},
    {"adaptive_mu_kkterror_red_iters": 5},
    {"adaptive_mu_kkterror_red_fact": 0.9},
    {"filter_margin_fact": 1e-4},
    {"filter_max_margin": 0.5},
    {"adaptive_mu_restore_previous_iterate": False},
    {"adaptive_mu_monotone_init_factor": 0.9},
    {"adaptive_mu_kkt_norm_type": "max-norm"},
    {"mu_strategy": "adaptive"},
    {"mu_oracle": "probing"},
    {"mu_oracle": "loqo"},
    {"fixed_mu_oracle": "loqo"},
    {"mu_init": 0.2},
    {"barrier_tol_factor": 10.5},
    {"mu_linear_decrease_factor": 0.01},
    {"mu_superlinear_decrease_power": 1.2},
    {"mu_allow_fast_monotone_decrease": False},
    {"tau_min": 0.75},
    {"sigma_max": 200},
    {"sigma_min": 1e-8},
    {"quality_function_norm_type": "2-norm"},
    {"quality_function_centrality": "log"},
    {"quality_function_balancing_term": "cubic"},
    {"quality_function_max_section_steps": 10},
    {"quality_function_max_section_steps": 5.5},
    {"quality_function_section_sigma_tol": 0.02},
    {"quality_function_section_qf_tol": 0.5},
    {"line_search_method": "penalty"},
    {"alpha_red_factor": 0.8},
    {"accept_every_trial_step": True},
    {"accept_after_max_steps": 3},
    {"alpha_for_y": "max"},
    {"alpha_for_y_tol": 5},
    {"tiny_step_tol": 1e-15},
    {"tiny_step_y_tol": 0.02},
    {"watchdog_shortened_iter_trigger": 20},
    {"watchdog_trial_iter_max": 5},
    {"theta_max_fact": 2e5},
    {"theta_min_fact": 0.002},
    {"eta_phi": 0.3},
    {"delta": 0.9},
    {"s_phi": 2.2},
    {"s_theta": 1.5},
    {"gamma_phi": 1e-6},
    {"gamma_theta": 1e-5},
    {"alpha_min_frac": 0.08},
    {"max_soc": 5},
    {"kappa_soc": 0.9},
    {"obj_max_inc": 5.3},
    {"max_filter_resets": 10},
    {"filter_reset_trigger": 3},
    {"corrector_type": "affine"},
    {"skip_corr_if_neg_curv": True},
    {"skip_corr_in_monotone_mode": False},
    {"corrector_compl_avrg_red_fact": 3},
    {"corrector_compl_avrg_red_fact": 3.5},
    {"soc_method": 1},
    {"nu_init": 1e-5},
    {"nu_inc": 1e-5},
    {"rho": 0.2},
    {"kappa_sigma": 1e8},
    {"recalc_y": True},
    {"recalc_y_feas_tol": 1e-4},
    {"slack_move": 1e-11},
    {"constraint_violation_norm_type": "2-norm"},
    # step calculation
    {"mehrotra_algorithm": False},
    {"fast_step_computation": True},
    {"min_refinement_steps": 3},
    {"max_refinement_steps": 12},
    {"residual_ratio_max": 1e-9},
    {"residual_ratio_singular": 1e-4},
    {"residual_improvement_factor": 1.3},
    {"neg_curv_test_tol": 1e-11},
    {"neg_curv_test_reg": False},
    {"max_hessian_perturbation": 1e19},
    {"min_hessian_perturbation": 1e-19},
    {"perturb_inc_fact_first": 50.3},
    {"perturb_inc_fact": 4.4},
    {"perturb_dec_fact": 0.25},
    {"first_hessian_perturbation": 0.002},
    {"jacobian_regularization_value": 1e-7},
    {"jacobian_regularization_exponent": 0.2},
    {"perturb_always_cd": False},
    # restoration phase
    {"expect_infeasible_problem": False},
    {"expect_infeasible_problem_ctol": 0.005},
    {"expect_infeasible_problem_ytol": 1e7},
    {"start_with_resto": False},
    {"soft_resto_pderror_reduction_factor": 0.99},
    {"max_soft_resto_iters": 5},
    {"required_infeasibility_reduction": 0.8},
    {"max_resto_iter": 4_000_000},
    {"evaluate_orig_obj_at_resto_trial": False},
    {"resto_penalty_parameter": 830.4},
    {"resto_proximity_weight": 2.4},
    {"bound_mult_reset_threshold": 804.4},
    {"constr_mult_reset_threshold": 1.4},
    {"resto_failure_feasibility_threshold": 0.4},
    # hessian approximation
    {"limited_memory_aug_solver": "extended"},
    {"limited_memory_max_history": 5},
    {"limited_memory_update_type": "sr1"},
    {"limited_memory_initialization": "scalar2"},
    {"limited_memory_init_val": 0.5},
    {"limited_memory_init_val_max": 2e9},
    {"limited_memory_init_val_min": 2e-9},
    {"limited_memory_max_skipping": 4},
    {"limited_memory_special_for_resto": False},
    {"hessian_approximation_space": "all-variables"},
    # linear solver
    # using ma27, ma57, ma77, ma86 leads to remaining at the start values
    # using ma97 leads to segmentation fault
    {"linear_solver_options": {"mumps_pivtol": 1e-5}},
    {"linear_solver_options": {"linear_system_scaling": None}},
    {"linear_solver_options": {"ma86_scaling": None}},
    {"linear_solver_options": {"mumps_pivtol": 1e-7}},
    {"linear_solver_options": {"mumps_pivtolmax": 0.2}},
    {"linear_solver_options": {"mumps_mem_percent": 2000}},
    {"linear_solver_options": {"mumps_permuting_scaling": 5}},
    {"linear_solver_options": {"mumps_pivot_order": 5}},
    {"linear_solver_options": {"mumps_scaling": 74}},
    {"linear_solver_options": {"mumps_dep_tol": 0.1}},
]


def criterion_and_derivative(x, task, algorithm_info):
    if task == "criterion":
        return (x**2).sum()
    elif task == "derivative":
        return 2 * x
    else:
        raise ValueError(f"Unknown task: {task}")


@pytest.mark.skipif(not IS_CYIPOPT_INSTALLED, reason="cyipopt not installed.")
@pytest.mark.parametrize("algo_options", test_cases)
def test_ipopt_algo_options(algo_options):
    res = ipopt(
        criterion_and_derivative=criterion_and_derivative,
        x=np.array([1, 2, 3]),
        lower_bounds=np.array([-np.inf, -np.inf, -np.inf]),
        upper_bounds=np.array([np.inf, np.inf, np.inf]),
        **algo_options,
    )
    aaae(res["solution_x"], np.zeros(3), decimal=7)
