"""Implement cyipopt's Interior Point Optimizer."""
import functools

from estimagic.config import IS_CYIPOPT_INSTALLED
from estimagic.optimization.algo_options import CONVERGENCE_RELATIVE_CRITERION_TOLERANCE
from estimagic.optimization.algo_options import STOPPING_MAX_ITERATIONS
from estimagic.optimization.scipy_optimizers import get_scipy_bounds
from estimagic.optimization.scipy_optimizers import process_scipy_result

try:
    import cyipopt
except ImportError:
    pass

DEFAULT_ALGO_INFO = {
    "primary_criterion_entry": "value",
    "parallelizes": False,
    "needs_scaling": False,
}

DEFAULT_LINEAR_SOLVER_OPTIONS = {
    "linear_scaling_on_demand": "yes",
    # ma27
    "ma27_pivtol": 1e-8,
    "ma27_pivtolmax": 0.001,
    "ma27_liw_init_factor": 5.0,
    "ma27_la_init_factor": 5.0,
    "ma27_meminc_factor": 2.0,
    "ma27_skip_inertia_check": "no",
    "ma27_ignore_singularity": "no",
    # ma57
    "ma57_pivtol": 1e-8,
    "ma57_pivtolmax": 0.0001,
    "ma57_pre_alloc": 1.05,
    "ma57_pivot_order": 5,
    "ma57_automatic_scaling": "no",
    "ma57_block_size": 16,
    "ma57_node_amalgamation": 16.0,
    "ma57_small_pivot_flag": 0.0,
    # ma77
    "ma77_buffer_lpage": 4096,
    "ma77_buffer_npage": 1600,
    "ma77_file_size": 2097152,
    "ma77_maxstore": 0,
    "ma77_nemin": 8,
    "ma77_small": 1e-20,
    "ma77_static": 0.0,
    "ma77_u": 1e-8,
    "ma77_umax": 0.0001,
    "ma77_order": "metis",
    # ma86
    "ma86_nemin": 32,
    "ma86_small": 1e-20,
    "ma86_static": 0.0,
    "ma86_u": 1e-8,
    "ma86_umax": 0.0001,
    "ma86_scaling": "mc64",
    "ma86_order": "auto",
    # ma97
    "ma97_nemin": 8,
    "ma97_small": 1e-20,
    "ma97_u": 1e-8,
    "ma97_umax": 0.0001,
    "ma97_scaling": "dynamic",
    "ma97_scaling1": "mc64",
    "ma97_switch1": "od_hd_reuse",
    "ma97_scaling2": "mc64",
    "ma97_switch2": "never",
    "ma97_scaling3": "mc64",
    "ma97_switch3": "never",
    "ma97_order": "auto",
    "ma97_solve_blas3": "no",
    # paradiso
    "pardiso_matching_strategy": "complete+2x2",
    "pardiso_redo_symbolic_fact_only_if_inertia_wrong": "no",
    "pardiso_repeated_perturbation_means_singular": "no",
    "pardiso_skip_inertia_check": "no",
    "pardiso_max_iterative_refinement_steps": 0,
    "pardiso_order": "metis",
    "pardiso_max_iter": 500,
    "pardiso_iter_relative_tol": 1e-6,
    "pardiso_iter_coarse_size": 5000,
    "pardiso_iter_max_levels": 10,
    "pardiso_iter_dropping_factor": 0.5,
    "pardiso_iter_dropping_schur": 0.1,
    "pardiso_iter_max_row_fill": 10_000_000,
    "pardiso_iter_inverse_norm_factor": 1e6,
    "pardiso_iterative": "no",
    "pardiso_max_droptol_corrections": 4,
    # paradiso MKL
    "pardisomkl_matching_strategy": "complete+2x2",
    "pardisomkl_redo_symbolic_fact_only_if_inertia_wrong": "no",
    "pardisomkl_repeated_perturbation_means_singular": "no",
    "pardisomkl_skip_inertia_check": "no",
    "pardisomkl_max_iterative_refinement_steps": 1,
    "pardisomkl_order": "metis",
    # SPRAL
    "spral_cpu_block_size": 256,
    "spral_gpu_perf_coeff": 1.0,
    "spral_ignore_numa": "yes",
    "spral_max_load_inbalance": 1.2,
    "spral_min_gpu_work": 5e9,
    "spral_nemin": 32,
    "spral_order": "matching",
    "spral_pivot_method": "block",
    "spral_scaling": "matching",
    "spral_scaling_1": "matching",
    "spral_scaling_2": "mc64",
    "spral_scaling_3": "none",
    "spral_switch_1": "at_start",
    "spral_switch_2": "on_demand",
    "spral_switch_3": "never",
    "spral_small": 1e-20,
    "spral_small_subtree_threshold": 4e6,
    "spral_u": 1e-8,
    "spral_umax": 0.0001,
    "spral_use_gpu": "yes",
    # WSMP
    "wsmp_num_threads": 1,
    "wsmp_ordering_option": 1,
    "wsmp_ordering_option2": 1,
    "wsmp_pivtol": 0.0001,
    "wsmp_pivtolmax": 0.1,
    "wsmp_scaling": 0,
    "wsmp_singularity_threshold": 1e-18,
    "wsmp_write_matrix_iteration": -1,
    "wsmp_skip_inertia_check": "no",
    "wsmp_no_pivoting": "no",
    "wsmp_max_iter": 1000,
    "wsmp_inexact_droptol": 0.0,
    "wsmp_inexact_fillin_limit": 0.0,
    # mumps
    "mumps_pivtol": 1e-6,
    "mumps_pivtolmax": 0.1,
    "mumps_mem_percent": 1000,
    "mumps_permuting_scaling": 7,
    "mumps_pivot_order": 7,
    "mumps_scaling": 77,
    "mumps_dep_tol": 0.0,
    # ma28
    "ma28_pivtol": 0.01,
}


def ipopt(
    criterion_and_derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    # convergence criteria
    convergence_relative_criterion_tolerance=CONVERGENCE_RELATIVE_CRITERION_TOLERANCE,
    dual_inf_tol=1.0,
    constr_viol_tol=0.0001,
    compl_inf_tol=0.0001,
    #
    s_max=100.0,
    mu_target=0.0,
    # stopping criteria
    stopping_max_iterations=STOPPING_MAX_ITERATIONS,
    stopping_max_wall_time_seconds=1e20,
    stopping_max_cpu_time=1e20,
    # acceptable heuristic
    acceptable_iter=15,
    acceptable_tol=1e-6,
    acceptable_dual_inf_tol=1e-10,
    acceptable_constr_viol_tol=0.01,
    acceptable_compl_inf_tol=0.01,
    acceptable_obj_change_tol=1e20,
    #
    diverging_iterates_tol=1e20,
    nlp_lower_bound_inf=-1e19,
    nlp_upper_bound_inf=1e19,
    fixed_variable_treatment="make_parameter",
    dependency_detector=None,
    dependency_detection_with_rhs=False,
    # bounds
    kappa_d=1e-5,
    bound_relax_factor=1e-8,
    honor_original_bounds=False,
    # derivatives
    check_derivatives_for_naninf=False,
    # not sure if we should support the following:
    jac_c_constant=False,
    jac_d_constant=False,
    hessian_constant=False,
    # initialization
    bound_push=0.01,
    bound_frac=0.01,
    slack_bound_push=0.01,
    slack_bound_frac=0.01,
    constr_mult_init_max=1000,
    bound_mult_init_val=1,
    bound_mult_init_method="constant",
    least_square_init_primal=False,
    least_square_init_duals=False,
    # warm start
    warm_start_init_point=False,
    warm_start_same_structure=False,
    warm_start_bound_push=0.001,
    warm_start_bound_frac=0.001,
    warm_start_slack_bound_push=0.001,
    warm_start_slack_bound_frac=0.001,
    warm_start_mult_bound_push=0.001,
    warm_start_mult_init_max=1e6,
    warm_start_entire_iterate=False,
    warm_start_target_mu=0.0,
    # miscellaneous
    option_file_name="",
    replace_bounds=False,
    skip_finalize_solution_call=False,
    timing_statistics=False,
    # barrier parameter update
    mu_max_fact=1000,
    mu_max=100_000,
    mu_min=1e-11,
    adaptive_mu_globalization="obj-constr-filter",
    adaptive_mu_kkterror_red_iters=4,
    adaptive_mu_kkterror_red_fact=0.9999,
    filter_margin_fact=1e-5,
    filter_max_margin=1,
    adaptive_mu_restore_previous_iterate=False,
    adaptive_mu_monotone_init_factor=0.8,
    adaptive_mu_kkt_norm_type="2-norm-squared",
    mu_strategy="monotone",
    mu_oracle="quality-function",
    fixed_mu_oracle="average_compl",
    mu_init=0.1,
    barrier_tol_factor=10,
    mu_linear_decrease_factor=0.2,
    mu_superlinear_decrease_power=1.5,
    mu_allow_fast_monotone_decrease=True,
    tau_min=0.99,
    sigma_max=100,
    sigma_min=1e-6,
    quality_function_norm_type="2-norm-squared",
    quality_function_centrality=None,
    quality_function_balancing_term=None,
    quality_function_max_section_steps=8,
    quality_function_section_sigma_tol=0.01,
    quality_function_section_qf_tol=0.0,
    # line search
    line_search_method="filter",
    alpha_red_factor=0.5,
    accept_every_trial_step=False,
    accept_after_max_steps=-1,
    alpha_for_y="primal",
    alpha_for_y_tol=10,
    tiny_step_tol=2.22045 * 1e-15,
    tiny_step_y_tol=0.01,
    watchdog_shortened_iter_trigger=10,
    watchdog_trial_iter_max=3,
    theta_max_fact=10_000,
    theta_min_fact=0.0001,
    eta_phi=1e-8,
    delta=1,
    s_phi=2.3,
    s_theta=1.1,
    gamma_phi=1e-8,
    gamma_theta=1e-5,
    alpha_min_frac=0.05,
    max_soc=4,
    kappa_soc=0.99,
    obj_max_inc=5,
    max_filter_resets=5,
    filter_reset_trigger=5,
    corrector_type=None,
    skip_corr_if_neg_curv=True,
    skip_corr_in_monotone_mode=True,
    corrector_compl_avrg_red_fact=1,
    soc_method=0,
    nu_init=1e-6,
    nu_inc=0.0001,
    rho=0.1,
    kappa_sigma=1e10,
    recalc_y=False,
    recalc_y_feas_tol=1e-6,
    slack_move=1.81899 * 1e-12,
    constraint_violation_norm_type="1-norm",
    # step calculation
    mehrotra_algorithm=False,
    fast_step_computation=False,
    min_refinement_steps=1,
    max_refinement_steps=10,
    residual_ratio_max=1e-10,
    residual_ratio_singular=1e-5,
    residual_improvement_factor=1,
    neg_curv_test_tol=0,
    neg_curv_test_reg=True,
    max_hessian_perturbation=1e20,
    min_hessian_perturbation=1e-20,
    perturb_inc_fact_first=100,
    perturb_inc_fact=8,
    perturb_dec_fact=0.333333,
    first_hessian_perturbation=0.0001,
    jacobian_regularization_value=1e-8,
    jacobian_regularization_exponent=0.25,
    perturb_always_cd=False,
    # restoration phase
    expect_infeasible_problem=False,
    expect_infeasible_problem_ctol=0.001,
    expect_infeasible_problem_ytol=1e8,
    start_with_resto=False,
    soft_resto_pderror_reduction_factor=0.9999,
    max_soft_resto_iters=10,
    required_infeasibility_reduction=0.9,
    max_resto_iter=3_000_000,
    evaluate_orig_obj_at_resto_trial=True,
    resto_penalty_parameter=1000,
    resto_proximity_weight=1,
    bound_mult_reset_threshold=1000,
    constr_mult_reset_threshold=0,
    resto_failure_feasibility_threshold=None,
    # hessian approximation
    limited_memory_aug_solver="sherman-morrison",
    limited_memory_max_history=6,
    limited_memory_update_type="bfgs",
    limited_memory_initialization="scalar1",
    limited_memory_init_val=1,
    limited_memory_init_val_max=1e8,
    limited_memory_init_val_min=1e-8,
    limited_memory_max_skipping=2,
    limited_memory_special_for_resto=False,
    hessian_approximation_space="nonlinear-variables",
    # linear solver
    linear_solver="mumps",
    linear_solver_options=None,
):
    """Minimize a scalar function using the Interior Point Optimizer.

    This implementation of the Interior Point Optimizer (:cite:`Waechter2005`,
    :cite:`Waechter2005a`, :cite:`Waechter2005b`, :cite:`Nocedal2009`) relies on
    `cyipopt <https://cyipopt.readthedocs.io/en/latest/index.html>`_, a Python
    wrapper for the `Ipopt optimization package
    <https://coin-or.github.io/Ipopt/index.html>`_.

    There are two levels of termination criteria. If the usual "desired"
    tolerances (see tol, dual_inf_tol etc) are satisfied at an iteration, the
    algorithm immediately terminates with a success message. On the other hand,
    if the algorithm encounters "acceptable_iter" many iterations in a row that
    are considered "acceptable", it will terminate before the desired
    convergence tolerance is met. This is useful in cases where the algorithm
    might not be able to achieve the "desired" level of accuracy.

    The options are analogous to the ones in the `ipopt documentation
    <https://coin-or.github.io/Ipopt/OPTIONS.html#>`_ with the exception of the
    linear solver options which are here bundled into a dictionary. Any argument
    that takes "yes" and "no" in the ipopt documentation can also be passed as a
    `True` and `False`, respectively. and any option that accepts "none" in ipopt
    accepts a Python `None`.

    - convergence.relative_criterion_tolerance (float): The algorithm terminates
        successfully, if the (scaled) non linear programming error becomes
        smaller than this value.

    - mu_target: Desired value of complementarity. Usually, the barrier
        parameter is driven to zero and the termination test for complementarity
        is measured with respect to zero complementarity. However, in some cases
        it might be desired to have Ipopt solve barrier problem for strictly
        positive value of the barrier parameter. In this case, the value of
        "mu_target" specifies the final value of the barrier parameter, and the
        termination tests are then defined with respect to the barrier problem
        for this value of the barrier parameter. The valid range for this real
        option is 0 ≤ mu_target and its default value is 0.
    - s_max (float): Scaling threshold for the NLP error.

    - stopping.max_iterations (int):  If the maximum number of iterations is
        reached, the optimization stops, but we do not count this as successful
        convergence. The difference to ``max_criterion_evaluations`` is that one
        iteration might need several criterion evaluations, for example in a
        line search or to determine if the trust region radius has to be shrunk.
    - stopping.max_wall_time_seconds (float): Maximum number of walltime clock
        seconds.
    - stopping.max_cpu_time (float): Maximum number of CPU seconds. A limit on
        CPU seconds that Ipopt can use to solve one problem. If during the
        convergence check this limit is exceeded, Ipopt will terminate with a
        corresponding message. The valid range for this real option is 0 <
        max_cpu_time and its default value is 1e20.

    - dual_inf_tol (float): Desired threshold for the dual infeasibility.
        Absolute tolerance on the dual infeasibility. Successful termination
        requires that the max-norm of the (unscaled) dual infeasibility is less
        than this threshold. The valid range for this real option is 0 <
        dual_inf_tol and its default value is 1.
    - constr_viol_tol (float): Desired threshold for the constraint and variable
        bound violation. Absolute tolerance on the constraint and variable bound
        violation. Successful termination requires that the max-norm of the
        (unscaled) constraint violation is less than this threshold. If option
        bound_relax_factor is not zero 0, then Ipopt relaxes given variable
        bounds. The value of constr_viol_tol is used to restrict the absolute
        amount of this bound relaxation. The valid range for this real option is
        0 < constr_viol_tol and its default value is 0.0001.
    - compl_inf_tol (float): Desired threshold for the complementarity
        conditions. Absolute tolerance on the complementarity. Successful
        termination requires that the max-norm of the (unscaled) complementarity
        is less than this threshold. The valid range for this real option is 0 <
        compl_inf_tol and its default value is 0.0001.

    - acceptable_iter (int): Number of "acceptable" iterates before triggering
        termination. If the algorithm encounters this many successive
        "acceptable" iterates (see above on the acceptable heuristic), it
        terminates, assuming that the problem has been solved to best possible
        accuracy given round-off. If it is set to zero, this heuristic is
        disabled. The valid range for this integer option is 0 ≤
        acceptable_iter.
    - acceptable_tol (float):"Acceptable" convergence tolerance (relative).
        Determines which (scaled) overall optimality error is considered to be
        "acceptable". The valid range for this real option is 0 <
        acceptable_tol.
    - acceptable_dual_inf_tol (float):  "Acceptance" threshold for the dual
        infeasibility. Absolute tolerance on the dual infeasibility.
        "Acceptable" termination requires that the (max-norm of the unscaled)
        dual infeasibility is less than this threshold; see also acceptable_tol.
        The valid range for this real option is 0 < acceptable_dual_inf_tol and
        its default value is 10+10.
    - acceptable_constr_viol_tol (float): "Acceptance" threshold for the
        constraint violation. Absolute tolerance on the constraint violation.
        "Acceptable" termination requires that the max-norm of the (unscaled)
        constraint violation is less than this threshold; see also
        acceptable_tol. The valid range for this real option is 0 <
        acceptable_constr_viol_tol and its default value is 0.01.
    - acceptable_compl_inf_tol (float): "Acceptance" threshold for the
        complementarity conditions. Absolute tolerance on the complementarity.
        "Acceptable" termination requires that the max-norm of the (unscaled)
        complementarity is less than this threshold; see also acceptable_tol.
        The valid range for this real option is 0 < acceptable_compl_inf_tol and
        its default value is 0.01.
    - acceptable_obj_change_tol (float): "Acceptance" stopping criterion based
        on objective function change. If the relative change of the objective
        function (scaled by Max(1,|f(x)|)) is less than this value, this part of
        the acceptable tolerance termination is satisfied; see also
        acceptable_tol. This is useful for the quasi-Newton option, which has
        trouble to bring down the dual infeasibility. The valid range for this
        real option is 0 ≤ acceptable_obj_change_tol and its default value is
        10+20.

    - diverging_iterates_tol (float): Threshold for maximal value of primal
        iterates. If any component of the primal iterates exceeded this value
        (in absolute terms), the optimization is aborted with the exit message
        that the iterates seem to be diverging. The valid range for this real
        option is 0 < diverging_iterates_tol and its default value is 10+20.
    - nlp_lower_bound_inf (float): any bound less or equal this value will be
        considered -inf (i.e. not lwer bounded). The valid range for this real
        option is unrestricted and its default value is -10+19.
    - nlp_upper_bound_inf (float): any bound greater or this value will be
        considered +inf (i.e. not upper bunded). The valid range for this real
        option is unrestricted and its default value is 10+19.
    - fixed_variable_treatment (str): Determines how fixed variables should be
        handled. The main difference between those options is that the starting
        point in the "make_constraint" case still has the fixed variables at
        their given values, whereas in the case "make_parameter(_nodual)" the
        functions are always evaluated with the fixed values for those
        variables. Also, for "relax_bounds", the fixing bound constraints are
        relaxed (according to" bound_relax_factor"). For all but
        "make_parameter_nodual", bound multipliers are computed for the fixed
        variables. The default value for this string option is "make_parameter".
        Possible values:
        - make_parameter: Remove fixed variable from optimization variables
        - make_parameter_nodual: Remove fixed variable from optimization
            variables and do not compute bound multipliers for fixed variables
        - make_constraint: Add equality constraints fixing variables
        - relax_bounds: Relax fixing bound constraints
    - dependency_detector (str): Indicates which linear solver should be used to
        detect lnearly dependent equality constraints. This is experimental and
        does not work well. The default value for this string option is "none".
        Possible values:
        - none: don't check; no extra work at beginning
        - mumps: use MUMPS
        - wsmp: use WSMP
        - ma28: use MA28
    - dependency_detection_with_rhs (str or bool): Indicates if the right hand
        sides of the constraints should be considered in addition to gradients
        during dependency detection. The default value for this string option is
        "no". Possible values: 'yes', 'no', True, False.

    - kappa_d (float): Weight for linear damping term (to handle one-sided
        bounds). See Section 3.7 in implementation paper. The valid range for
        this real option is 0 ≤ kappa_d and its default value is 10-05.
    - bound_relax_factor (float): Factor for initial relaxation of the bounds.
        Before start of the optimization, the bounds given by the user are
        relaxed. This option sets the factor for this relaxation. Additional,
        the constraint violation tolerance constr_viol_tol is used to bound the
        relaxation by an absolute value. If it is set to zero, then then bounds
        relaxation is disabled. See Eqn.(35) in implementation paper. Note that
        the constraint violation reported by Ipopt at the end of the solution
        process does not include violations of the original (non-relaxed)
        variable bounds. See also option honor_original_bounds. The valid range
        for this real option is 0 ≤ bound_relax_factor and its default value is
        10-08.
    - honor_original_bounds (str or bool): Indicates whether final points should
        be projected into original bunds. Ipopt might relax the bounds during
        the optimization (see, e.g., option "bound_relax_factor"). This option
        determines whether the final point should be projected back into the
        user-provide original bounds after the optimization. Note that
        violations of constraints and complementarity reported by Ipopt at the
        end of the solution process are for the non-projected point. The default
        value for this string option is "no". Possible values: 'yes', 'no',
        True, False

    - check_derivatives_for_naninf (str): whether to check for NaN / inf in the
        derivative matrices. Activating this option will cause an error if an
        invalid number is detected in the constraint Jacobians or the Lagrangian
        Hessian. If this is not activated, the test is skipped, and the
        algorithm might proceed with invalid numbers and fail. If test is
        activated and an invalid number is detected, the matrix is written to
        output with print_level corresponding to J_MORE_DETAILED; so beware of
        large output! The default value for this string option is "no".
    - jac_c_constant (str or bool): Indicates whether to assume that all
        equality constraints are linear Activating this option will cause Ipopt
        to ask for the Jacobian of the equality constraints only once from the
        NLP and reuse this information later. The default value for this string
        option is "no". Possible values: yes, no, True, False.
    - jac_d_constant (str or bool): Indicates whether to assume that all
        inequality constraints are linear Activating this option will cause
        Ipopt to ask for the Jacobian of the inequality constraints only once
        from the NLP and reuse this information later. The default value for
        this string option is "no". Possible values: yes, no, True, False
    - hessian_constant: Indicates whether to assume the problem is a QP
        (quadratic objective, linear constraints). Activating this option will
        cause Ipopt to ask for the Hessian of the Lagrangian function only once
        from the NLP and reuse this information later. The default value for
        this string option is "no". Possible values: yes, no, True, False

    - bound_push (float): Desired minimum absolute distance from the initial
        point to bound. Determines how much the initial point might have to be
        modified in order to be sufficiently inside the bounds (together with
        "bound_frac"). (This is kappa_1 in Section 3.6 of implementation paper.)
        The valid range for this real option is 0 < bound_push and its default
        value is 0.01.
    - bound_frac (float): Desired minimum relative distance from the initial
        point to bound. Determines how much the initial point might have to be
        modified in order to be sufficiently inside the bounds (together with
        "bound_push"). (This is kappa_2 in Section 3.6 of implementation paper.)
        The valid range for this real option is 0 < bound_frac ≤ 0.5 and its
        default value is 0.01.
    - slack_bound_push (float): Desired minimum absolute distance from the
        initial slack to bound. Determines how much the initial slack variables
        might have to be modified in order to be sufficiently inside the
        inequality bounds (together with "slack_bound_frac"). (This is kappa_1
        in Section 3.6 of implementation paper.) The valid range for this real
        option is 0 < slack_bound_push and its default value is 0.01.
    - slack_bound_frac (float): Desired minimum relative distance from the
        initial slack to bound. Determines how much the initial slack variables
        might have to be modified in order to be sufficiently inside the
        inequality bounds (together with "slack_bound_push"). (This is kappa_2
        in Section 3.6 of implementation paper.) The valid range for this real
        option is 0 < slack_bound_frac ≤ 0.5 and its default value is 0.01.
    - constr_mult_init_max (float): Maximum allowed least-square guess of
        constraint multipliers. Determines how large the initial least-square
        guesses of the constraint multipliers are allowed to be (in max-norm).
        If the guess is larger than this value, it is discarded and all
        constraint multipliers are set to zero. This options is also used when
        initializing the restoration phase. By default,
        "resto.constr_mult_init_max" (the one used in RestoIterateInitializer)
        is set to zero. The valid range for this real option is 0 ≤
        constr_mult_init_max and its default value is 1000.
    - bound_mult_init_val (float): Initial value for the bound multipliers. All
        dual variables corresponding to bound constraints are initialized to
        this value. The valid range for this real option is 0 <
        bound_mult_init_val and its default value is 1.
    - bound_mult_init_method (str): Initialization method for bound multipliers
        This option defines how the iterates for the bound multipliers are
        initialized. If "constant" is chosen, then all bound multipliers are
        initialized to the value of "bound_mult_init_val". If "mu-based" is
        chosen, the each value is initialized to the the value of "mu_init"
        divided by the corresponding slack variable. This latter option might be
        useful if the starting point is close to the optimal solution. The
        default value for this string option is "constant". Possible values:
        - "constant": set all bound multipliers to the value of
            bound_mult_init_val
        - "mu-based": initialize to mu_init/x_slack
    - least_square_init_primal (str or bool): Least square initialization of the
        primal variables. If set to yes, Ipopt ignores the user provided point
        and solves a least square problem for the primal variables (x and s) to
        fit the linearized equality and inequality constraints.This might be
        useful if the user doesn't know anything about the starting point, or
        for solving an LP or QP. The default value for this string option is
        "no".  Possible values:
        - "no": take user-provided point
        - "yes": overwrite user-provided point with least-square estimates
    - least_square_init_duals: Least square initialization of all dual variables
        If set to yes, Ipopt tries to compute least-square multipliers
        (considering ALL dual variables). If successful, the bound multipliers
        are possibly corrected to be at least bound_mult_init_val. This might be
        useful if the user doesn't know anything about the starting point, or
        for solving an LP or QP. This overwrites option
        "bound_mult_init_method". The default value for this string option is
        "no". Possible values:
        - "no": use bound_mult_init_val and least-square equality constraint
            multipliers
        - "yes": overwrite user-provided point with least-square estimates
    - warm_start_init_point (str or bool): Warm-start for initial point
        Indicates whether this optimization should use a warm start
        initialization, where values of primal and dual variables are given
        (e.g., from a previous optimization of a related problem.) The default
        value for this string option is "no". Possible values:
        - "no" or False: do not use the warm start initialization
        - "yes" or True: use the warm start initialization
    - warm_start_same_structure (str or bool): Advanced feature! Indicates
        whether a problem with a structure identical t the previous one is to be
        solved. If enabled, then the algorithm assumes that an NLP is now to be
        solved whose structure is identical to one that already was considered
        (with the same NLP object). The default value for this string option is
        "no". Possible values: yes, no, True, False.
    - warm_start_bound_push (float): same as bound_push for the regular
        initializer. The valid range for this real option is 0 <
        warm_start_bound_push and its default value is 0.001.
    - warm_start_bound_frac (float): same as bound_frac for the regular
        initializer The valid range for this real option is 0 <
        warm_start_bound_frac ≤ 0.5 and its default value is 0.001.
    - warm_start_slack_bound_push (float): same as slack_bound_push for the
        regular initializer The valid range for this real option is 0 <
        warm_start_slack_bound_push and its default value is 0.001.
    - warm_start_slack_bound_frac (float): same as slack_bound_frac for the
        regular initializer The valid range for this real option is 0 <
        warm_start_slack_bound_frac ≤ 0.5 and its default value is 0.001.
    - warm_start_mult_bound_push (float): same as mult_bound_push for the
        regular initializer The valid range for this real option is 0 <
        warm_start_mult_bound_push and its default value is 0.001.
    - warm_start_mult_init_max (float): Maximum initial value for the equality
        multipliers. The valid range for this real option is unrestricted and
        its default value is 10+06.
    - warm_start_entire_iterate (str or bool): Tells algorithm whether to use
        the GetWarmStartIterate method in the NLP. The default value for this
        string option is "no".  Possible values:
        - "no": call GetStartingPoint in the NLP
        - "yes": call GetWarmStartIterate in the NLP
    - warm_start_target_mu (float): Advanced and experimental! The valid range
        for this real option is unrestricted and its default value is 0.

    - option_file_name (str): File name of options file. By default, the name of
        the Ipopt options file is "ipopt.opt" - or something else if specified
        in the IpoptApplication::Initialize call. If this option is set by
        SetStringValue BEFORE the options file is read, it specifies the name of
        the options file. It does not make any sense to specify this option
        within the options file. Setting this option to an empty string disables
        reading of an options file.
    - replace_bounds (bool or str): Whether all variable bounds should be
        replaced by inequality constraints. This option must be set for the
        inexact algorithm. The default value for this string option is "no".
        Possible values: "yes", "no", True, False.
    - skip_finalize_solution_call (str or bool): Whether a call to
        NLP::FinalizeSolution after optimization should be suppressed.    In
        some Ipopt applications, the user might want to call the
        FinalizeSolution method separately. Setting this option to "yes" will
        cause the IpoptApplication object to suppress the default call to that
        method. The default value for this string option is "no". Possible
        values: "yes", "no", True, False
    - timing_statistics (str or bool): Indicates whether to measure time spend
        in components of Ipopt and NLP evaluation.  The overall algorithm time
        is unaffected by this option. The default value for this string option
        is "no". Possible values: "yes", "no", True, False

    - mu_max_fact (float): Factor for initialization of maximum value for
        barrier parameter. This option determines the upper bound on the barrier
        parameter. This upper bound is computed as the average complementarity
        at the initial point times the value of this option. (Only used if
        option "mu_strategy" is chosen as "adaptive".) The valid range for this
        real option is 0 < mu_max_fact and its default value is 1000.
    - mu_max (float): Maximum value for barrier parameter. This option specifies
        an upper bound on the barrier parameter in the adaptive mu selection
        mode. If this option is set, it overwrites the effect of mu_max_fact.
        (Only used if option "mu_strategy" is chosen as "adaptive".) The valid
        range for this real option is 0 < mu_max and its default value is
        100000.
    - mu_min (float): Minimum value for barrier parameter. This option specifies
        the lower bound on the barrier parameter in the adaptive mu selection
        mode. By default, it is set to the minimum of 1e-11 and
        min("tol","compl_inf_tol")/("barrier_tol_factor"+1), which should be a
        reasonable value. (Only used if option "mu_strategy" is chosen as
        "adaptive".) The valid range for this real option is 0 < mu_min and its
        default value is 10-11.
    - adaptive_mu_globalization (str): Globalization strategy for the adaptive
        mu selection mode. To achieve global convergence of the adaptive
        version, the algorithm has to switch to the monotone mode
        (Fiacco-McCormick approach) when convergence does not seem to appear.
        This option sets the criterion used to decide when to do this switch.
        (Only used if option "mu_strategy" is chosen as "adaptive".) The default
        value for this string option is "obj-constr-filter". Possible values:
        - "kkt-error": nonmonotone decrease of kkt-error
        - "obj-constr-filter": 2-dim filter for objective and constraint
            violation
        - "never-monotone-mode": disables globalization.
    - adaptive_mu_kkterror_red_iters (float): advanced feature! Maximum number
        of iterations requiring sufficient progress. For the "kkt-error" based
        globalization strategy, sufficient progress must be made for
        "adaptive_mu_kkterror_red_iters" iterations. If this number of
        iterations is exceeded, the globalization strategy switches to the
        monotone mode. The valid range for this integer option is 0 ≤
        adaptive_mu_kkterror_red_iters and its default value is 4.
    - adaptive_mu_kkterror_red_fact (float): advanced feature! Sufficient
        decrease factor for "kkt-error" globalization strategy. For the
        "kkt-error" based globalization strategy, the error must decrease by
        this factor to be deemed sufficient decrease. The valid range for this
        real option is 0 < adaptive_mu_kkterror_red_fact < 1 and its default
        value is 0.9999.
    - filter_margin_fact (float): advanced feature! Factor determining width of
        margin for obj-constr-filter adaptive globalization strategy. When using
        the adaptive globalization strategy, "obj-constr-filter", sufficient
        progress for a filter entry is defined as follows: (new obj) < (filter
        obj) - filter_margin_fact*(new constr-viol) OR (new constr-viol) <
        (filter constr-viol) - filter_margin_fact*(new constr-viol). For the
        description of the "kkt-error-filter" option see "filter_max_margin".
        The valid range for this real option is 0 < filter_margin_fact < 1 and
        its default value is 10-05.
    - filter_max_margin (float): advanced feature! Maximum width of margin in
        obj-constr-filter adaptive globalization strategy. The valid range for
        this real option is 0 < filter_max_margin and its default value is 1.
    - adaptive_mu_restore_previous_iterate (str or bool): advanced feature!
        Indicates if the previous accepted iterate should be restored if the
        monotone mode is entered. When the globalization strategy for the
        adaptive barrier algorithm switches to the monotone mode, it can either
        start from the most recent iterate (no), or from the last iterate that
        was accepted (yes). The default value for this string option is "no".
        Possible values: "yes", "no", True, False
    - adaptive_mu_monotone_init_factor (float): advanced feature! Determines the
        initial value of the barrier parameter when switching to the monotone
        mode. When the globalization strategy for the adaptive barrier algorithm
        switches to the monotone mode and fixed_mu_oracle is chosen as
        "average_compl", the barrier parameter is set to the current average
        complementarity times the value of "adaptive_mu_monotone_init_factor".
        The valid range for this real option is 0 <
        adaptive_mu_monotone_init_factor and its default value is 0.8.
    - adaptive_mu_kkt_norm_type (advanced): Norm used for the KKT error in the
        adaptive mu globalization strategies. When computing the KKT error for
        the globalization strategies, the norm to be used is specified with this
        option. Note, this option is also used in the QualityFunctionMuOracle.
        The default value for this string option is "2-norm-squared". Possible
        values:
        - "1-norm": use the 1-norm (abs sum)
        - "2-norm-squared": use the 2-norm squared (sum of squares)
        - "max-norm": use the infinity norm (max)
        - "2-norm": use 2-norm
    - mu_strategy: Update strategy for barrier parameter. Determines which
        barrier parameter update strategy is to be used. The default value for
        this string option is "monotone". Possible values:
        - "monotone": use the monotone (Fiacco-McCormick) strategy
        - "adaptive": use the adaptive update strategy
    - mu_oracle: Oracle for a new barrier parameter in the adaptive strategy.
        Determines how a new barrier parameter is computed in each "free-mode"
        iteration of the adaptive barrier parameter strategy. (Only considered
        if "adaptive" is selected for option "mu_strategy"). The default value
        for this string option is "quality-function". Possible values:
        - "probing": Mehrotra's probing heuristic
        - "loqo": LOQO's centrality rule
        - "quality-function": minimize a quality function
    - fixed_mu_oracle (str): Oracle for the barrier parameter when switching to
        fixed mode. Determines how the first value of the barrier parameter
        should be computed when switching to the "monotone mode" in the adaptive
        strategy. (Only considered if "adaptive" is selected for option
        "mu_strategy".) The default value for this string option is
        "average_compl". Possible values:
        - "probing": Mehrotra's probing heuristic
        - "loqo": LOQO's centrality rule
        - "quality-function": minimize a quality function
        - "average_compl": base on current average complementarity
    - mu_init (float): Initial value for the barrier parameter. This option
        determines the initial value for the barrier parameter (mu). It is only
        relevant in the monotone, Fiacco-McCormick version of the algorithm.
        (i.e., if "mu_strategy" is chosen as "monotone") The valid range for
        this real option is 0 < mu_init and its default value is 0.1.
    - barrier_tol_factor (float): Factor for mu in barrier stop test. The
        convergence tolerance for each barrier problem in the monotone mode is
        the value of the barrier parameter times "barrier_tol_factor". This
        option is also used in the adaptive mu strategy during the monotone
        mode. This is kappa_epsilon in implementation paper. The valid range for
        this real option is 0 < barrier_tol_factor and its default value is 10.
    - mu_linear_decrease_factor (float): Determines linear decrease rate of
        barrier parameter. For the Fiacco-McCormick update procedure the new
        barrier parameter mu is obtained by taking the minimum of
        mu*"mu_linear_decrease_factor" and mu^"superlinear_decrease_power". This
        is kappa_mu in implementation paper. This option is also used in the
        adaptive mu strategy during the monotone mode. The valid range for this
        real option is 0 < mu_linear_decrease_factor < 1 and its default value
        is 0.2.
    - mu_superlinear_decrease_power (float): Determines superlinear decrease
        rate of barrier parameter. For the Fiacco-McCormick update procedure the
        new barrier parameter mu is obtained by taking the minimum of
        mu*"mu_linear_decrease_factor" and mu^"superlinear_decrease_power". This
        is theta_mu in implementation paper. This option is also used in the
        adaptive mu strategy during the monotone mode. The valid range for this
        real option is 1 < mu_superlinear_decrease_power < 2 and its default
        value is 1.5.
    - mu_allow_fast_monotone_decrease (str or bool): Advanced feature! Allow
        skipping of barrier problem if barrier test i already met. The default
        value for this string option is "yes". Possible values:
        - "no": Take at least one iteration per barrier problem even if the
            barrier test is already met for the updated barrier parameter
        - "yes": Allow fast decrease of mu if barrier test it met
    - tau_min (float): Advanced feature! Lower bound on fraction-to-the-boundary
        parameter tau. This is tau_min in the implementation paper. This option
        is also used in the adaptive mu strategy during the monotone mode. The
        valid range for this real option is 0 < tau_min < 1 and its default
        value is 0.99.
    - sigma_max (float): Advanced feature! Maximum value of the centering
        parameter. This is the upper bound for the centering parameter chosen by
        the quality function based barrier parameter update. Only used if option
        "mu_oracle" is set to "quality-function". The valid range for this real
        option is 0 < sigma_max and its default value is 100.
    - sigma_min (float): Advanced feature! Minimum value of the centering
        parameter. This is the lower bound for the centering parameter chosen by
        the quality function based barrier parameter update. Only used if option
        "mu_oracle" is set to "quality-function". The valid range for this real
        option is 0 ≤ sigma_min and its default value is 10-06.
    - quality_function_norm_type (str): Advanced feature. Norm used for
        components of the quality function. Only used if option "mu_oracle" is
        set to "quality-function". The default value for this string option is
        "2-norm-squared". Possible values:
        - "1-norm": use the 1-norm (abs sum)
        - "2-norm-squared": use the 2-norm squared (sum of squares)
        - "max-norm": use the infinity norm (max)
        - "2-norm": use 2-norm
    - quality_function_centrality (str): Advanced feature. The penalty term for
        centrality that is included in qality function. This determines whether
        a term is added to the quality function to penalize deviation from
        centrality with respect to complementarity. The complementarity measure
        here is the xi in the Loqo update rule. Only used if option "mu_oracle"
        is set to "quality-function". The default value for this string option
        is "none". Possible values:
        - "none": no penalty term is added
        - "log": complementarity * the log of the centrality measure
        - "reciprocal": complementarity * the reciprocal of the centrality
            measure
        - "cubed-reciprocal": complementarity * the reciprocal of the centrality
            measure cubed
    - quality_function_balancing_term (str): Advanced feature. The balancing
        term included in the quality function for centrality. This determines
        whether a term is added to the quality function that penalizes
        situations where the complementarity is much smaller than dual and
        primal infeasibilities. Only used if option "mu_oracle" is set to
        "quality-function". The default value for this string option is "none".
        Possible values:
        - "none": no balancing term is added
        - "cubic": Max(0,Max(dual_inf,primal_inf)-compl)^3
    - quality_function_max_section_steps (int): Maximum number of search steps
        during direct search procedure determining the optimal centering
        parameter. The golden section search is performed for the quality
        function based mu oracle. Only used if option "mu_oracle" is set to
        "quality-function". The valid range for this integer option is 0 ≤
        quality_function_max_section_steps and its default value is 8.
    - quality_function_section_sigma_tol (float): advanced feature! Tolerance
        for the section search procedure determining the optimal centering
        parameter (in sigma space). The golden section search is performed for
        the quality function based mu oracle. Only used if option "mu_oracle" is
        set to "quality-function". The valid range for this real option is 0 ≤
        quality_function_section_sigma_tol < 1 and its default value is 0.01.
    - quality_function_section_qf_tol (float): advanced feature! Tolerance for
        the golden section search procedure determining the optimal centering
        parameter (in the function value space). The golden section search is
        performed for the quality function based mu oracle. Only used if option
        "mu_oracle" is set to "quality-function". The valid range for this real
        option is 0 ≤ quality_function_section_qf_tol < 1 and its default value
        is
        0.

    - line_search_method (str): Advanced feature. Globalization method used in
      backtracking line search. Only the "filter" choice is officially
      supported. But sometimes, good results might be obtained with the other
      choices. The default value for this string option is "filter". Possible
      values:
        - "filter": Filter method
        - "cg-penalty": Chen-Goldfarb penalty function
        - "penalty": Standard penalty function
    - alpha_red_factor (float): Advanced feature. Fractional reduction of the
      trial step size in the backtracking lne search. At every step of the
      backtracking line search, the trial step size is reduced by this factor.
      The valid range for this real option is 0 < alpha_red_factor < 1 and its
      default value is 0.5.
    - accept_every_trial_step (str or bool): Always accept the first trial step.
      Setting this option to "yes" essentially disables the line search and
      makes the algorithm take aggressive steps, without global convergence
      guarantees. The default value for this string option is "no". Possible
      values: "yes", "no", True, False.
    - accept_after_max_steps (float): advanced feature. Accept a trial point
      after maximal this number of steps een if it does not satisfy line search
      conditions. Setting this to -1 disables this option. The valid range for
      this integer option is -1 ≤ accept_after_max_steps and its default value
      is -1.
    - alpha_for_y (str): Method to determine the step size for constraint
      multipliers (alpha_y) . The default value for this string option is
      "primal". Possible values:
        - "primal": use primal step size
        - "bound-mult": use step size for the bound multipliers (good for LPs)
        - "min": use the min of primal and bound multipliers
        - "max": use the max of primal and bound multipliers
        - "full": take a full step of size one
        - "min-dual-infeas": choose step size minimizing new dual infeasibility
        - "safer-min-dual-infeas": like "min_dual_infeas", but safeguarded by
          "min" and "max"
        - "primal-and-full": use the primal step size, and full step if delta_x
          <= alpha_for_y_tol
        - "dual-and-full": use the dual step size, and full step if delta_x <=
          alpha_for_y_tol
        - "acceptor": Call LSAcceptor to get step size for y
    - alpha_for_y_tol (float): Tolerance for switching to full equality
        multiplier steps. This is only relevant if "alpha_for_y" is chosen
        "primal-and-full" or "dual-and-full". The step size for the equality
        constraint multipliers is taken to be one if the max-norm of the primal
        step is less than this tolerance. The valid range for this real option
        is 0 ≤ alpha_for_y_tol and its default value is 10.
    - tiny_step_tol (float): Advanced feature. Tolerance for detecting
      numerically insignificant steps. If the search direction in the primal
      variables (x and s) is, in relative terms for each component, less than
      this value, the algorithm accepts the full step without line search. If
      this happens repeatedly, the algorithm will terminate with a corresponding
      exit message. The default value is 10 times machine precision. The valid
      range for this real option is 0 ≤ tiny_step_tol and its default value is
      2.22045 · 10-15.
    - tiny_step_y_tol (float): Advanced feature. Tolerance for quitting because
      of numerically insignificant steps. If the search direction in the primal
      variables (x and s) is, in relative terms for each component, repeatedly
      less than tiny_step_tol, and the step in the y variables is smaller than
      this threshold, the algorithm will terminate. The valid range for this
      real option is 0 ≤ tiny_step_y_tol and its default value is 0.01.
    - watchdog_shortened_iter_trigger (int): Number of shortened iterations that
      trigger the watchdog. If the number of successive iterations in which the
      backtracking line search did not accept the first trial point exceeds this
      number, the watchdog procedure is activated. Choosing "0" here disables
      the watchdog procedure. The valid range for this integer option is 0 ≤
      watchdog_shortened_iter_trigger and its default value is 10.
    - watchdog_trial_iter_max (int): Maximum number of watchdog iterations. This
      option determines the number of trial iterations allowed before the
      watchdog procedure is aborted and the algorithm returns to the stored
      point. The valid range for this integer option is 1 ≤
      watchdog_trial_iter_max and its default value is 3.
    - theta_max_fact (float): Advanced feature. Determines upper bound for
      constraint violation in the filter. The algorithmic parameter theta_max is
      determined as theta_max_fact times the maximum of 1 and the constraint
      violation at initial point. Any point with a constraint violation larger
      than theta_max is unacceptable to the filter (see Eqn. (21) in the
      implementation paper). The valid range for this real option is 0 <
      theta_max_fact and its default value is 10000.
    - theta_min_fact (float): advanced feature. Determines constraint violation
      threshold in the switching rule. The algorithmic parameter theta_min is
      determined as theta_min_fact times the maximum of 1 and the constraint
      violation at initial point. The switching rules treats an iteration as an
      h-type iteration whenever the current constraint violation is larger than
      theta_min (see paragraph before Eqn. (19) in the implementation paper).
      The valid range for this real option is 0 < theta_min_fact and its default
      value is 0.0001.
    - eta_phi (float): advanced! Relaxation factor in the Armijo condition. See
      Eqn. (20) in the implementation paper. The valid range for this real
      option is 0 < eta_phi < 0.5 and its default value is 10-08.
    - delta (float): advanced! Multiplier for constraint violation in the
      switching rule. See Eqn. (19) in the implementation paper. The valid range
      for this real option is 0 < delta and its default value is 1.
    - s_phi (float): advanced! Exponent for linear barrier function model in the
      switching rule. See Eqn. (19) in the implementation paper. The valid range
      for this real option is 1 < s_phi and its default value is 2.3.
    - s_theta (float): advanced! Exponent for current constraint violation in
      the switching rule. See Eqn. (19) in the implementation paper. The valid
      range for this real option is 1 < s_theta and its default value is 1.1.
    - gamma_phi (float): advanced! Relaxation factor in the filter margin for
      the barrier function. See Eqn. (18a) in the implementation paper. The
      valid range for this real option is 0 < gamma_phi < 1 and its default
      value is 10-08.
    - gamma_theta (float): advanced! Relaxation factor in the filter margin for
      the constraint violation. See Eqn. (18b) in the implementation paper. The
      valid range for this real option is 0 < gamma_theta < 1 and its default
      value is 10-05.
    - alpha_min_frac (float): advanced! Safety factor for the minimal step size
      (before switching to restoration phase). This is gamma_alpha in Eqn. (20)
      in the implementation paper. The valid range for this real option is 0 <
      alpha_min_frac < 1 and its default value is 0.05.
    - max_soc (int): Maximum number of second order correction trial steps at
      each iteration. Choosing 0 disables the second order corrections. This is
      p^{max} of Step A-5.9 of Algorithm A in the implementation paper. The
      valid range for this integer option is 0 ≤ max_soc and its default value
      is 4.
    - kappa_soc (float): advanced! Factor in the sufficient reduction rule for
      second order correction. This option determines how much a second order
      correction step must reduce the constraint violation so that further
      correction steps are attempted. See Step A-5.9 of Algorithm A in the
      implementation paper. The valid range for this real option is 0 <
      kappa_soc and its default value is 0.99.
    - obj_max_inc (float): advanced! Determines the upper bound on the
      acceptable increase of barrier objective function. Trial points are
      rejected if they lead to an increase in the barrier objective function by
      more than obj_max_inc orders of magnitude. The valid range for this real
      option is 1 < obj_max_inc and its default value is 5.
    - max_filter_resets (int): advanced! Maximal allowed number of filter
      resets. A positive number enables a heuristic that resets the filter,
      whenever in more than "filter_reset_trigger" successive iterations the
      last rejected trial steps size was rejected because of the filter. This
      option determine the maximal number of resets that are allowed to take
      place. The valid range for this integer option is 0 ≤ max_filter_resets
      and its default value is 5.
    - filter_reset_trigger (int): Advanced! Number of iterations that trigger
      the filter reset. If the filter reset heuristic is active and the number
      of successive iterations in which the last rejected trial step size was
      rejected because of the filter, the filter is reset. The valid range for
      this integer option is 1 ≤ filter_reset_trigger and its default value is
      5.
    - corrector_type (str): advanced! The type of corrector steps that should be
      taken. If "mu_strategy" is "adaptive", this option determines what kind of
      corrector steps should be tried. Changing this option is experimental. The
      default value for this string option is "none". Possible values:
        - "none": no corrector
        - "affine": corrector step towards mu=0
        - "primal-dual": corrector step towards current mu
    - skip_corr_if_neg_curv (str or bool): advanced! Whether to skip the
      corrector step in negative curvature ieration. The corrector step is not
      tried if negative curvature has been encountered during the computation of
      the search direction in the current iteration. This option is only used if
      "mu_strategy" is "adaptive". Changing this option is experimental. The
      default value for this string option is "yes". Possible values: "yes",
      "no", True, False
    - skip_corr_in_monotone_mode (str or bool): Advanced! Whether to skip the
      corrector step during monotone brrier parameter mode. The corrector step
      is not tried if the algorithm is currently in the monotone mode (see also
      option "barrier_strategy"). This option is only used if "mu_strategy" is
      "adaptive". Changing this option is experimental. The default value for
      this string option is "yes". Possible values: "yes", "no", True, False
    - corrector_compl_avrg_red_fact (int): advanced! Complementarity tolerance
      factor for accepting corrector step. This option determines the factor by
      which complementarity is allowed to increase for a corrector step to be
      accepted. Changing this option is experimental. The valid range for this
      real option is 0 < corrector_compl_avrg_red_fact and its default value is
      1.
    - soc_method (int): Ways to apply second order correction. This option
      determines the way to apply second order correction, 0 is the method
      described in the implementation paper. 1 is the modified way which adds
      alpha on the rhs of x and s rows. Officially, the valid range for this
      integer option is 0 ≤ soc_method ≤ 1 and its default value is 0 but only 0
      and 1 are allowed.

    - nu_init (float): advanced! Initial value of the penalty parameter. The
      valid range for this real option is 0 < nu_init and its default value is
      10-06.
    - nu_inc (float): advanced! Increment of the penalty parameter. The valid
      range for this real option is 0 < nu_inc and its default value is 0.0001.
    - rho (float): advanced! Value in penalty parameter update formula. The
      valid range for this real option is 0 < rho < 1 and its default value is
      0.1.
    - kappa_sigma (float): advanced! Factor limiting the deviation of dual
      variables from primal estimates. If the dual variables deviate from their
      primal estimates, a correction is performed. See Eqn. (16) in the
      implementation paper. Setting the value to less than 1 disables the
      correction. The valid range for this real option is 0 < kappa_sigma and
      its default value is 10+10.
    - recalc_y (str or bool): Tells the algorithm to recalculate the equality
      and inequality multipliers as least square estimates. This asks the
      algorithm to recompute the multipliers, whenever the current infeasibility
      is less than recalc_y_feas_tol. Choosing yes might be helpful in the
      quasi-Newton option. However, each recalculation requires an extra
      factorization of the linear system. If a limited memory quasi-Newton
      option is chosen, this is used by default. The default value for this
      string option is "no". Possible values:
        - "no" or False: use the Newton step to update the multipliers
        - "yes" or True: use least-square multiplier estimates
    - recalc_y_feas_tol (float): Feasibility threshold for recomputation of
      multipliers. If recalc_y is chosen and the current infeasibility is less
      than this value, then the multipliers are recomputed. The valid range for
      this real option is 0 < recalc_y_feas_tol and its default value is 10-06.
    - slack_move (float): advanced! Correction size for very small slacks. Due
      to numerical issues or the lack of an interior, the slack variables might
      become very small. If a slack becomes very small compared to machine
      precision, the corresponding bound is moved slightly. This parameter
      determines how large the move should be. Its default value is
      mach_eps^{3/4}. See also end of Section 3.5 in implementation paper - but
      actual implementation might be somewhat different. The valid range for
      this real option is 0 ≤ slack_move and its default value is 1.81899 ·
      10-12.
    - constraint_violation_norm_type (str): advanced! Norm to be used for the
      constraint violation in te line search. Determines which norm should be
      used when the algorithm computes the constraint violation in the line
      search. The default value for this string option is "1-norm". Possible
      values:
        - "1-norm": use the 1-norm
        - "2-norm": use the 2-norm
        - "max-norm": use the infinity norm

    - mehrotra_algorithm (str or bool): Indicates whether to do Mehrotra's
      predictor-corrector algorithm. If enabled, line search is disabled and the
      (unglobalized) adaptive mu strategy is chosen with the "probing" oracle,
      and "corrector_type=affine" is used without any safeguards; you should not
      set any of those options explicitly in addition. Also, unless otherwise
      specified, the values of "bound_push", "bound_frac", and
      "bound_mult_init_val" are set more aggressive, and sets
      "alpha_for_y=bound_mult". The Mehrotra's predictor-corrector algorithm
      works usually very well for LPs and convex QPs. The default value for this
      string option is "no". Possible values: "yes", "no", True, False.
    - fast_step_computation (str or bool): Indicates if the linear system should
      be solved quickly. If enabled, the algorithm assumes that the linear
      system that is solved to obtain the search direction is solved
      sufficiently well. In that case, no residuals are computed to verify the
      solution and the computation of the search direction is a little faster.
      The default value for this string option is "no". Possible values: "yes",
      "no", True, False.
    - min_refinement_steps (int): Minimum number of iterative refinement steps
      per linear system solve. Iterative refinement (on the full unsymmetric
      system) is performed for each right hand side. This option determines the
      minimum number of iterative refinements (i.e. at least
      "min_refinement_steps" iterative refinement steps are enforced per right
      hand side.) The valid range for this integer option is 0 ≤
      min_refinement_steps and its default value is 1.
    - max_refinement_steps (int): Maximum number of iterative refinement steps
      per linear system solve. Iterative refinement (on the full unsymmetric
      system) is performed for each right hand side. This option determines the
      maximum number of iterative refinement steps. The valid range for this
      integer option is 0 ≤ max_refinement_steps and its default value is 10.
    - residual_ratio_max (float): advanced! Iterative refinement tolerance.
      Iterative refinement is performed until the residual test ratio is less
      than this tolerance (or until "max_refinement_steps" refinement steps are
      performed). The valid range for this real option is 0 < residual_ratio_max
      and its default value is 10-10.
    - residual_ratio_singular (float): advanced! Threshold for declaring linear
      system singular after filed iterative refinement. If the residual test
      ratio is larger than this value after failed iterative refinement, the
      algorithm pretends that the linear system is singular. The valid range for
      this real option is 0 < residual_ratio_singular and its default value is
      10-05.
    - residual_improvement_factor (float): advanced! Minimal required reduction
      of residual test ratio in iterative refinement. If the improvement of the
      residual test ratio made by one iterative refinement step is not better
      than this factor, iterative refinement is aborted. The valid range for
      this real option is 0 < residual_improvement_factor and its default value
      is 1.

    - neg_curv_test_tol (float): Tolerance for heuristic to ignore wrong
      inertia. If nonzero, incorrect inertia in the augmented system is ignored,
      and Ipopt tests if the direction is a direction of positive curvature.
      This tolerance is alpha_n in the paper by Zavala and Chiang (2014) and it
      determines when the direction is considered to be sufficiently positive. A
      value in the range of [1e-12, 1e-11] is recommended. The valid range for
      this real option is 0 ≤ neg_curv_test_tol and its default value is 0.
    - neg_curv_test_reg (str or bool): Whether to do the curvature test with the
      primal regularization (see Zvala and Chiang, 2014). The default value for
      this string option is "yes". Possible values:
        - "yes" or True: use primal regularization with the inertia-free
          curvature test
        - "no" or False: use original IPOPT approach, in which the primal
          regularization is ignored
    - max_hessian_perturbation (float): Maximum value of regularization
      parameter for handling negative curvature. In order to guarantee that the
      search directions are indeed proper descent directions, Ipopt requires
      that the inertia of the (augmented) linear system for the step computation
      has the correct number of negative and positive eigenvalues. The idea is
      that this guides the algorithm away from maximizers and makes Ipopt more
      likely converge to first order optimal points that are minimizers. If the
      inertia is not correct, a multiple of the identity matrix is added to the
      Hessian of the Lagrangian in the augmented system. This parameter gives
      the maximum value of the regularization parameter. If a regularization of
      that size is not enough, the algorithm skips this iteration and goes to
      the restoration phase. This is delta_w^max in the implementation paper.
      The valid range for this real option is 0 < max_hessian_perturbation and
      its default value is 10+20.
    - min_hessian_perturbation (float): Smallest perturbation of the Hessian
      block. The size of the perturbation of the Hessian block is never selected
      smaller than this value, unless no perturbation is necessary. This is
      delta_w^min in implementation paper. The valid range for this real option
      is 0 ≤ min_hessian_perturbation and its default value is 10-20.
    - perturb_inc_fact_first (float): Increase factor for x-s perturbation for
      very first perturbation. The factor by which the perturbation is increased
      when a trial value was not sufficient - this value is used for the
      computation of the very first perturbation and allows a different value
      for the first perturbation than that used for the remaining perturbations.
      This is bar_kappa_w^+ in the implementation paper. The valid range for
      this real option is 1 < perturb_inc_fact_first and its default value is
      100.
    - perturb_inc_fact (float): Increase factor for x-s perturbation. The factor
      by which the perturbation is increased when a trial value was not
      sufficient
      - this value is used for the computation of all perturbations except for
        the first. This is kappa_w^+ in the implementation paper. The valid
        range for this real option is 1 < perturb_inc_fact and its default value
        is 8.
    - perturb_dec_fact (float): Decrease factor for x-s perturbation. The factor
      by which the perturbation is decreased when a trial value is deduced from
      the size of the most recent successful perturbation. This is kappa_w^- in
      the implementation paper. The valid range for this real option is 0 <
      perturb_dec_fact < 1 and its default value is 0.333333.
    - first_hessian_perturbation (float): Size of first x-s perturbation tried.
      The first value tried for the x-s perturbation in the inertia correction
      scheme. This is delta_0 in the implementation paper. The valid range for
      this real option is 0 < first_hessian_perturbation and its default value
      is 0.0001.
    - jacobian_regularization_value (float): Size of the regularization for
      rank-deficient constraint Jacobians. This is bar delta_c in the
      implementation paper. The valid range for this real option is 0 ≤
      jacobian_regularization_value and its default value is 10-08.
    - jacobian_regularization_exponent (float): advanced! Exponent for mu in the
      regularization for rnk-deficient constraint Jacobians. This is kappa_c in
      the implementation paper. The valid range for this real option is 0 ≤
      jacobian_regularization_exponent and its default value is 0.25.
    - perturb_always_cd (str or bool): advanced! Active permanent perturbation
      of constraint linearization. Enabling this option leads to using the
      delta_c and delta_d perturbation for the computation of every search
      direction. Usually, it is only used when the iteration matrix is singular.
      The default value for this string option is "no". Possible values: "yes",
      "no", True, False.

    - expect_infeasible_problem (str or bool): Enable heuristics to quickly
      detect an infeasible problem. This options is meant to activate heuristics
      that may speed up the infeasibility determination if you expect that there
      is a good chance for the problem to be infeasible. In the filter line
      search procedure, the restoration phase is called more quickly than
      usually, and more reduction in the constraint violation is enforced before
      the restoration phase is left. If the problem is square, this option is
      enabled automatically. The default value for this string option is "no".
      Possible values: "yes", "no", True, False.
    - expect_infeasible_problem_ctol (float): Threshold for disabling
      "expect_infeasible_problem" option. If the constraint violation becomes
      smaller than this threshold, the "expect_infeasible_problem" heuristics in
      the filter line search are disabled. If the problem is square, this
      options is set to 0. The valid range for this real option is 0 ≤
      expect_infeasible_problem_ctol and its default value is 0.001.
    - expect_infeasible_problem_ytol (float): Multiplier threshold for
      activating "xpect_infeasible_problem" option. If the max norm of the
      constraint multipliers becomes larger than this value and
      "expect_infeasible_problem" is chosen, then the restoration phase is
      entered. The valid range for this real option is 0 <
      expect_infeasible_problem_ytol and its default value is 10+08.
    - start_with_resto (str or bool): Whether to switch to restoration phase in
      first iteration.Setting this option to "yes" forces the algorithm to
      switch to the feasibility restoration phase in the first iteration. If the
      initial point is feasible, the algorithm will abort with a failure. The
      default value for this string option is "no". Possible values: "yes",
      "no", True, False
    - soft_resto_pderror_reduction_factor (float): Required reduction in
      primal-dual error in the soft restoration phase. The soft restoration
      phase attempts to reduce the primal-dual error with regular steps. If the
      damped primal-dual step (damped only to satisfy the
      fraction-to-the-boundary rule) is not decreasing the primal-dual error by
      at least this factor, then the regular restoration phase is called.
      Choosing "0" here disables the soft restoration phase. The valid range for
      this real option is 0 ≤ soft_resto_pderror_reduction_factor and its
      default value is 0.9999.
    - max_soft_resto_iters (int): advanced! Maximum number of iterations
      performed successively in soft rstoration phase. If the soft restoration
      phase is performed for more than so many iterations in a row, the regular
      restoration phase is called. The valid range for this integer option is 0
      ≤ max_soft_resto_iters and its default value is 10.
    - required_infeasibility_reduction (float): Required reduction of
      infeasibility before leaving restoration phase. The restoration phase
      algorithm is performed, until a point is found that is acceptable to the
      filter and the infeasibility has been reduced by at least the fraction
      given by this option. The valid range for this real option is 0 ≤
      required_infeasibility_reduction < 1 and its default value is 0.9.
    - max_resto_iter (int): advanced! Maximum number of successive iterations in
      restoration phase.The algorithm terminates with an error message if the
      number of iterations successively taken in the restoration phase exceeds
      this number. The valid range for this integer option is 0 ≤ max_resto_iter
      and its default value is 3000000.
    - evaluate_orig_obj_at_resto_trial (str or bool): Determines if the original
      objective function should be evaluated at restoration phase trial points.
      Enabling this option makes the restoration phase algorithm evaluate the
      objective function of the original problem at every trial point
      encountered during the restoration phase, even if this value is not
      required. In this way, it is guaranteed that the original objective
      function can be evaluated without error at all accepted iterates;
      otherwise the algorithm might fail at a point where the restoration phase
      accepts an iterate that is good for the restoration phase problem, but not
      the original problem. On the other hand, if the evaluation of the original
      objective is expensive, this might be costly. The default value for this
      string option is "yes". Possible values: "yes", "no", True, False
    - resto_penalty_parameter (float): advanced! Penalty parameter in the
      restoration phase objective fnction. This is the parameter rho in equation
      (31a) in the Ipopt implementation paper. The valid range for this real
      option is 0 < resto_penalty_parameter and its default value is 1000.
    - resto_proximity_weight (float): advanced! Weighting factor for the
      proximity term in restoration pase objective. This determines how the
      parameter zeta in equation (29a) in the implementation paper is computed.
      zeta here is resto_proximity_weight*sqrt(mu), where mu is the current
      barrier parameter. The valid range for this real option is 0 ≤
      resto_proximity_weight and its default value is 1.
    - bound_mult_reset_threshold (float): Threshold for resetting bound
      multipliers after the restoration pase. After returning from the
      restoration phase, the bound multipliers are updated with a Newton step
      for complementarity. Here, the change in the primal variables during the
      entire restoration phase is taken to be the corresponding primal Newton
      step. However, if after the update the largest bound multiplier exceeds
      the threshold specified by this option, the multipliers are all reset to
      1. The valid range for this real option is 0 ≤ bound_mult_reset_threshold
         and its default value is 1000.
    - constr_mult_reset_threshold (float): Threshold for resetting equality and
      inequality multipliers ater restoration phase. After returning from the
      restoration phase, the constraint multipliers are recomputed by a least
      square estimate. This option triggers when those least-square estimates
      should be ignored. The valid range for this real option is 0 ≤
      constr_mult_reset_threshold and its default value is 0.
    - resto_failure_feasibility_threshold (float): advanced! Threshold for
      primal infeasibility to declare filure of restoration phase. If the
      restoration phase is terminated because of the "acceptable" termination
      criteria and the primal infeasibility is smaller than this value, the
      restoration phase is declared to have failed. The default value is
      actually 1e2*tol, where tol is the general termination tolerance. The
      valid range for this real option is 0 ≤
      resto_failure_feasibility_threshold and its default value is
      0.

    - limited_memory_aug_solver (str): advanced! Strategy for solving the
      augmented system for low-rank Hessian. The default value for this string
      option is "sherman-morrison". Possible values:
        - "sherman-morrison": use Sherman-Morrison formula
        - "extended": use an extended augmented system
    - limited_memory_max_history (int): Maximum size of the history for the
      limited quasi-Newton Hessian approximation. This option determines the
      number of most recent iterations that are taken into account for the
      limited-memory quasi-Newton approximation. The valid range for this
      integer option is 0 ≤ limited_memory_max_history and its default value is
      6.
    - limited_memory_update_type (str): Quasi-Newton update formula for the
      limited memory quasi-Newton approximation. The default value for this
      string option is "bfgs". Possible values:
        - "bfgs": BFGS update (with skipping)
        - "sr1": SR1 (not working well)
    - limited_memory_initialization (str): Initialization strategy for the
      limited memory quasi-Newton aproximation. Determines how the diagonal
      Matrix B_0 as the first term in the limited memory approximation should be
      computed. The default value for this string option is "scalar1". Possible
      values:
        - "scalar1": sigma = s^Ty/s^Ts
        - "scalar2": sigma = y^Ty/s^Ty
        - "scalar3": arithmetic average of scalar1 and scalar2
        - "scalar4": geometric average of scalar1 and scalar2
        - "constant": sigma = limited_memory_init_val
    - limited_memory_init_val (float): Value for B0 in low-rank update. The
      starting matrix in the low rank update, B0, is chosen to be this multiple
      of the identity in the first iteration (when no updates have been
      performed yet), and is constantly chosen as this value, if
      "limited_memory_initialization" is "constant". The valid range for this
      real option is 0 < limited_memory_init_val and its default value is 1.
    - limited_memory_init_val_max (float): Upper bound on value for B0 in
      low-rank update. The starting matrix in the low rank update, B0, is chosen
      to be this multiple of the identity in the first iteration (when no
      updates have been performed yet), and is constantly chosen as this value,
      if "limited_memory_initialization" is "constant". The valid range for this
      real option is 0 < limited_memory_init_val_max and its default value is
      10+08.
    - limited_memory_init_val_min (float): Lower bound on value for B0 in
      low-rank update. The starting matrix in the low rank update, B0, is chosen
      to be this multiple of the identity in the first iteration (when no
      updates have been performed yet), and is constantly chosen as this value,
      if "limited_memory_initialization" is "constant". The valid range for this
      real option is 0 < limited_memory_init_val_min and its default value is
      10-08.
    - limited_memory_max_skipping (int): Threshold for successive iterations
      where update is skipped. If the update is skipped more than this number of
      successive iterations, the quasi-Newton approximation is reset. The valid
      range for this integer option is 1 ≤ limited_memory_max_skipping and its
      default value is 2.
    - limited_memory_special_for_resto (str or bool): Determines if the
      quasi-Newton updates should be special dring the restoration phase. Until
      Nov 2010, Ipopt used a special update during the restoration phase, but it
      turned out that this does not work well. The new default uses the regular
      update procedure and it improves results. If for some reason you want to
      get back to the original update, set this option to "yes". The default
      value for this string option is "no". Possible values: "yes", "no", True,
      False.
    - hessian_approximation_space (str): advanced! Indicates in which subspace
      the Hessian information is to be approximated. The default value for this
      string option is "nonlinear-variables". Possible values:
        - "nonlinear-variables": only in space of nonlinear variables.
        - "all-variables": in space of all variables (without slacks)
    - linear_solver (str): Linear solver used for step computations. Determines
      which linear algebra package is to be used for the solution of the
      augmented linear system (for obtaining the search directions). The default
      value for this string option is "ma27". Possible values:
        - mumps (use the Mumps package, default)
        - ma27 (load the Harwell routine MA27 from library at runtime)
        - ma57 (load the Harwell routine MA57 from library at runtime)
        - ma77 (load the Harwell routine HSL_MA77 from library at runtime)
        - ma86 (load the Harwell routine MA86 from library at runtime)
        - ma97 (load the Harwell routine MA97 from library at runtime)
        - pardiso (load the Pardiso package from pardiso-project.org from
          user-provided library at runtime)
        - custom (use custom linear solver (expert use))
    - linear_solver_options (dict or None): dictionary with the linear solver
      options, possibly including `linear_system_scaling`, `hsllib` and
      `pardisolib`. See the `ipopt documentation for details
      <https://coin-or.github.io/Ipopt/OPTIONS.html>`_. The linear solver
      options are not automatically converted to float at the moment.

    The following options are not supported:
      - `num_linear_variables`: since estimagic may reparametrize your problem
        and this changes the parameter problem, we do not support this option.
      - scaling options (`nlp_scaling_method`, `obj_scaling_factor`,
        `nlp_scaling_max_gradient`, `nlp_scaling_obj_target_gradient`,
        `nlp_scaling_constr_target_gradient`, `nlp_scaling_min_value`)
      - `hessian_approximation`
      - derivative checks
      - print options. Use estimagic's dashboard to monitor your optimization.

    """
    if not IS_CYIPOPT_INSTALLED:
        raise NotImplementedError(
            "The cyipopt package is not installed and required for 'ipopt'. You can "
            "install the package with: `conda install -c conda-forge cyipopt`"
        )
    if acceptable_tol <= convergence_relative_criterion_tolerance:
        raise ValueError(
            "The acceptable tolerance must be larger than the desired tolerance."
        )
    if mu_strategy not in ["monotone", "adaptive"]:
        raise ValueError(
            f"Unknown barrier strategy: {mu_strategy}. It must be 'monotone' or "
            "'adaptive'."
        )
    if nlp_upper_bound_inf < 0:
        raise ValueError("nlp_upper_bound_inf should be > 0.")
    if nlp_lower_bound_inf > 0:
        raise ValueError("nlp_lower_bound_inf should be < 0.")

    linear_solver_options = (
        {} if linear_solver_options is None else linear_solver_options
    )

    # The default value is actually 1e2*tol, where tol is the general
    # termination tolerance.
    if resto_failure_feasibility_threshold is None:
        resto_failure_feasibility_threshold = (
            1e2 * convergence_relative_criterion_tolerance
        )

    # convert None to str none section
    linear_solver_options_with_none = [
        "ma86_scaling",
        "ma97_scaling",
        "ma97_scaling1",
        "ma97_scaling2",
        "ma97_scaling3",
        "spral_scaling",
        "spral_scaling_1",
        "spral_scaling_2",
        "spral_scaling_3",
        "linear_system_scaling",
    ]
    for key, val in linear_solver_options.items():
        if key in linear_solver_options_with_none:
            linear_solver_options[key] = _convert_none_to_str(val)

    converted_bool_to_str_options = {
        "dependency_detection_with_rhs": _convert_bool_to_str(
            dependency_detection_with_rhs, "dependency_detection_with_rhs"
        ),
        "check_derivatives_for_naninf": _convert_bool_to_str(
            check_derivatives_for_naninf, "check_derivatives_for_naninf"
        ),
        "jac_c_constant": _convert_bool_to_str(jac_c_constant, "jac_c_constant"),
        "jac_d_constant": _convert_bool_to_str(jac_d_constant, "jac_d_constant"),
        "hessian_constant": _convert_bool_to_str(hessian_constant, "hessian_constant"),
        "least_square_init_primal": _convert_bool_to_str(
            least_square_init_primal, "least_square_init_primal"
        ),
        "least_square_init_duals": _convert_bool_to_str(
            least_square_init_duals, "least_square_init_duals"
        ),
        "warm_start_init_point": _convert_bool_to_str(
            warm_start_init_point, "warm_start_init_point"
        ),
        "warm_start_same_structure": _convert_bool_to_str(
            warm_start_same_structure, "warm_start_same_structure"
        ),
        "warm_start_entire_iterate": _convert_bool_to_str(
            warm_start_entire_iterate, "warm_start_entire_iterate"
        ),
        "replace_bounds": _convert_bool_to_str(replace_bounds, "replace_bounds"),
        "skip_finalize_solution_call": _convert_bool_to_str(
            skip_finalize_solution_call, "skip_finalize_solution_call"
        ),
        "timing_statistics": _convert_bool_to_str(
            timing_statistics, "timing_statistics"
        ),
        "adaptive_mu_restore_previous_iterate": _convert_bool_to_str(
            adaptive_mu_restore_previous_iterate, "adaptive_mu_restore_previous_iterate"
        ),
        "mu_allow_fast_monotone_decrease": _convert_bool_to_str(
            mu_allow_fast_monotone_decrease, "mu_allow_fast_monotone_decrease"
        ),
        "accept_every_trial_step": _convert_bool_to_str(
            accept_every_trial_step, "accept_every_trial_step"
        ),
        "skip_corr_if_neg_curv": _convert_bool_to_str(
            skip_corr_if_neg_curv, "skip_corr_if_neg_curv"
        ),
        "skip_corr_in_monotone_mode": _convert_bool_to_str(
            skip_corr_in_monotone_mode, "skip_corr_in_monotone_mode"
        ),
        "recalc_y": _convert_bool_to_str(recalc_y, "recalc_y"),
        "mehrotra_algorithm": _convert_bool_to_str(
            mehrotra_algorithm, "mehrotra_algorithm"
        ),
        "fast_step_computation": _convert_bool_to_str(
            fast_step_computation, "fast_step_computation"
        ),
        "neg_curv_test_reg": _convert_bool_to_str(
            neg_curv_test_reg, "neg_curv_test_reg"
        ),
        "perturb_always_cd": _convert_bool_to_str(
            perturb_always_cd, "perturb_always_cd"
        ),
        "expect_infeasible_problem": _convert_bool_to_str(
            expect_infeasible_problem, "expect_infeasible_problem"
        ),
        "start_with_resto": _convert_bool_to_str(start_with_resto, "start_with_resto"),
        "evaluate_orig_obj_at_resto_trial": _convert_bool_to_str(
            evaluate_orig_obj_at_resto_trial, "evaluate_orig_obj_at_resto_trial"
        ),
        "limited_memory_special_for_resto": _convert_bool_to_str(
            limited_memory_special_for_resto, "limited_memory_special_for_resto"
        ),
        "honor_original_bounds": _convert_bool_to_str(
            honor_original_bounds, "honor_original_bounds"
        ),
    }
    algo_info = DEFAULT_ALGO_INFO.copy()
    algo_info["name"] = "ipopt"

    gradient = functools.partial(
        criterion_and_derivative, task="derivative", algorithm_info=algo_info
    )

    func = functools.partial(
        criterion_and_derivative,
        task="criterion",
        algorithm_info=algo_info,
    )

    options = {
        # disable verbosity
        "print_level": 0,
        "ma77_print_level": -1,
        "ma86_print_level": -1,
        "ma97_print_level": -1,
        "pardiso_msglvl": 0,
        # disable scaling
        "nlp_scaling_method": "none",
        # disable derivative checker
        "derivative_test": "none",
        #
        "s_max": float(s_max),
        "max_iter": stopping_max_iterations,
        "max_wall_time": float(stopping_max_wall_time_seconds),
        "max_cpu_time": stopping_max_cpu_time,
        "dual_inf_tol": dual_inf_tol,
        "constr_viol_tol": constr_viol_tol,
        "compl_inf_tol": compl_inf_tol,
        # acceptable heuristic
        "acceptable_iter": int(acceptable_iter),
        "acceptable_tol": acceptable_tol,
        "acceptable_dual_inf_tol": acceptable_dual_inf_tol,
        "acceptable_constr_viol_tol": acceptable_constr_viol_tol,
        "acceptable_compl_inf_tol": acceptable_compl_inf_tol,
        "acceptable_obj_change_tol": acceptable_obj_change_tol,
        # bounds and more
        "diverging_iterates_tol": diverging_iterates_tol,
        "nlp_lower_bound_inf": nlp_lower_bound_inf,
        "nlp_upper_bound_inf": nlp_upper_bound_inf,
        "fixed_variable_treatment": fixed_variable_treatment,
        "dependency_detector": _convert_none_to_str(dependency_detector),
        "kappa_d": kappa_d,
        "bound_relax_factor": bound_relax_factor,
        "honor_original_bounds": honor_original_bounds,
        # initialization
        "bound_push": bound_push,
        "bound_frac": bound_frac,
        "slack_bound_push": slack_bound_push,
        "slack_bound_frac": slack_bound_frac,
        "constr_mult_init_max": float(constr_mult_init_max),
        "bound_mult_init_val": float(bound_mult_init_val),
        "bound_mult_init_method": bound_mult_init_method,
        # warm start
        "warm_start_bound_push": warm_start_bound_push,
        "warm_start_bound_frac": warm_start_bound_frac,
        "warm_start_slack_bound_push": warm_start_slack_bound_push,
        "warm_start_slack_bound_frac": warm_start_slack_bound_frac,
        "warm_start_mult_bound_push": warm_start_mult_bound_push,
        "warm_start_mult_init_max": warm_start_mult_init_max,
        "warm_start_target_mu": warm_start_target_mu,
        # more miscellaneous
        "option_file_name": option_file_name,
        # barrier parameter update
        "mu_target": float(mu_target),
        "mu_max_fact": float(mu_max_fact),
        "mu_max": float(mu_max),
        "mu_min": float(mu_min),
        "adaptive_mu_globalization": adaptive_mu_globalization,
        "adaptive_mu_kkterror_red_iters": adaptive_mu_kkterror_red_iters,
        "adaptive_mu_kkterror_red_fact": adaptive_mu_kkterror_red_fact,
        "filter_margin_fact": float(filter_margin_fact),
        "filter_max_margin": float(filter_max_margin),
        "adaptive_mu_monotone_init_factor": adaptive_mu_monotone_init_factor,
        "adaptive_mu_kkt_norm_type": adaptive_mu_kkt_norm_type,
        "mu_strategy": mu_strategy,
        "mu_oracle": mu_oracle,
        "fixed_mu_oracle": fixed_mu_oracle,
        "mu_init": mu_init,
        "barrier_tol_factor": float(barrier_tol_factor),
        "mu_linear_decrease_factor": mu_linear_decrease_factor,
        "mu_superlinear_decrease_power": mu_superlinear_decrease_power,
        "tau_min": tau_min,
        "sigma_max": float(sigma_max),
        "sigma_min": sigma_min,
        "quality_function_norm_type": quality_function_norm_type,
        "quality_function_centrality": _convert_none_to_str(
            quality_function_centrality
        ),
        "quality_function_balancing_term": _convert_none_to_str(
            quality_function_balancing_term
        ),
        "quality_function_max_section_steps": int(quality_function_max_section_steps),
        "quality_function_section_sigma_tol": quality_function_section_sigma_tol,
        "quality_function_section_qf_tol": quality_function_section_qf_tol,
        # linear search
        "line_search_method": line_search_method,
        "alpha_red_factor": alpha_red_factor,
        "accept_after_max_steps": accept_after_max_steps,
        "alpha_for_y": alpha_for_y,
        "alpha_for_y_tol": float(alpha_for_y_tol),
        "tiny_step_tol": tiny_step_tol,
        "tiny_step_y_tol": tiny_step_y_tol,
        "watchdog_shortened_iter_trigger": watchdog_shortened_iter_trigger,
        "watchdog_trial_iter_max": watchdog_trial_iter_max,
        "theta_max_fact": float(theta_max_fact),
        "theta_min_fact": theta_min_fact,
        "eta_phi": eta_phi,
        "delta": float(delta),
        "s_phi": s_phi,
        "s_theta": s_theta,
        "gamma_phi": gamma_phi,
        "gamma_theta": gamma_theta,
        "alpha_min_frac": alpha_min_frac,
        "max_soc": max_soc,
        "kappa_soc": kappa_soc,
        "obj_max_inc": float(obj_max_inc),
        "max_filter_resets": max_filter_resets,
        "filter_reset_trigger": filter_reset_trigger,
        "corrector_type": _convert_none_to_str(corrector_type),
        "corrector_compl_avrg_red_fact": float(corrector_compl_avrg_red_fact),
        "soc_method": soc_method,
        "nu_init": nu_init,
        "nu_inc": nu_inc,
        "rho": rho,
        "kappa_sigma": kappa_sigma,
        "recalc_y_feas_tol": recalc_y_feas_tol,
        "slack_move": slack_move,
        "constraint_violation_norm_type": constraint_violation_norm_type,
        # step calculation
        "min_refinement_steps": min_refinement_steps,
        "max_refinement_steps": max_refinement_steps,
        "residual_ratio_max": residual_ratio_max,
        "residual_ratio_singular": residual_ratio_singular,
        "residual_improvement_factor": float(residual_improvement_factor),
        "neg_curv_test_tol": float(neg_curv_test_tol),
        "max_hessian_perturbation": max_hessian_perturbation,
        "min_hessian_perturbation": min_hessian_perturbation,
        "perturb_inc_fact_first": float(perturb_inc_fact_first),
        "perturb_inc_fact": float(perturb_inc_fact),
        "perturb_dec_fact": float(perturb_dec_fact),
        "first_hessian_perturbation": float(first_hessian_perturbation),
        "jacobian_regularization_value": float(jacobian_regularization_value),
        "jacobian_regularization_exponent": float(jacobian_regularization_exponent),
        # restoration phase
        "expect_infeasible_problem_ctol": expect_infeasible_problem_ctol,
        "expect_infeasible_problem_ytol": expect_infeasible_problem_ytol,
        "soft_resto_pderror_reduction_factor": soft_resto_pderror_reduction_factor,
        "max_soft_resto_iters": max_soft_resto_iters,
        "required_infeasibility_reduction": float(required_infeasibility_reduction),
        "max_resto_iter": max_resto_iter,
        "resto_penalty_parameter": float(resto_penalty_parameter),
        "resto_proximity_weight": float(resto_proximity_weight),
        "bound_mult_reset_threshold": float(bound_mult_reset_threshold),
        "constr_mult_reset_threshold": float(constr_mult_reset_threshold),
        "resto_failure_feasibility_threshold": float(
            resto_failure_feasibility_threshold
        ),
        # hessian approximation
        "limited_memory_aug_solver": limited_memory_aug_solver,
        "limited_memory_max_history": limited_memory_max_history,
        "limited_memory_update_type": limited_memory_update_type,
        "limited_memory_initialization": limited_memory_initialization,
        "limited_memory_init_val": float(limited_memory_init_val),
        "limited_memory_init_val_max": limited_memory_init_val_max,
        "limited_memory_init_val_min": limited_memory_init_val_min,
        "limited_memory_max_skipping": limited_memory_max_skipping,
        "hessian_approximation_space": hessian_approximation_space,
        # linear solver
        "linear_solver": linear_solver,
        **linear_solver_options,
        #
        **converted_bool_to_str_options,
    }

    raw_res = cyipopt.minimize_ipopt(
        fun=func,
        x0=x,
        bounds=get_scipy_bounds(lower_bounds, upper_bounds),
        jac=gradient,
        constraints=(),
        tol=convergence_relative_criterion_tolerance,
        options=options,
    )

    res = process_scipy_result(raw_res)
    return res


def _convert_bool_to_str(var, name):
    """Convert input to either 'yes' or 'no' and check the output is yes or no.

    Args:
        var (str or bool): user input
        name (str): name of the variable.

    Returns:
        out (str): "yes" or "no".

    """
    if var is True:
        out = "yes"
    elif var is False:
        out = "no"
    else:
        out = var
    if out not in {"yes", "no"}:
        raise ValueError(
            f"{name} must be 'yes', 'no', True or False. You specified {var}."
        )
    return out


def _convert_none_to_str(var):
    out = "none" if var is None else var
    return out
