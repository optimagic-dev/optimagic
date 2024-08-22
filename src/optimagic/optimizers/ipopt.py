"""Implement cyipopt's Interior Point Optimizer."""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds as ScipyBounds

from optimagic import mark
from optimagic.config import IS_CYIPOPT_INSTALLED
from optimagic.exceptions import NotInstalledError
from optimagic.optimization.algo_options import (
    CONVERGENCE_FTOL_REL,
    STOPPING_MAXITER,
)
from optimagic.optimization.algorithm import Algorithm, InternalOptimizeResult
from optimagic.optimization.internal_optimization_problem import (
    InternalBounds,
    InternalOptimizationProblem,
)
from optimagic.optimizers.scipy_optimizers import process_scipy_result
from optimagic.typing import (
    AggregationLevel,
    GtOneFloat,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    YesNoBool,
)

if IS_CYIPOPT_INSTALLED:
    import cyipopt


@mark.minimizer(
    name="ipopt",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_CYIPOPT_INSTALLED,
    is_global=False,
    needs_jac=True,
    needs_hess=False,
    supports_parallelism=False,
    supports_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=True,
    disable_history=False,
)
@dataclass(frozen=True)
class Ipopt(Algorithm):
    # convergence criteria
    convergence_ftol_rel: NonNegativeFloat = CONVERGENCE_FTOL_REL
    dual_inf_tol: PositiveFloat = 1.0
    constr_viol_tol: PositiveFloat = 0.0001
    compl_inf_tol: PositiveFloat = 0.0001
    s_max: float = 100
    mu_target: NonNegativeFloat = 0.0
    # stopping criteria
    stopping_maxiter: PositiveInt = STOPPING_MAXITER
    stopping_max_wall_time_seconds: PositiveFloat = 1e20
    stopping_max_cpu_time: PositiveFloat = 1e20
    # acceptable criteria
    acceptable_iter: NonNegativeInt = 15
    acceptable_tol: PositiveFloat = 1e-6
    acceptable_dual_inf_tol: PositiveFloat = 1e-10
    acceptable_constr_viol_tol: PositiveFloat = 0.01
    acceptable_compl_inf_tol: PositiveFloat = 0.01
    acceptable_obj_change_tol: PositiveFloat = 1e20
    diverging_iterates_tol: PositiveFloat = 1e20
    nlp_lower_bound_inf: float = -1e19
    nlp_upper_bound_inf: float = 1e19
    fixed_variable_treatment: Literal[
        "make_parameter",
        "make_parameter_nodual",
        "relax_bounds",
        "make_constraint",
    ] = "make_parameter"
    dependency_detector: Literal["none", "mumps", "wsmp", "ma28"] | None = None
    dependency_detection_with_rhs: YesNoBool = False
    # bounds
    kappa_d: NonNegativeFloat = 1e-5
    bound_relax_factor: NonNegativeFloat = 1e-8
    honor_original_bounds: YesNoBool = False
    # derivatives
    check_derivatives_for_naninf: YesNoBool = False
    # not sure if we should support the following:
    jac_c_constant: YesNoBool = False
    jac_d_constant: YesNoBool = False
    hessian_constant: YesNoBool = False
    # scaling
    nlp_scaling_method: (
        Literal[
            "none",
            "user-scaling",
            "gradient-based",
            "equilibration-based",
        ]
        | None
    ) = "gradient-based"
    obj_scaling_factor: float = 1
    nlp_scaling_max_gradient: PositiveFloat = 100
    nlp_scaling_obj_target_gradient: NonNegativeFloat = 0.0
    nlp_scaling_constr_target_gradient: NonNegativeFloat = 0.0
    nlp_scaling_min_value: NonNegativeFloat = 1e-8
    # initialization
    bound_push: PositiveFloat = 0.01
    # TODO: refine type to fix the range (0,0.5]
    bound_frac: PositiveFloat = 0.01
    slack_bound_push: PositiveFloat = 0.01
    # TODO: refine type to fix the range (0,0.5]
    slack_bound_frac: PositiveFloat = 0.01
    constr_mult_init_max: NonNegativeFloat = 1000
    bound_mult_init_val: PositiveFloat = 1
    bound_mult_init_method: Literal[
        "constant",
        "mu-based",
    ] = "constant"
    least_square_init_primal: YesNoBool = False
    least_square_init_duals: YesNoBool = False
    # warm start
    warm_start_init_point: YesNoBool = False
    warm_start_same_structure: YesNoBool = False
    warm_start_bound_push: PositiveFloat = 0.001
    warm_start_bound_frac: PositiveFloat = 0.001
    warm_start_slack_bound_push: PositiveFloat = 0.001
    # TODO: refine type to fix the range (0,0.5])
    warm_start_slack_bound_frac: PositiveFloat = 0.001
    warm_start_mult_bound_push: PositiveFloat = 0.001
    warm_start_mult_init_max: float = 1e6
    warm_start_entire_iterate: YesNoBool = False
    warm_start_target_mu: float = 0.0
    # miscellaneous
    option_file_name: str = ""
    replace_bounds: YesNoBool = False
    skip_finalize_solution_call: YesNoBool = False
    timing_statistics: YesNoBool = False
    # barrier parameter update
    mu_max_fact: PositiveFloat = 1000
    mu_max: PositiveFloat = 100_000
    mu_min: PositiveFloat = 1e-11
    adaptive_mu_globalization: Literal[
        "obj-constr-filter",
        "kkt-error",
        "never-monotone-mode",
    ] = "obj-constr-filter"
    adaptive_mu_kkterror_red_iters: NonNegativeInt = 4
    # TODO: refine type to fix the range (0,1)
    adaptive_mu_kkterror_red_fact: PositiveFloat = 0.9999
    # TODO: refine type to fix the range (0,1)
    filter_margin_fact: PositiveFloat = 1e-5
    filter_max_margin: PositiveFloat = 1
    adaptive_mu_restore_previous_iterate: YesNoBool = False
    adaptive_mu_monotone_init_factor: PositiveFloat = 0.8
    adaptive_mu_kkt_norm_type: Literal[
        "max-norm",
        "2-norm-squared",
        "1-norm",
        "2-norm",
    ] = "2-norm-squared"
    mu_strategy: Literal["monotone", "adaptive"] = "monotone"
    mu_oracle: Literal[
        "probing",
        "quality-function",
        "loqo",
    ] = "quality-function"
    fixed_mu_oracle: Literal[
        "probing",
        "loqo",
        "quality-function",
        "average_compl",
    ] = "average_compl"
    mu_init: PositiveFloat = 0.1
    barrier_tol_factor: PositiveFloat = 10
    # TODO: refine type to fix the range (0,1)
    mu_linear_decrease_factor: PositiveFloat = 0.2
    # TODO: refine type to fix the range (1,2)
    mu_superlinear_decrease_power: GtOneFloat = 1.5
    mu_allow_fast_monotone_decrease: YesNoBool = True
    # TODO: refine type to fix the range (0,1)
    tau_min: PositiveFloat = 0.99
    sigma_max: PositiveFloat = 100
    sigma_min: NonNegativeFloat = 1e-6
    quality_function_norm_type: Literal[
        "max-norm",
        "2-norm-squared",
        "1-norm",
        "2-norm",
    ] = "2-norm-squared"
    quality_function_centrality: (
        Literal[
            "none",
            "reciprocal",
            "log",
            "cubed-reciprocal",
        ]
        | None
    ) = None
    quality_function_balancing_term: Literal["none", "cubic"] | None = None
    quality_function_max_section_steps: NonNegativeInt = 8
    # TODO: refine type to fix the range [0,1)
    quality_function_section_sigma_tol: NonNegativeFloat = 0.01
    # TODO: refine type to fix the range [0,1)
    quality_function_section_qf_tol: NonNegativeFloat = 0.0
    # line search
    line_search_method: Literal[
        "filter",
        "penalty",
        "cg-penalty",
    ] = "filter"
    # TODO: refine type to fix the range (0,1)
    alpha_red_factor: PositiveFloat = 0.5
    accept_every_trial_step: YesNoBool = False
    accept_after_max_steps: Literal[-1] | NonNegativeInt = -1
    alpha_for_y: Literal[
        "primal",
        "bound-mult",
        "min",
        "max",
        "full",
        "min-dual-infeas",
        "safer-min-dual-infeas",
        "primal-and-full",
        "dual-and-full",
        "acceptor",
    ] = "primal"
    alpha_for_y_tol: NonNegativeFloat = 10
    tiny_step_tol: NonNegativeFloat = 2.22045 * 1e-15
    tiny_step_y_tol: NonNegativeFloat = 0.01
    watchdog_shortened_iter_trigger: NonNegativeInt = 10
    watchdog_trial_iter_max: PositiveInt = 3
    theta_max_fact: PositiveFloat = 10_000
    theta_min_fact: PositiveFloat = 0.0001
    # TODO: refine type to fix the range (0,0.5)
    eta_phi: PositiveFloat = 1e-8
    delta: PositiveFloat = 1
    s_phi: GtOneFloat = 2.3
    s_theta: GtOneFloat = 1.1
    # TODO: refine type to fix the range (0,1)
    gamma_phi: PositiveFloat = 1e-8
    # TODO: refine type to fix the range (0,1)
    gamma_theta: PositiveFloat = 1e-5
    # TODO: refine type to fix the range (0,1)
    alpha_min_frac: PositiveFloat = 0.05
    max_soc: NonNegativeInt = 4
    kappa_soc: PositiveFloat = 0.99
    obj_max_inc: float = 5.0
    max_filter_resets: NonNegativeInt = 5
    filter_reset_trigger: PositiveInt = 5
    corrector_type: (
        Literal[
            "none",
            "affine",
            "primal-dual",
        ]
        | None
    ) = None
    skip_corr_if_neg_curv: YesNoBool = True
    skip_corr_in_monotone_mode: YesNoBool = True
    corrector_compl_avrg_red_fact: PositiveFloat = 1
    soc_method: Literal[0, 1] = 0
    nu_init: PositiveFloat = 1e-6
    nu_inc: PositiveFloat = 0.0001
    # TODO: refine type to fix the range (0,1)
    rho: PositiveFloat = 0.1
    kappa_sigma: PositiveFloat = 1e10
    recalc_y: YesNoBool = False
    recalc_y_feas_tol: PositiveFloat = 1e-6
    slack_move: NonNegativeFloat = 1.81899 * 1e-12
    constraint_violation_norm_type: Literal[
        "1-norm",
        "2-norm",
        "max-norm",
    ] = "1-norm"
    # step calculation
    mehrotra_algorithm: YesNoBool = False
    fast_step_computation: YesNoBool = False
    min_refinement_steps: NonNegativeInt = 1
    max_refinement_steps: NonNegativeInt = 10
    residual_ratio_max: PositiveFloat = 1e-10
    residual_ratio_singular: PositiveFloat = 1e-5
    residual_improvement_factor: PositiveFloat = 1
    neg_curv_test_tol: NonNegativeFloat = 0
    neg_curv_test_reg: YesNoBool = True
    max_hessian_perturbation: PositiveFloat = 1e20
    min_hessian_perturbation: NonNegativeFloat = 1e-20
    perturb_inc_fact_first: GtOneFloat = 100
    perturb_inc_fact: GtOneFloat = 8
    # TODO: refine type to fix the range (0,1)
    perturb_dec_fact: PositiveFloat = 0.333333
    first_hessian_perturbation: PositiveFloat = 0.0001
    jacobian_regularization_value: NonNegativeFloat = 1e-8
    jacobian_regularization_exponent: NonNegativeFloat = 0.25
    perturb_always_cd: YesNoBool = False
    # restoration phase
    expect_infeasible_problem: YesNoBool = False
    expect_infeasible_problem_ctol: NonNegativeFloat = 0.001
    expect_infeasible_problem_ytol: PositiveFloat = 1e8
    start_with_resto: YesNoBool = False
    soft_resto_pderror_reduction_factor: NonNegativeFloat = 0.9999
    max_soft_resto_iters: NonNegativeInt = 10
    # TODO: refine type to fix the range [0,1)
    required_infeasibility_reduction: NonNegativeFloat = 0.9
    max_resto_iter: NonNegativeInt = 3_000_000
    evaluate_orig_obj_at_resto_trial: YesNoBool = True
    resto_penalty_parameter: PositiveFloat = 1000
    resto_proximity_weight: NonNegativeFloat = 1
    bound_mult_reset_threshold: NonNegativeFloat = 1000
    constr_mult_reset_threshold: NonNegativeFloat = 0
    resto_failure_feasibility_threshold: NonNegativeFloat | None = None
    # hessian approximation
    limited_memory_aug_solver: Literal[
        "sherman-morrison",
        "extended",
    ] = "sherman-morrison"
    limited_memory_max_history: NonNegativeInt = 6
    limited_memory_update_type: Literal[
        "bfgs",
        "sr1",
    ] = "bfgs"
    limited_memory_initialization: Literal[
        "scalar1",
        "scalar2",
        "scalar3",
        "scalar4",
        "constant",
    ] = "scalar1"
    limited_memory_init_val: PositiveFloat = 1
    limited_memory_init_val_max: PositiveFloat = 1e8
    limited_memory_init_val_min: PositiveFloat = 1e-8
    limited_memory_max_skipping: PositiveInt = 2
    limited_memory_special_for_resto: YesNoBool = False
    hessian_approximation: Literal[
        "limited-memory",
        "exact",
    ] = "limited-memory"
    hessian_approximation_space: Literal[
        "nonlinear-variables",
        "all-variables",
    ] = "nonlinear-variables"
    # linear solver
    linear_solver: Literal[
        "mumps", "ma27", "ma57", "ma77", "ma86", "ma97", "pardiso", "custom"
    ] = "mumps"
    linear_solver_options: dict[str, Any] | None = None

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if not self.algo_info.is_available:
            raise NotInstalledError(
                "The 'ipopt' algorithm requires the cyipopt package to be installed. "
                "You can it with: `conda install -c conda-forge cyipopt`."
            )
        if self.acceptable_tol <= self.convergence_ftol_rel:
            raise ValueError(
                "The acceptable tolerance must be larger than the desired tolerance."
            )
        if self.mu_strategy not in ["monotone", "adaptive"]:
            raise ValueError(
                f"Unknown barrier strategy: {self.mu_strategy}."
                " It must be 'monotone' or 'adaptive'."
            )
        if self.nlp_upper_bound_inf < 0:
            raise ValueError("nlp_upper_bound_inf should be > 0.")
        if self.nlp_lower_bound_inf > 0:
            raise ValueError("nlp_lower_bound_inf should be < 0.")
        linear_solver_options = (
            {} if self.linear_solver_options is None else self.linear_solver_options
        )
        if self.resto_failure_feasibility_threshold is None:
            resto_failure_feasibility_threshold = 1e2 * self.convergence_ftol_rel
        else:
            resto_failure_feasibility_threshold = (
                self.resto_failure_feasibility_threshold
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
        boolean_linear_solver_options = [
            "linear_scaling_on_demand"
            "ma27_skip_inertia_check"
            "ma27_ignore_singularity"
            "ma57_automatic_scaling"
            "ma97_solve_blas3"
            "pardiso_redo_symbolic_fact_only_if_inertia_wrong"
            "pardiso_repeated_perturbation_means_singular"
            "pardiso_skip_inertia_check"
            "pardiso_iterative"
            "pardisomkl_redo_symbolic_fact_only_if_inertia_wrong"
            "pardisomkl_repeated_perturbation_means_singular"
            "pardisomkl_skip_inertia_check"
            "spral_ignore_numa"
            "spral_use_gpu"
            "wsmp_skip_inertia_check"
            "wsmp_no_pivoting"
        ]
        for key, val in linear_solver_options.items():
            if key in boolean_linear_solver_options:
                linear_solver_options[key] = _convert_bool_to_str(val, key)

        convert_bool_to_str_options = {
            "dependency_detection_with_rhs": self.dependency_detection_with_rhs,
            "check_derivatives_for_naninf": self.check_derivatives_for_naninf,
            "jac_c_constant": self.jac_c_constant,
            "jac_d_constant": self.jac_d_constant,
            "hessian_constant": self.hessian_constant,
            "least_square_init_primal": self.least_square_init_primal,
            "least_square_init_duals": self.least_square_init_duals,
            "warm_start_init_point": self.warm_start_init_point,
            "warm_start_same_structure": self.warm_start_same_structure,
            "warm_start_entire_iterate": self.warm_start_entire_iterate,
            "replace_bounds": self.replace_bounds,
            "skip_finalize_solution_call": self.skip_finalize_solution_call,
            "timing_statistics": self.timing_statistics,
            "adaptive_mu_restore_previous_iterate": (
                self.adaptive_mu_restore_previous_iterate
            ),
            "mu_allow_fast_monotone_decrease": self.mu_allow_fast_monotone_decrease,
            "accept_every_trial_step": self.accept_every_trial_step,
            "skip_corr_if_neg_curv": self.skip_corr_if_neg_curv,
            "skip_corr_in_monotone_mode": self.skip_corr_in_monotone_mode,
            "recalc_y": self.recalc_y,
            "mehrotra_algorithm": self.mehrotra_algorithm,
            "fast_step_computation": self.fast_step_computation,
            "neg_curv_test_reg": self.neg_curv_test_reg,
            "perturb_always_cd": self.perturb_always_cd,
            "expect_infeasible_problem": self.expect_infeasible_problem,
            "start_with_resto": self.start_with_resto,
            "evaluate_orig_obj_at_resto_trial": self.evaluate_orig_obj_at_resto_trial,
            "limited_memory_special_for_resto": self.limited_memory_special_for_resto,
            "honor_original_bounds": self.honor_original_bounds,
        }
        converted_bool_to_str_options = {
            key: _convert_bool_to_str(val, key)
            for key, val in convert_bool_to_str_options.items()
        }

        options = {
            # disable verbosity
            "print_level": 0,
            "ma77_print_level": -1,
            "ma86_print_level": -1,
            "ma97_print_level": -1,
            "pardiso_msglvl": 0,
            # disable derivative checker
            "derivative_test": "none",
            "s_max": float(self.s_max),
            "max_iter": self.stopping_maxiter,
            "max_wall_time": float(self.stopping_max_wall_time_seconds),
            "max_cpu_time": self.stopping_max_cpu_time,
            "dual_inf_tol": self.dual_inf_tol,
            "constr_viol_tol": self.constr_viol_tol,
            "compl_inf_tol": self.compl_inf_tol,
            # acceptable heuristic
            "acceptable_iter": self.acceptable_iter,
            "acceptable_tol": self.acceptable_tol,
            "acceptable_dual_inf_tol": self.acceptable_dual_inf_tol,
            "acceptable_constr_viol_tol": self.acceptable_constr_viol_tol,
            "acceptable_compl_inf_tol": self.acceptable_compl_inf_tol,
            "acceptable_obj_change_tol": self.acceptable_obj_change_tol,
            # bounds and more
            "diverging_iterates_tol": self.diverging_iterates_tol,
            "nlp_lower_bound_inf": self.nlp_lower_bound_inf,
            "nlp_upper_bound_inf": self.nlp_upper_bound_inf,
            "fixed_variable_treatment": self.fixed_variable_treatment,
            "dependency_detector": _convert_none_to_str(self.dependency_detector),
            "kappa_d": self.kappa_d,
            "bound_relax_factor": self.bound_relax_factor,
            "honor_original_bounds": self.honor_original_bounds,
            # scaling
            "nlp_scaling_method": _convert_none_to_str(self.nlp_scaling_method),
            "obj_scaling_factor": float(self.obj_scaling_factor),
            "nlp_scaling_max_gradient": float(self.nlp_scaling_max_gradient),
            "nlp_scaling_obj_target_gradient": float(
                self.nlp_scaling_obj_target_gradient
            ),
            "nlp_scaling_constr_target_gradient": float(
                self.nlp_scaling_constr_target_gradient
            ),
            "nlp_scaling_min_value": float(self.nlp_scaling_min_value),
            # initialization
            "bound_push": self.bound_push,
            "bound_frac": self.bound_frac,
            "slack_bound_push": self.slack_bound_push,
            "slack_bound_frac": self.slack_bound_frac,
            "constr_mult_init_max": float(self.constr_mult_init_max),
            "bound_mult_init_val": float(self.bound_mult_init_val),
            "bound_mult_init_method": self.bound_mult_init_method,
            # warm start
            "warm_start_bound_push": self.warm_start_bound_push,
            "warm_start_bound_frac": self.warm_start_bound_frac,
            "warm_start_slack_bound_push": self.warm_start_slack_bound_push,
            "warm_start_slack_bound_frac": self.warm_start_slack_bound_frac,
            "warm_start_mult_bound_push": self.warm_start_mult_bound_push,
            "warm_start_mult_init_max": self.warm_start_mult_init_max,
            "warm_start_target_mu": self.warm_start_target_mu,
            # more miscellaneous
            "option_file_name": self.option_file_name,
            # barrier parameter update
            "mu_target": float(self.mu_target),
            "mu_max_fact": float(self.mu_max_fact),
            "mu_max": float(self.mu_max),
            "mu_min": float(self.mu_min),
            "adaptive_mu_globalization": self.adaptive_mu_globalization,
            "adaptive_mu_kkterror_red_iters": self.adaptive_mu_kkterror_red_iters,
            "adaptive_mu_kkterror_red_fact": self.adaptive_mu_kkterror_red_fact,
            "filter_margin_fact": float(self.filter_margin_fact),
            "filter_max_margin": float(self.filter_max_margin),
            "adaptive_mu_monotone_init_factor": self.adaptive_mu_monotone_init_factor,
            "adaptive_mu_kkt_norm_type": self.adaptive_mu_kkt_norm_type,
            "mu_strategy": self.mu_strategy,
            "mu_oracle": self.mu_oracle,
            "fixed_mu_oracle": self.fixed_mu_oracle,
            "mu_init": self.mu_init,
            "barrier_tol_factor": float(self.barrier_tol_factor),
            "mu_linear_decrease_factor": self.mu_linear_decrease_factor,
            "mu_superlinear_decrease_power": self.mu_superlinear_decrease_power,
            "tau_min": self.tau_min,
            "sigma_max": float(self.sigma_max),
            "sigma_min": float(self.sigma_min),
            "quality_function_norm_type": self.quality_function_norm_type,
            "quality_function_centrality": _convert_none_to_str(
                self.quality_function_centrality
            ),
            "quality_function_balancing_term": _convert_none_to_str(
                self.quality_function_balancing_term
            ),
            "quality_function_max_section_steps": (
                self.quality_function_max_section_steps
            ),
            "quality_function_section_sigma_tol": (
                self.quality_function_section_sigma_tol
            ),
            "quality_function_section_qf_tol": self.quality_function_section_qf_tol,
            # linear search
            "line_search_method": self.line_search_method,
            "alpha_red_factor": self.alpha_red_factor,
            "accept_after_max_steps": self.accept_after_max_steps,
            "alpha_for_y": self.alpha_for_y,
            "alpha_for_y_tol": float(self.alpha_for_y_tol),
            "tiny_step_tol": self.tiny_step_tol,
            "tiny_step_y_tol": self.tiny_step_y_tol,
            "watchdog_shortened_iter_trigger": self.watchdog_shortened_iter_trigger,
            "watchdog_trial_iter_max": self.watchdog_trial_iter_max,
            "theta_max_fact": float(self.theta_max_fact),
            "theta_min_fact": self.theta_min_fact,
            "eta_phi": self.eta_phi,
            "delta": float(self.delta),
            "s_phi": self.s_phi,
            "s_theta": self.s_theta,
            "gamma_phi": self.gamma_phi,
            "gamma_theta": self.gamma_theta,
            "alpha_min_frac": self.alpha_min_frac,
            "max_soc": self.max_soc,
            "kappa_soc": self.kappa_soc,
            "obj_max_inc": float(self.obj_max_inc),
            "max_filter_resets": self.max_filter_resets,
            "filter_reset_trigger": self.filter_reset_trigger,
            "corrector_type": _convert_none_to_str(self.corrector_type),
            "corrector_compl_avrg_red_fact": float(self.corrector_compl_avrg_red_fact),
            "soc_method": self.soc_method,
            "nu_init": self.nu_init,
            "nu_inc": self.nu_inc,
            "rho": self.rho,
            "kappa_sigma": self.kappa_sigma,
            "recalc_y_feas_tol": self.recalc_y_feas_tol,
            "slack_move": self.slack_move,
            "constraint_violation_norm_type": self.constraint_violation_norm_type,
            # step calculation
            "min_refinement_steps": self.min_refinement_steps,
            "max_refinement_steps": self.max_refinement_steps,
            "residual_ratio_max": self.residual_ratio_max,
            "residual_ratio_singular": self.residual_ratio_singular,
            "residual_improvement_factor": float(self.residual_improvement_factor),
            "neg_curv_test_tol": float(self.neg_curv_test_tol),
            "max_hessian_perturbation": self.max_hessian_perturbation,
            "min_hessian_perturbation": self.min_hessian_perturbation,
            "perturb_inc_fact_first": float(self.perturb_inc_fact_first),
            "perturb_inc_fact": float(self.perturb_inc_fact),
            "perturb_dec_fact": float(self.perturb_dec_fact),
            "first_hessian_perturbation": float(self.first_hessian_perturbation),
            "jacobian_regularization_value": float(self.jacobian_regularization_value),
            "jacobian_regularization_exponent": float(
                self.jacobian_regularization_exponent
            ),
            # restoration phase
            "expect_infeasible_problem_ctol": self.expect_infeasible_problem_ctol,
            "expect_infeasible_problem_ytol": self.expect_infeasible_problem_ytol,
            "soft_resto_pderror_reduction_factor": (
                self.soft_resto_pderror_reduction_factor
            ),
            "max_soft_resto_iters": self.max_soft_resto_iters,
            "required_infeasibility_reduction": float(
                self.required_infeasibility_reduction
            ),
            "max_resto_iter": self.max_resto_iter,
            "resto_penalty_parameter": float(self.resto_penalty_parameter),
            "resto_proximity_weight": float(self.resto_proximity_weight),
            "bound_mult_reset_threshold": float(self.bound_mult_reset_threshold),
            "constr_mult_reset_threshold": float(self.constr_mult_reset_threshold),
            "resto_failure_feasibility_threshold": float(
                resto_failure_feasibility_threshold
            ),
            # hessian approximation
            "limited_memory_aug_solver": self.limited_memory_aug_solver,
            "limited_memory_max_history": self.limited_memory_max_history,
            "limited_memory_update_type": self.limited_memory_update_type,
            "limited_memory_initialization": self.limited_memory_initialization,
            "limited_memory_init_val": float(self.limited_memory_init_val),
            "limited_memory_init_val_max": self.limited_memory_init_val_max,
            "limited_memory_init_val_min": self.limited_memory_init_val_min,
            "limited_memory_max_skipping": self.limited_memory_max_skipping,
            "hessian_approximation": self.hessian_approximation,
            "hessian_approximation_space": self.hessian_approximation_space,
            # linear solver
            "linear_solver": self.linear_solver,
            **linear_solver_options,
            **converted_bool_to_str_options,
        }

        raw_res = cyipopt.minimize_ipopt(
            fun=problem.fun,
            x0=x0,
            bounds=_get_scipy_bounds(problem.bounds),
            jac=problem.jac,
            constraints=problem.nonlinear_constraints,
            tol=self.convergence_ftol_rel,
            options=options,
        )

        res = process_scipy_result(raw_res)

        return res


def _get_scipy_bounds(bounds: InternalBounds) -> ScipyBounds:
    return ScipyBounds(lb=bounds.lower, ub=bounds.upper)


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
