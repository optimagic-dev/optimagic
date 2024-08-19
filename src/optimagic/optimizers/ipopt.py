"""Implement cyipopt's Interior Point Optimizer."""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from annotated_types import Gt
from numpy.typing import NDArray

from optimagic.config import IS_CYIPOPT_INSTALLED
from optimagic.exceptions import NotInstalledError
from optimagic.optimization.algo_options import (
    CONVERGENCE_FTOL_REL,
    STOPPING_MAXITER,
)
from optimagic.optimization.algorithm import Algorithm, InternalOptimizeResult
from optimagic.optimization.internal_problem import InternalOptimizationProblem
from optimagic.typing import (
    AggregationLevel,
    GeOneInt,
    GtOneFloat,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    YesNoBool,
)

if IS_CYIPOPT_INSTALLED:
    pass


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
        "consant",
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
    line_search_method: [
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
    watchdog_trial_iter_max: GeOneInt = 3
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
    obj_max_inc: Gt = 5
    max_filter_resets: NonNegativeInt = 5
    filter_reset_trigger: GeOneInt = 5
    corrector_type: Literal[
        "none",
        "affine",
        "primal-dual",
    ] = None
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
    limited_memory_max_skipping: GeOneInt = 2
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
                f"Unknown barrier strategy: {self.mu_strategy}. It must be 'monotone' or "
                "'adaptive'."
            )
        if self.nlp_upper_bound_inf < 0:
            raise ValueError("nlp_upper_bound_inf should be > 0.")
        if self.nlp_lower_bound_inf > 0:
            raise ValueError("nlp_lower_bound_inf should be < 0.")
        linear_solver_options = (
            {} if self.linear_solver_options is None else self.linear_solver_options
        )
