"""Implement cyipopt's Interior Point Optimizer."""

from __future__ import annotations

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


@mark.minimizer(
    name="ipopt",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_CYIPOPT_INSTALLED,
    is_global=False,
    needs_jac=True,
    needs_hess=False,
    needs_bounds=False,
    supports_parallelism=False,
    supports_bounds=True,
    supports_infinite_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=True,
    disable_history=False,
)
@dataclass(frozen=True)
class Ipopt(Algorithm):
    """Minimize a scalar function using the Interior Point Optimizer (Ipopt).

    Ipopt (:cite:`Waechter2005b`) is an open-source solver for large-scale smooth
    nonlinear optimization developed within the COIN-OR project. It implements a
    primal-dual interior point method with a filter line search
    (:cite:`Waechter2005`, :cite:`Waechter2005a`) and, optionally, an adaptive
    update strategy for the barrier parameter (:cite:`Nocedal2009`). optimagic's
    support for Ipopt is built on `cyipopt
    <https://cyipopt.readthedocs.io/en/latest/index.html>`_, a Python wrapper for
    the `Ipopt optimization package
    <https://coin-or.github.io/Ipopt/index.html>`_.

    Ipopt is a local optimizer for smooth (ideally twice continuously
    differentiable) scalar objective functions. It supports bounds as well as
    nonlinear equality and inequality constraints and is designed to scale to very
    large problems. It requires first derivatives of the objective function and
    the constraints. By default, second derivatives are approximated with a
    limited-memory quasi-Newton method, so no Hessian has to be provided. Ipopt is
    not suitable for noisy or discontinuous objective functions.

    There are two levels of termination criteria. If the usual "desired"
    tolerances (see ``convergence_ftol_rel``, ``dual_inf_tol`` etc.) are satisfied
    at an iteration, the algorithm immediately terminates with a success message.
    On the other hand, if the algorithm encounters ``acceptable_iter`` many
    iterations in a row that are considered "acceptable", it will terminate before
    the desired convergence tolerance is met. This is useful in cases where the
    algorithm might not be able to achieve the "desired" level of accuracy.

    The options are analogous to the ones in the `Ipopt options reference
    <https://coin-or.github.io/Ipopt/OPTIONS.html>`_, with the exception of the
    linear solver options, which are here bundled into the dictionary
    ``linear_solver_options``. Any option that takes "yes" and "no" in the Ipopt
    documentation can also be passed as ``True`` and ``False``, respectively, and
    any option that accepts "none" in Ipopt accepts a Python ``None``.

    The following options are not supported:

    - ``num_linear_variables``: since optimagic may reparametrize your problem and
      this changes the parameter problem, we do not support this option
    - derivative checks
    - print options.

    .. note::
       To use this algorithm you need to have `cyipopt installed
       <https://cyipopt.readthedocs.io/en/latest/index.html>`_
       (``conda install -c conda-forge cyipopt``).

    """

    # convergence criteria
    convergence_ftol_rel: NonNegativeFloat = CONVERGENCE_FTOL_REL
    """Relative convergence tolerance of the algorithm.

    The algorithm terminates successfully if the (scaled) nonlinear programming
    error becomes smaller than this value. This is passed to Ipopt as the ``tol``
    option. optimagic's default (2e-9) is stricter than Ipopt's own default of
    1e-8.

    """

    dual_inf_tol: PositiveFloat = 1.0
    """Desired threshold for the dual infeasibility.

    Successful termination requires that the max-norm of the (unscaled) dual
    infeasibility is less than this threshold.

    """

    constr_viol_tol: PositiveFloat = 0.0001
    """Desired threshold for the constraint and variable bound violation.

    Successful termination requires that the max-norm of the (unscaled) constraint
    violation is less than this threshold. If ``bound_relax_factor`` is not zero,
    Ipopt relaxes the given variable bounds; the value of ``constr_viol_tol`` is
    then used to restrict the absolute amount of this bound relaxation.

    """

    compl_inf_tol: PositiveFloat = 0.0001
    """Desired threshold for the complementarity conditions.

    Successful termination requires that the max-norm of the (unscaled)
    complementarity is less than this threshold.

    """

    s_max: float = 100
    """Scaling threshold for the NLP error.

    See paragraph after Eqn. (6) in :cite:`Waechter2005b`.

    """

    mu_target: NonNegativeFloat = 0.0
    """Desired value of the complementarity.

    Usually, the barrier parameter is driven to zero and the termination test for
    complementarity is measured with respect to zero complementarity. However, in
    some cases it might be desirable to have Ipopt solve the barrier problem for a
    strictly positive value of the barrier parameter. In this case, ``mu_target``
    specifies the final value of the barrier parameter and the termination tests
    are then defined with respect to the barrier problem for this value.

    """

    # stopping criteria
    stopping_maxiter: PositiveInt = STOPPING_MAXITER
    """Maximum number of iterations.

    If the maximum number of iterations is reached, the optimization stops, but we
    do not count this as successful convergence. One iteration might need several
    criterion evaluations, e.g. in a line search. optimagic's default (1,000,000)
    is much larger than Ipopt's own default of 3000.

    """

    stopping_max_wall_time_seconds: PositiveFloat = 1e20
    """Maximum number of wall clock seconds Ipopt may use to solve one problem.

    If this limit is exceeded during the convergence check, Ipopt terminates with
    a corresponding message.

    """

    stopping_max_cpu_time: PositiveFloat = 1e20
    """Maximum number of CPU seconds Ipopt may use to solve one problem.

    If this limit is exceeded during the convergence check, Ipopt terminates with
    a corresponding message.

    """

    # acceptable criteria
    acceptable_iter: NonNegativeInt = 15
    """Number of successive "acceptable" iterates before termination.

    If the algorithm encounters this many successive "acceptable" iterates, it
    terminates, assuming that the problem has been solved to the best possible
    accuracy given round-off. If set to zero, this heuristic is disabled.

    """

    acceptable_tol: PositiveFloat = 1e-6
    """"Acceptable" convergence tolerance (relative).

    Determines which (scaled) overall optimality error is considered to be
    "acceptable". Must be larger than ``convergence_ftol_rel``.

    """

    acceptable_dual_inf_tol: PositiveFloat = 1e-10
    """"Acceptance" threshold for the dual infeasibility.

    "Acceptable" termination requires that the max-norm of the (unscaled) dual
    infeasibility is less than this threshold; see also ``acceptable_tol``. Note
    that optimagic's default (1e-10) is much stricter than Ipopt's own default of
    1e10.

    """

    acceptable_constr_viol_tol: PositiveFloat = 0.01
    """"Acceptance" threshold for the constraint violation.

    "Acceptable" termination requires that the max-norm of the (unscaled)
    constraint violation is less than this threshold; see also
    ``acceptable_tol``.

    """

    acceptable_compl_inf_tol: PositiveFloat = 0.01
    """"Acceptance" threshold for the complementarity conditions.

    "Acceptable" termination requires that the max-norm of the (unscaled)
    complementarity is less than this threshold; see also ``acceptable_tol``.

    """

    acceptable_obj_change_tol: PositiveFloat = 1e20
    """"Acceptance" stopping criterion based on the objective function change.

    If the relative change of the objective function (scaled by
    ``max(1, |f(x)|)``) is less than this value, this part of the acceptable
    tolerance termination is satisfied; see also ``acceptable_tol``. This is
    useful for the quasi-Newton option, which has trouble bringing down the dual
    infeasibility.

    """

    diverging_iterates_tol: PositiveFloat = 1e20
    """Threshold for the maximal value of the primal iterates.

    If any component of the primal iterates exceeds this value in absolute terms,
    the optimization is aborted with the exit message that the iterates seem to be
    diverging.

    """

    nlp_lower_bound_inf: float = -1e19
    """Any bound less or equal to this value will be considered minus infinity
    (i.e. the variable is treated as not lower bounded)."""

    nlp_upper_bound_inf: float = 1e19
    """Any bound greater or equal to this value will be considered plus infinity
    (i.e. the variable is treated as not upper bounded)."""

    fixed_variable_treatment: Literal[
        "make_parameter",
        "make_parameter_nodual",
        "relax_bounds",
        "make_constraint",
    ] = "make_parameter"
    """Determines how fixed variables are handled.

    Possible values:

    - "make_parameter": remove fixed variables from the optimization variables
    - "make_parameter_nodual": as "make_parameter", but do not compute bound
      multipliers for fixed variables
    - "make_constraint": add equality constraints fixing the variables
    - "relax_bounds": relax the fixing bound constraints (according to
      ``bound_relax_factor``)

    The main difference is that in the "make_constraint" case the starting point
    still has the fixed variables at their given values, whereas with
    "make_parameter(_nodual)" the functions are always evaluated with the fixed
    values for those variables. For all but "make_parameter_nodual", bound
    multipliers are computed for the fixed variables.

    """

    dependency_detector: Literal["none", "mumps", "wsmp", "ma28"] | None = None
    """Which linear solver should be used to detect linearly dependent equality
    constraints.

    ``None`` (or "none") means no check is performed. This feature is experimental
    and does not work well.

    """

    dependency_detection_with_rhs: YesNoBool = False
    """Whether the right hand sides of the constraints should be considered in
    addition to the gradients during dependency detection."""

    # bounds
    kappa_d: NonNegativeFloat = 1e-5
    """Weight for the linear damping term (to handle one-sided bounds).

    See Section 3.7 in :cite:`Waechter2005b`.

    """

    bound_relax_factor: NonNegativeFloat = 1e-8
    """Factor for the initial relaxation of the bounds.

    Before the start of the optimization, the bounds given by the user are relaxed
    by this relative factor. In addition, ``constr_viol_tol`` is used to bound the
    relaxation by an absolute value. If set to zero, the bound relaxation is
    disabled. See Eqn. (35) in :cite:`Waechter2005b`. Note that the constraint
    violation reported by Ipopt at the end of the solution process does not
    include violations of the original (non-relaxed) variable bounds. See also
    ``honor_original_bounds``.

    """

    honor_original_bounds: YesNoBool = False
    """Whether the final point should be projected back into the original bounds.

    Ipopt might relax the bounds during the optimization (see
    ``bound_relax_factor``). This option determines whether the final point is
    projected back into the user-provided original bounds after the optimization.
    Note that violations of constraints and complementarity reported by Ipopt at
    the end of the solution process are for the non-projected point.

    """

    # derivatives
    check_derivatives_for_naninf: YesNoBool = False
    """Whether to check for NaN or Inf in the derivative matrices.

    Activating this option will cause an error if an invalid number is detected in
    the constraint Jacobians or the Lagrangian Hessian. If it is not activated,
    the test is skipped and the algorithm might proceed with invalid numbers and
    fail.

    """

    # not sure if we should support the following:
    jac_c_constant: YesNoBool = False
    """Whether to assume that all equality constraints are linear.

    Activating this option will cause Ipopt to ask for the Jacobian of the
    equality constraints only once and reuse this information later.

    """

    jac_d_constant: YesNoBool = False
    """Whether to assume that all inequality constraints are linear.

    Activating this option will cause Ipopt to ask for the Jacobian of the
    inequality constraints only once and reuse this information later.

    """

    hessian_constant: YesNoBool = False
    """Whether to assume that the problem is a QP (quadratic objective, linear
    constraints).

    Activating this option will cause Ipopt to ask for the Hessian of the
    Lagrangian function only once and reuse this information later.

    """

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
    """Technique used for scaling the problem internally before it is solved.

    Possible values:

    - ``None`` or "none": no problem scaling is performed
    - "user-scaling": the scaling parameters come from the user
    - "gradient-based": scale the problem so that the maximum gradient at the
      starting point is ``nlp_scaling_max_gradient``
    - "equilibration-based": scale the problem so that first derivatives are of
      order 1 at random points (uses the Harwell routine MC19)

    """

    obj_scaling_factor: float = 1
    """Scaling factor for the objective function.

    The scaling is only seen internally by Ipopt, the unscaled objective is
    reported in the output. If additional scaling parameters are computed (e.g.
    with user-scaling or gradient-based scaling), both factors are multiplied. If
    this value is negative, Ipopt maximizes the objective function instead of
    minimizing it.

    """

    nlp_scaling_max_gradient: PositiveFloat = 100
    """Maximum gradient after NLP scaling.

    This is the gradient scaling cut-off. If the maximum gradient is above this
    value, gradient based scaling is performed, with scaling parameters calculated
    to scale the maximum gradient back to this value. This is g_max in Section 3.8
    of :cite:`Waechter2005b`. Only used if ``nlp_scaling_method`` is
    "gradient-based".

    """

    nlp_scaling_obj_target_gradient: NonNegativeFloat = 0.0
    """Advanced feature. Target value for the objective function gradient size.

    If a positive number is chosen, the scaling factor for the objective function
    is computed so that the gradient has the max-norm of the given size at the
    starting point. This overrides ``nlp_scaling_max_gradient`` for the objective
    function.

    """

    nlp_scaling_constr_target_gradient: NonNegativeFloat = 0.0
    """Advanced feature. Target value for the constraint function gradient size.

    If a positive number is chosen, the scaling factors for the constraint
    functions are computed so that the gradient has the max-norm of the given size
    at the starting point. This overrides ``nlp_scaling_max_gradient`` for the
    constraint functions.

    """

    nlp_scaling_min_value: NonNegativeFloat = 1e-8
    """Minimum value of gradient-based scaling values.

    This is the lower bound for the scaling factors computed by the gradient-based
    scaling method. If some derivatives of some functions are huge, the scaling
    factors will otherwise become very small, and the (unscaled) final constraint
    violation, for example, might then be significant. Only used if
    ``nlp_scaling_method`` is "gradient-based".

    """

    # initialization
    bound_push: PositiveFloat = 0.01
    """Desired minimum absolute distance from the initial point to the bounds.

    Determines how much the initial point might have to be modified in order to be
    sufficiently inside the bounds (together with ``bound_frac``). This is kappa_1
    in Section 3.6 of :cite:`Waechter2005b`.

    """

    # TODO: refine type to fix the range (0,0.5]
    bound_frac: PositiveFloat = 0.01
    """Desired minimum relative distance from the initial point to the bounds.

    Determines how much the initial point might have to be modified in order to be
    sufficiently inside the bounds (together with ``bound_push``). This is kappa_2
    in Section 3.6 of :cite:`Waechter2005b`. The valid range is (0, 0.5].

    """

    slack_bound_push: PositiveFloat = 0.01
    """Desired minimum absolute distance from the initial slack to the bounds.

    Determines how much the initial slack variables might have to be modified in
    order to be sufficiently inside the inequality bounds (together with
    ``slack_bound_frac``). This is kappa_1 in Section 3.6 of
    :cite:`Waechter2005b`.

    """

    # TODO: refine type to fix the range (0,0.5]
    slack_bound_frac: PositiveFloat = 0.01
    """Desired minimum relative distance from the initial slack to the bounds.

    Determines how much the initial slack variables might have to be modified in
    order to be sufficiently inside the inequality bounds (together with
    ``slack_bound_push``). This is kappa_2 in Section 3.6 of
    :cite:`Waechter2005b`. The valid range is (0, 0.5].

    """

    constr_mult_init_max: NonNegativeFloat = 1000
    """Maximum allowed least-squares guess of the constraint multipliers.

    Determines how large the initial least-squares guesses of the constraint
    multipliers are allowed to be (in max-norm). If the guess is larger than this
    value, it is discarded and all constraint multipliers are set to zero. This
    option is also used when initializing the restoration phase.

    """

    bound_mult_init_val: PositiveFloat = 1
    """Initial value for the bound multipliers.

    All dual variables corresponding to bound constraints are initialized to this
    value.

    """

    bound_mult_init_method: Literal[
        "constant",
        "mu-based",
    ] = "constant"
    """Initialization method for the bound multipliers.

    If "constant" is chosen, all bound multipliers are initialized to the value of
    ``bound_mult_init_val``. If "mu-based" is chosen, each value is initialized to
    ``mu_init`` divided by the corresponding slack variable. The latter might be
    useful if the starting point is close to the optimal solution.

    """

    least_square_init_primal: YesNoBool = False
    """Least-squares initialization of the primal variables.

    If enabled, Ipopt ignores the user-provided point and solves a least-squares
    problem for the primal variables (x and s) to fit the linearized equality and
    inequality constraints. This might be useful if the user doesn't know anything
    about the starting point, or for solving an LP or QP.

    """

    least_square_init_duals: YesNoBool = False
    """Least-squares initialization of all dual variables.

    If enabled, Ipopt tries to compute least-squares multipliers, considering all
    dual variables. If successful, the bound multipliers are possibly corrected to
    be at least ``bound_mult_init_val``. This might be useful if the user doesn't
    know anything about the starting point, or for solving an LP or QP. This
    overwrites the option ``bound_mult_init_method``.

    """

    # warm start
    warm_start_init_point: YesNoBool = False
    """Whether this optimization should use a warm start initialization, where
    values of the primal and dual variables are given (e.g. from a previous
    optimization of a related problem)."""

    warm_start_same_structure: YesNoBool = False
    """Advanced feature.

    Indicates whether a problem with a structure identical to the previous one is
    to be solved. If enabled, the algorithm assumes that an NLP is now to be
    solved whose structure is identical to one that was already considered (with
    the same NLP object).

    """

    warm_start_bound_push: PositiveFloat = 0.001
    """Same as ``bound_push`` for the warm start initializer."""

    # TODO: refine type to fix the range (0,0.5]
    warm_start_bound_frac: PositiveFloat = 0.001
    """Same as ``bound_frac`` for the warm start initializer.

    The valid range is (0, 0.5].

    """

    warm_start_slack_bound_push: PositiveFloat = 0.001
    """Same as ``slack_bound_push`` for the warm start initializer."""

    # TODO: refine type to fix the range (0,0.5])
    warm_start_slack_bound_frac: PositiveFloat = 0.001
    """Same as ``slack_bound_frac`` for the warm start initializer.

    The valid range is (0, 0.5].

    """

    warm_start_mult_bound_push: PositiveFloat = 0.001
    """Same as ``bound_push`` for the bound multipliers in the warm start
    initializer."""

    warm_start_mult_init_max: float = 1e6
    """Maximum initial value for the equality constraint multipliers when using a
    warm start."""

    warm_start_entire_iterate: YesNoBool = False
    """Whether to use the GetWarmStartIterate method in the NLP instead of
    GetStartingPoint when using a warm start."""

    warm_start_target_mu: float = 0.0
    """Advanced and experimental feature."""

    # miscellaneous
    option_file_name: str = ""
    """File name of an Ipopt options file.

    By default, optimagic passes an empty string, which disables reading of an
    options file. If a name is given, Ipopt reads additional options from that
    file.

    """

    replace_bounds: YesNoBool = False
    """Whether all variable bounds should be replaced by inequality constraints.

    This option must be set for the inexact algorithm.

    """

    skip_finalize_solution_call: YesNoBool = False
    """Whether the call to NLP::FinalizeSolution after the optimization should be
    suppressed."""

    timing_statistics: YesNoBool = False
    """Whether to measure the time spent in the components of Ipopt and the NLP
    evaluation.

    The overall algorithm time is unaffected by this option.

    """

    # barrier parameter update
    mu_max_fact: PositiveFloat = 1000
    """Factor for the initialization of the maximum value of the barrier
    parameter.

    The upper bound on the barrier parameter is computed as the average
    complementarity at the initial point times the value of this option. Only used
    if ``mu_strategy`` is "adaptive".

    """

    mu_max: PositiveFloat = 100_000
    """Maximum value for the barrier parameter.

    This option specifies an upper bound on the barrier parameter in the adaptive
    mu selection mode. If this option is set, it overwrites the effect of
    ``mu_max_fact``. Only used if ``mu_strategy`` is "adaptive".

    """

    mu_min: PositiveFloat = 1e-11
    """Minimum value for the barrier parameter.

    This option specifies the lower bound on the barrier parameter in the adaptive
    mu selection mode. In Ipopt, it is by default set to the minimum of 1e-11 and
    ``min(tol, compl_inf_tol) / (barrier_tol_factor + 1)``, which should be a
    reasonable value. Only used if ``mu_strategy`` is "adaptive".

    """

    adaptive_mu_globalization: Literal[
        "obj-constr-filter",
        "kkt-error",
        "never-monotone-mode",
    ] = "obj-constr-filter"
    """Globalization strategy for the adaptive mu selection mode.

    To achieve global convergence of the adaptive version, the algorithm has to
    switch to the monotone mode (Fiacco-McCormick approach) when convergence does
    not seem to appear. This option sets the criterion used to decide when to make
    this switch. Possible values:

    - "kkt-error": nonmonotone decrease of the KKT error
    - "obj-constr-filter": two-dimensional filter for objective and constraint
      violation
    - "never-monotone-mode": disables globalization

    Only used if ``mu_strategy`` is "adaptive".

    """

    adaptive_mu_kkterror_red_iters: NonNegativeInt = 4
    """Advanced feature. Maximum number of iterations requiring sufficient
    progress.

    For the "kkt-error" based globalization strategy, sufficient progress must be
    made for this many iterations. If this number of iterations is exceeded, the
    globalization strategy switches to the monotone mode.

    """

    # TODO: refine type to fix the range (0,1)
    adaptive_mu_kkterror_red_fact: PositiveFloat = 0.9999
    """Advanced feature. Sufficient decrease factor for the "kkt-error"
    globalization strategy.

    For the "kkt-error" based globalization strategy, the error must decrease by
    this factor to be deemed sufficient decrease. The valid range is (0, 1).

    """

    # TODO: refine type to fix the range (0,1)
    filter_margin_fact: PositiveFloat = 1e-5
    """Advanced feature. Factor determining the width of the margin for the
    "obj-constr-filter" adaptive globalization strategy.

    Sufficient progress for a filter entry is defined as: (new obj) < (filter obj)
    - filter_margin_fact * (new constr-viol) or (new constr-viol) < (filter
    constr-viol) - filter_margin_fact * (new constr-viol). For the description of
    the "kkt-error-filter" option see ``filter_max_margin``. The valid range is
    (0, 1).

    """

    filter_max_margin: PositiveFloat = 1
    """Advanced feature.

    Maximum width of the margin in the "obj-constr-filter" adaptive globalization
    strategy.

    """

    adaptive_mu_restore_previous_iterate: YesNoBool = False
    """Advanced feature. Whether the previous accepted iterate should be restored
    if the monotone mode is entered.

    When the globalization strategy for the adaptive barrier algorithm switches to
    the monotone mode, it can either start from the most recent iterate (False) or
    from the last iterate that was accepted (True).

    """

    adaptive_mu_monotone_init_factor: PositiveFloat = 0.8
    """Advanced feature. Determines the initial value of the barrier parameter
    when switching to the monotone mode.

    When the globalization strategy for the adaptive barrier algorithm switches to
    the monotone mode and ``fixed_mu_oracle`` is chosen as "average_compl", the
    barrier parameter is set to the current average complementarity times the
    value of this option.

    """

    adaptive_mu_kkt_norm_type: Literal[
        "max-norm",
        "2-norm-squared",
        "1-norm",
        "2-norm",
    ] = "2-norm-squared"
    """Advanced feature. Norm used for the KKT error in the adaptive mu
    globalization strategies.

    When computing the KKT error for the globalization strategies, the norm to be
    used is specified with this option. Note that this option is also used in the
    quality function based mu oracle.

    """

    mu_strategy: Literal["monotone", "adaptive"] = "monotone"
    """Update strategy for the barrier parameter.

    "monotone" uses the monotone (Fiacco-McCormick) strategy; "adaptive" uses the
    adaptive update strategy from :cite:`Nocedal2009`.

    """

    mu_oracle: Literal[
        "probing",
        "quality-function",
        "loqo",
    ] = "quality-function"
    """Oracle for a new barrier parameter in the adaptive strategy.

    Determines how a new barrier parameter is computed in each "free-mode"
    iteration of the adaptive barrier parameter strategy. Possible values:

    - "probing": Mehrotra's probing heuristic
    - "loqo": LOQO's centrality rule
    - "quality-function": minimize a quality function

    Only used if ``mu_strategy`` is "adaptive".

    """

    fixed_mu_oracle: Literal[
        "probing",
        "loqo",
        "quality-function",
        "average_compl",
    ] = "average_compl"
    """Oracle for the barrier parameter when switching to the fixed mode.

    Determines how the first value of the barrier parameter should be computed
    when switching to the "monotone mode" in the adaptive strategy. In addition to
    the choices for ``mu_oracle``, "average_compl" bases the value on the current
    average complementarity. Only used if ``mu_strategy`` is "adaptive".

    """

    mu_init: PositiveFloat = 0.1
    """Initial value for the barrier parameter.

    This option determines the initial value for the barrier parameter mu. It is
    only relevant in the monotone, Fiacco-McCormick version of the algorithm,
    i.e. if ``mu_strategy`` is "monotone".

    """

    barrier_tol_factor: PositiveFloat = 10
    """Factor for mu in the barrier stop test.

    The convergence tolerance for each barrier problem in the monotone mode is the
    value of the barrier parameter times this factor. This option is also used in
    the adaptive mu strategy during the monotone mode. This is kappa_epsilon in
    :cite:`Waechter2005b`.

    """

    # TODO: refine type to fix the range (0,1)
    mu_linear_decrease_factor: PositiveFloat = 0.2
    """Determines the linear decrease rate of the barrier parameter.

    For the Fiacco-McCormick update procedure, the new barrier parameter mu is
    obtained by taking the minimum of ``mu * mu_linear_decrease_factor`` and
    ``mu ** mu_superlinear_decrease_power``. This is kappa_mu in
    :cite:`Waechter2005b`. This option is also used in the adaptive mu strategy
    during the monotone mode. The valid range is (0, 1).

    """

    # TODO: refine type to fix the range (1,2)
    mu_superlinear_decrease_power: GtOneFloat = 1.5
    """Determines the superlinear decrease rate of the barrier parameter.

    For the Fiacco-McCormick update procedure, the new barrier parameter mu is
    obtained by taking the minimum of ``mu * mu_linear_decrease_factor`` and
    ``mu ** mu_superlinear_decrease_power``. This is theta_mu in
    :cite:`Waechter2005b`. This option is also used in the adaptive mu strategy
    during the monotone mode. The valid range is (1, 2).

    """

    mu_allow_fast_monotone_decrease: YesNoBool = True
    """Advanced feature. Allow skipping of the barrier problem if the barrier test
    is already met.

    If False, the algorithm takes at least one iteration per barrier problem, even
    if the barrier test is already met for the updated barrier parameter.

    """

    # TODO: refine type to fix the range (0,1)
    tau_min: PositiveFloat = 0.99
    """Advanced feature. Lower bound on the fraction-to-the-boundary parameter
    tau.

    This is tau_min in :cite:`Waechter2005b`. This option is also used in the
    adaptive mu strategy during the monotone mode. The valid range is (0, 1).

    """

    sigma_max: PositiveFloat = 100
    """Advanced feature. Maximum value of the centering parameter.

    This is the upper bound for the centering parameter chosen by the quality
    function based barrier parameter update. Only used if ``mu_oracle`` is
    "quality-function".

    """

    sigma_min: NonNegativeFloat = 1e-6
    """Advanced feature. Minimum value of the centering parameter.

    This is the lower bound for the centering parameter chosen by the quality
    function based barrier parameter update. Only used if ``mu_oracle`` is
    "quality-function".

    """

    quality_function_norm_type: Literal[
        "max-norm",
        "2-norm-squared",
        "1-norm",
        "2-norm",
    ] = "2-norm-squared"
    """Advanced feature. Norm used for the components of the quality function.

    Only used if ``mu_oracle`` is "quality-function".

    """

    quality_function_centrality: (
        Literal[
            "none",
            "reciprocal",
            "log",
            "cubed-reciprocal",
        ]
        | None
    ) = None
    """Advanced feature. The penalty term for centrality that is included in the
    quality function.

    This determines whether a term is added to the quality function to penalize
    deviation from centrality with respect to complementarity. The complementarity
    measure here is the xi in the LOQO update rule. Possible values:

    - ``None`` or "none": no penalty term is added
    - "log": complementarity times the log of the centrality measure
    - "reciprocal": complementarity times the reciprocal of the centrality measure
    - "cubed-reciprocal": complementarity times the reciprocal of the centrality
      measure cubed

    Only used if ``mu_oracle`` is "quality-function".

    """

    quality_function_balancing_term: Literal["none", "cubic"] | None = None
    """Advanced feature. The balancing term included in the quality function for
    centrality.

    This determines whether a term is added to the quality function that penalizes
    situations where the complementarity is much smaller than the dual and primal
    infeasibilities. ``None`` (or "none") adds no balancing term; "cubic" adds
    ``max(0, max(dual_inf, primal_inf) - compl)**3``. Only used if ``mu_oracle``
    is "quality-function".

    """

    quality_function_max_section_steps: NonNegativeInt = 8
    """Maximum number of search steps during the direct search procedure
    determining the optimal centering parameter.

    The golden section search is performed for the quality function based mu
    oracle. Only used if ``mu_oracle`` is "quality-function".

    """

    # TODO: refine type to fix the range [0,1)
    quality_function_section_sigma_tol: NonNegativeFloat = 0.01
    """Advanced feature. Tolerance for the section search procedure determining
    the optimal centering parameter (in sigma space).

    The golden section search is performed for the quality function based mu
    oracle. Only used if ``mu_oracle`` is "quality-function". The valid range is
    [0, 1).

    """

    # TODO: refine type to fix the range [0,1)
    quality_function_section_qf_tol: NonNegativeFloat = 0.0
    """Advanced feature. Tolerance for the golden section search procedure
    determining the optimal centering parameter (in the function value space).

    Only used if ``mu_oracle`` is "quality-function". The valid range is [0, 1).

    """

    # line search
    line_search_method: Literal[
        "filter",
        "penalty",
        "cg-penalty",
    ] = "filter"
    """Advanced feature. Globalization method used in the backtracking line
    search.

    Only the "filter" choice is officially supported, but sometimes good results
    might be obtained with "cg-penalty" (Chen-Goldfarb penalty function) or
    "penalty" (standard penalty function).

    """

    # TODO: refine type to fix the range (0,1)
    alpha_red_factor: PositiveFloat = 0.5
    """Advanced feature. Fractional reduction of the trial step size in the
    backtracking line search.

    At every step of the backtracking line search, the trial step size is reduced
    by this factor. The valid range is (0, 1).

    """

    accept_every_trial_step: YesNoBool = False
    """Always accept the first trial step.

    Setting this option to True essentially disables the line search and makes the
    algorithm take aggressive steps, without global convergence guarantees.

    """

    accept_after_max_steps: Literal[-1] | NonNegativeInt = -1
    """Advanced feature. Accept a trial point after at most this number of steps,
    even if it does not satisfy the line search conditions.

    Setting this to -1 disables this heuristic.

    """

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
    """Method to determine the step size for the equality constraint multipliers
    (alpha_y).

    Possible values:

    - "primal": use the primal step size
    - "bound-mult": use the step size for the bound multipliers (good for LPs)
    - "min": use the min of the primal and bound multiplier step sizes
    - "max": use the max of the primal and bound multiplier step sizes
    - "full": take a full step of size one
    - "min-dual-infeas": choose the step size minimizing the new dual
      infeasibility
    - "safer-min-dual-infeas": like "min-dual-infeas", but safeguarded by "min"
      and "max"
    - "primal-and-full": use the primal step size, and a full step if
      ``delta_x <= alpha_for_y_tol``
    - "dual-and-full": use the dual step size, and a full step if
      ``delta_x <= alpha_for_y_tol``
    - "acceptor": call LSAcceptor to get the step size for y

    """

    alpha_for_y_tol: NonNegativeFloat = 10
    """Tolerance for switching to full equality multiplier steps.

    This is only relevant if ``alpha_for_y`` is "primal-and-full" or
    "dual-and-full". The step size for the equality constraint multipliers is
    taken to be one if the max-norm of the primal step is less than this
    tolerance.

    """

    tiny_step_tol: NonNegativeFloat = 2.22045 * 1e-15
    """Advanced feature. Tolerance for detecting numerically insignificant steps.

    If the search direction in the primal variables (x and s) is, in relative
    terms for each component, less than this value, the algorithm accepts the full
    step without a line search. If this happens repeatedly, the algorithm
    terminates with a corresponding exit message. The default value is 10 times
    machine precision.

    """

    tiny_step_y_tol: NonNegativeFloat = 0.01
    """Advanced feature. Tolerance for quitting because of numerically
    insignificant steps.

    If the search direction in the primal variables (x and s) is, in relative
    terms for each component, repeatedly less than ``tiny_step_tol`` and the step
    in the y variables is smaller than this threshold, the algorithm terminates.

    """

    watchdog_shortened_iter_trigger: NonNegativeInt = 10
    """Number of shortened iterations that trigger the watchdog procedure.

    If the number of successive iterations in which the backtracking line search
    did not accept the first trial point exceeds this number, the watchdog
    procedure is activated. Choosing 0 disables the watchdog procedure.

    """

    watchdog_trial_iter_max: PositiveInt = 3
    """Maximum number of watchdog iterations.

    This option determines the number of trial iterations allowed before the
    watchdog procedure is aborted and the algorithm returns to the stored point.

    """

    theta_max_fact: PositiveFloat = 10_000
    """Advanced feature. Determines the upper bound for the constraint violation
    in the filter.

    The algorithmic parameter theta_max is determined as this value times the
    maximum of 1 and the constraint violation at the initial point. Any point with
    a constraint violation larger than theta_max is unacceptable to the filter
    (see Eqn. (21) in :cite:`Waechter2005b`).

    """

    theta_min_fact: PositiveFloat = 0.0001
    """Advanced feature. Determines the constraint violation threshold in the
    switching rule.

    The algorithmic parameter theta_min is determined as this value times the
    maximum of 1 and the constraint violation at the initial point. The switching
    rule treats an iteration as an h-type iteration whenever the current
    constraint violation is larger than theta_min (see paragraph before Eqn. (19)
    in :cite:`Waechter2005b`).

    """

    # TODO: refine type to fix the range (0,0.5)
    eta_phi: PositiveFloat = 1e-8
    """Advanced feature. Relaxation factor in the Armijo condition.

    See Eqn. (20) in :cite:`Waechter2005b`. The valid range is (0, 0.5).

    """

    delta: PositiveFloat = 1
    """Advanced feature. Multiplier for the constraint violation in the switching
    rule.

    See Eqn. (19) in :cite:`Waechter2005b`.

    """

    s_phi: GtOneFloat = 2.3
    """Advanced feature. Exponent for the linear barrier function model in the
    switching rule.

    See Eqn. (19) in :cite:`Waechter2005b`.

    """

    s_theta: GtOneFloat = 1.1
    """Advanced feature. Exponent for the current constraint violation in the
    switching rule.

    See Eqn. (19) in :cite:`Waechter2005b`.

    """

    # TODO: refine type to fix the range (0,1)
    gamma_phi: PositiveFloat = 1e-8
    """Advanced feature. Relaxation factor in the filter margin for the barrier
    function.

    See Eqn. (18a) in :cite:`Waechter2005b`. The valid range is (0, 1).

    """

    # TODO: refine type to fix the range (0,1)
    gamma_theta: PositiveFloat = 1e-5
    """Advanced feature. Relaxation factor in the filter margin for the constraint
    violation.

    See Eqn. (18b) in :cite:`Waechter2005b`. The valid range is (0, 1).

    """

    # TODO: refine type to fix the range (0,1)
    alpha_min_frac: PositiveFloat = 0.05
    """Advanced feature. Safety factor for the minimal step size (before switching
    to the restoration phase).

    This is gamma_alpha in Eqn. (20) in :cite:`Waechter2005b`. The valid range is
    (0, 1).

    """

    max_soc: NonNegativeInt = 4
    """Maximum number of second order correction trial steps at each iteration.

    Choosing 0 disables the second order corrections. This is p^max of Step A-5.9
    of Algorithm A in :cite:`Waechter2005b`.

    """

    kappa_soc: PositiveFloat = 0.99
    """Advanced feature. Factor in the sufficient reduction rule for the second
    order correction.

    This option determines how much a second order correction step must reduce the
    constraint violation so that further correction steps are attempted. See Step
    A-5.9 of Algorithm A in :cite:`Waechter2005b`.

    """

    obj_max_inc: float = 5.0
    """Advanced feature. Upper bound on the acceptable increase of the barrier
    objective function.

    Trial points are rejected if they lead to an increase in the barrier objective
    function by more than this many orders of magnitude. The valid range is
    (1, inf).

    """

    max_filter_resets: NonNegativeInt = 5
    """Advanced feature. Maximal allowed number of filter resets.

    A positive number enables a heuristic that resets the filter whenever in more
    than ``filter_reset_trigger`` successive iterations the last rejected trial
    step size was rejected because of the filter. This option determines the
    maximal number of resets that are allowed to take place.

    """

    filter_reset_trigger: PositiveInt = 5
    """Advanced feature. Number of iterations that trigger the filter reset.

    If the filter reset heuristic is active and the last rejected trial step size
    was rejected because of the filter for this many successive iterations, the
    filter is reset.

    """

    corrector_type: (
        Literal[
            "none",
            "affine",
            "primal-dual",
        ]
        | None
    ) = None
    """Advanced feature. The type of corrector steps that should be taken.

    If ``mu_strategy`` is "adaptive", this option determines what kind of
    corrector steps should be tried. Possible values are ``None`` (or "none", no
    corrector), "affine" (corrector step towards mu=0) and "primal-dual"
    (corrector step towards the current mu). Changing this option is experimental.

    """

    skip_corr_if_neg_curv: YesNoBool = True
    """Advanced feature. Whether to skip the corrector step in negative curvature
    iterations.

    The corrector step is not tried if negative curvature has been encountered
    during the computation of the search direction in the current iteration. Only
    used if ``mu_strategy`` is "adaptive". Changing this option is experimental.

    """

    skip_corr_in_monotone_mode: YesNoBool = True
    """Advanced feature. Whether to skip the corrector step during the monotone
    barrier parameter mode.

    The corrector step is not tried if the algorithm is currently in the monotone
    mode. Only used if ``mu_strategy`` is "adaptive". Changing this option is
    experimental.

    """

    corrector_compl_avrg_red_fact: PositiveFloat = 1
    """Advanced feature. Complementarity tolerance factor for accepting a
    corrector step.

    This option determines the factor by which the complementarity is allowed to
    increase for a corrector step to be accepted. Changing this option is
    experimental.

    """

    soc_method: Literal[0, 1] = 0
    """Ways to apply the second order correction.

    0 is the method described in :cite:`Waechter2005b`; 1 is a modified way which
    adds alpha on the right hand side of the x and s rows.

    """

    nu_init: PositiveFloat = 1e-6
    """Advanced feature.

    Initial value of the penalty parameter in the penalty function based line
    search.

    """

    nu_inc: PositiveFloat = 0.0001
    """Advanced feature.

    Increment of the penalty parameter in the penalty function based line search.

    """

    # TODO: refine type to fix the range (0,1)
    rho: PositiveFloat = 0.1
    """Advanced feature. Value in the penalty parameter update formula.

    The valid range is (0, 1).

    """

    kappa_sigma: PositiveFloat = 1e10
    """Advanced feature. Factor limiting the deviation of the dual variables from
    their primal estimates.

    If the dual variables deviate from their primal estimates, a correction is
    performed. See Eqn. (16) in :cite:`Waechter2005b`. Setting the value to less
    than 1 disables the correction.

    """

    recalc_y: YesNoBool = False
    """Whether to recalculate the equality and inequality multipliers as
    least-squares estimates.

    This asks the algorithm to recompute the multipliers whenever the current
    infeasibility is less than ``recalc_y_feas_tol``. Choosing True might be
    helpful in the quasi-Newton option, but each recalculation requires an extra
    factorization of the linear system. If a limited-memory quasi-Newton option is
    chosen, this is used by default.

    """

    recalc_y_feas_tol: PositiveFloat = 1e-6
    """Feasibility threshold for the recomputation of multipliers.

    If ``recalc_y`` is chosen and the current infeasibility is less than this
    value, the multipliers are recomputed.

    """

    slack_move: NonNegativeFloat = 1.81899 * 1e-12
    """Advanced feature. Correction size for very small slacks.

    Due to numerical issues or the lack of an interior, the slack variables might
    become very small. If a slack becomes very small compared to machine
    precision, the corresponding bound is moved slightly; this parameter
    determines how large the move should be. The default is mach_eps^(3/4). See
    also the end of Section 3.5 in :cite:`Waechter2005b` (the actual
    implementation might be somewhat different).

    """

    constraint_violation_norm_type: Literal[
        "1-norm",
        "2-norm",
        "max-norm",
    ] = "1-norm"
    """Advanced feature.

    Norm to be used for the constraint violation in the line search.

    """

    # step calculation
    mehrotra_algorithm: YesNoBool = False
    """Whether to do Mehrotra's predictor-corrector algorithm.

    If enabled, the line search is disabled and the (unglobalized) adaptive mu
    strategy is chosen with the "probing" oracle, and "corrector_type=affine" is
    used without any safeguards; you should not explicitly set any of those
    options in addition. Also, unless otherwise specified, the values of
    ``bound_push``, ``bound_frac`` and ``bound_mult_init_val`` are set more
    aggressively, and "alpha_for_y=bound-mult" is used. Mehrotra's
    predictor-corrector algorithm usually works very well for LPs and convex QPs.

    """

    fast_step_computation: YesNoBool = False
    """Whether the linear system should be solved quickly.

    If enabled, the algorithm assumes that the linear system that is solved to
    obtain the search direction is solved sufficiently well. In that case, no
    residuals are computed to verify the solution and the computation of the
    search direction is a little faster.

    """

    min_refinement_steps: NonNegativeInt = 1
    """Minimum number of iterative refinement steps per linear system solve.

    Iterative refinement (on the full unsymmetric system) is performed for each
    right hand side; at least this many iterative refinement steps are enforced
    per right hand side.

    """

    max_refinement_steps: NonNegativeInt = 10
    """Maximum number of iterative refinement steps per linear system solve.

    Iterative refinement (on the full unsymmetric system) is performed for each
    right hand side.

    """

    residual_ratio_max: PositiveFloat = 1e-10
    """Advanced feature. Iterative refinement tolerance.

    Iterative refinement is performed until the residual test ratio is less than
    this tolerance (or until ``max_refinement_steps`` refinement steps are
    performed).

    """

    residual_ratio_singular: PositiveFloat = 1e-5
    """Advanced feature. Threshold for declaring the linear system singular after
    failed iterative refinement.

    If the residual test ratio is larger than this value after failed iterative
    refinement, the algorithm pretends that the linear system is singular.

    """

    residual_improvement_factor: PositiveFloat = 1
    """Advanced feature. Minimal required reduction of the residual test ratio in
    iterative refinement.

    If the improvement of the residual test ratio made by one iterative refinement
    step is not better than this factor, iterative refinement is aborted.

    """

    neg_curv_test_tol: NonNegativeFloat = 0
    """Tolerance for the heuristic to ignore wrong inertia.

    If nonzero, incorrect inertia in the augmented system is ignored and Ipopt
    tests if the direction is a direction of positive curvature. This tolerance is
    alpha_n in :cite:`Chiang2014` and it determines when the direction is
    considered to be sufficiently positive. A value in the range [1e-12, 1e-11] is
    recommended.

    """

    neg_curv_test_reg: YesNoBool = True
    """Whether to do the curvature test with the primal regularization (see
    :cite:`Chiang2014`).

    If False, the original Ipopt approach is used, in which the primal
    regularization is ignored.

    """

    max_hessian_perturbation: PositiveFloat = 1e20
    """Maximum value of the regularization parameter for handling negative
    curvature.

    To guarantee that the search directions are proper descent directions, Ipopt
    requires that the inertia of the (augmented) linear system for the step
    computation has the correct number of negative and positive eigenvalues. This
    guides the algorithm away from maximizers and makes Ipopt more likely to
    converge to first order optimal points that are minimizers. If the inertia is
    not correct, a multiple of the identity matrix is added to the Hessian of the
    Lagrangian in the augmented system. This parameter gives the maximum value of
    the regularization parameter; if a regularization of that size is not enough,
    the algorithm skips this iteration and goes to the restoration phase. This is
    delta_w^max in :cite:`Waechter2005b`.

    """

    min_hessian_perturbation: NonNegativeFloat = 1e-20
    """Smallest perturbation of the Hessian block.

    The size of the perturbation of the Hessian block is never selected smaller
    than this value, unless no perturbation is necessary. This is delta_w^min in
    :cite:`Waechter2005b`.

    """

    perturb_inc_fact_first: GtOneFloat = 100
    """Increase factor for the x-s perturbation for the very first perturbation.

    The factor by which the perturbation is increased when a trial value was not
    sufficient; this value is used for the computation of the very first
    perturbation and allows a different value for the first perturbation than
    that used for the remaining perturbations. This is bar_kappa_w^+ in
    :cite:`Waechter2005b`.

    """

    perturb_inc_fact: GtOneFloat = 8
    """Increase factor for the x-s perturbation.

    The factor by which the perturbation is increased when a trial value was not
    sufficient; this value is used for the computation of all perturbations except
    for the first. This is kappa_w^+ in :cite:`Waechter2005b`.

    """

    # TODO: refine type to fix the range (0,1)
    perturb_dec_fact: PositiveFloat = 0.333333
    """Decrease factor for the x-s perturbation.

    The factor by which the perturbation is decreased when a trial value is
    deduced from the size of the most recent successful perturbation. This is
    kappa_w^- in :cite:`Waechter2005b`. The valid range is (0, 1).

    """

    first_hessian_perturbation: PositiveFloat = 0.0001
    """Size of the first x-s perturbation tried.

    This is the first value tried for the x-s perturbation in the inertia
    correction scheme. It is delta_0 in :cite:`Waechter2005b`.

    """

    jacobian_regularization_value: NonNegativeFloat = 1e-8
    """Size of the regularization for rank-deficient constraint Jacobians.

    This is bar_delta_c in :cite:`Waechter2005b`.

    """

    jacobian_regularization_exponent: NonNegativeFloat = 0.25
    """Advanced feature. Exponent for mu in the regularization for rank-deficient
    constraint Jacobians.

    This is kappa_c in :cite:`Waechter2005b`.

    """

    perturb_always_cd: YesNoBool = False
    """Advanced feature. Activate the permanent perturbation of the constraint
    linearization.

    Enabling this option leads to using the delta_c and delta_d perturbations for
    the computation of every search direction. Usually, they are only used when
    the iteration matrix is singular.

    """

    # restoration phase
    expect_infeasible_problem: YesNoBool = False
    """Enable heuristics to quickly detect an infeasible problem.

    This option activates heuristics that may speed up the infeasibility
    determination if you expect that there is a good chance for the problem to be
    infeasible. In the filter line search procedure, the restoration phase is
    called more quickly than usual, and more reduction in the constraint violation
    is enforced before the restoration phase is left. If the problem is square,
    this option is enabled automatically.

    """

    expect_infeasible_problem_ctol: NonNegativeFloat = 0.001
    """Threshold for disabling the ``expect_infeasible_problem`` option.

    If the constraint violation becomes smaller than this threshold, the
    ``expect_infeasible_problem`` heuristics in the filter line search are
    disabled. If the problem is square, this option is set to 0.

    """

    expect_infeasible_problem_ytol: PositiveFloat = 1e8
    """Multiplier threshold for activating the ``expect_infeasible_problem``
    option.

    If the max-norm of the constraint multipliers becomes larger than this value
    and ``expect_infeasible_problem`` is chosen, then the restoration phase is
    entered.

    """

    start_with_resto: YesNoBool = False
    """Whether to switch to the restoration phase in the first iteration.

    Setting this option to True forces the algorithm to switch to the feasibility
    restoration phase in the first iteration. If the initial point is feasible,
    the algorithm will abort with a failure.

    """

    soft_resto_pderror_reduction_factor: NonNegativeFloat = 0.9999
    """Required reduction in the primal-dual error in the soft restoration phase.

    The soft restoration phase attempts to reduce the primal-dual error with
    regular steps. If the damped primal-dual step (damped only to satisfy the
    fraction-to-the-boundary rule) is not decreasing the primal-dual error by at
    least this factor, then the regular restoration phase is called. Choosing 0
    here disables the soft restoration phase.

    """

    max_soft_resto_iters: NonNegativeInt = 10
    """Advanced feature. Maximum number of iterations performed successively in
    the soft restoration phase.

    If the soft restoration phase is performed for more than this many iterations
    in a row, the regular restoration phase is called.

    """

    # TODO: refine type to fix the range [0,1)
    required_infeasibility_reduction: NonNegativeFloat = 0.9
    """Required reduction of the infeasibility before leaving the restoration
    phase.

    The restoration phase algorithm is performed until a point is found that is
    acceptable to the filter and the infeasibility has been reduced by at least
    the fraction given by this option. The valid range is [0, 1).

    """

    max_resto_iter: NonNegativeInt = 3_000_000
    """Advanced feature. Maximum number of successive iterations in the
    restoration phase.

    The algorithm terminates with an error message if the number of iterations
    successively taken in the restoration phase exceeds this number.

    """

    evaluate_orig_obj_at_resto_trial: YesNoBool = True
    """Whether the original objective function should be evaluated at restoration
    phase trial points.

    Enabling this option makes the restoration phase algorithm evaluate the
    objective function of the original problem at every trial point encountered
    during the restoration phase, even if this value is not required. This
    guarantees that the original objective function can be evaluated without error
    at all accepted iterates; otherwise the algorithm might fail at a point where
    the restoration phase accepts an iterate that is good for the restoration
    phase problem but not the original problem. On the other hand, if the
    evaluation of the original objective is expensive, this might be costly.

    """

    resto_penalty_parameter: PositiveFloat = 1000
    """Advanced feature. Penalty parameter in the restoration phase objective
    function.

    This is the parameter rho in equation (31a) in :cite:`Waechter2005b`.

    """

    resto_proximity_weight: NonNegativeFloat = 1
    """Advanced feature. Weighting factor for the proximity term in the
    restoration phase objective.

    This determines how the parameter zeta in equation (29a) in
    :cite:`Waechter2005b` is computed. zeta here is this value times the square
    root of mu, where mu is the current barrier parameter.

    """

    bound_mult_reset_threshold: NonNegativeFloat = 1000
    """Threshold for resetting the bound multipliers after the restoration phase.

    After returning from the restoration phase, the bound multipliers are updated
    with a Newton step for complementarity, where the change in the primal
    variables during the entire restoration phase is taken to be the corresponding
    primal Newton step. However, if after the update the largest bound multiplier
    exceeds this threshold, the multipliers are all reset to 1.

    """

    constr_mult_reset_threshold: NonNegativeFloat = 0
    """Threshold for resetting the equality and inequality multipliers after the
    restoration phase.

    After returning from the restoration phase, the constraint multipliers are
    recomputed by a least-squares estimate. This option triggers when those
    least-squares estimates should be ignored.

    """

    resto_failure_feasibility_threshold: NonNegativeFloat | None = None
    """Advanced feature. Threshold for the primal infeasibility to declare failure
    of the restoration phase.

    If the restoration phase is terminated because of the "acceptable" termination
    criteria and the primal infeasibility is smaller than this value, the
    restoration phase is declared to have failed. If ``None`` (the default), it is
    set to ``100 * convergence_ftol_rel``.

    """

    # hessian approximation
    limited_memory_aug_solver: Literal[
        "sherman-morrison",
        "extended",
    ] = "sherman-morrison"
    """Advanced feature. Strategy for solving the augmented system for the
    low-rank Hessian.

    "sherman-morrison" uses the Sherman-Morrison formula; "extended" uses an
    extended augmented system.

    """

    limited_memory_max_history: NonNegativeInt = 6
    """Maximum size of the history for the limited-memory quasi-Newton Hessian
    approximation.

    This option determines the number of most recent iterations that are taken
    into account for the limited-memory quasi-Newton approximation.

    """

    limited_memory_update_type: Literal[
        "bfgs",
        "sr1",
    ] = "bfgs"
    """Quasi-Newton update formula for the limited-memory quasi-Newton
    approximation.

    "bfgs" is the BFGS update (with skipping); "sr1" is the SR1 update (not
    working well).

    """

    limited_memory_initialization: Literal[
        "scalar1",
        "scalar2",
        "scalar3",
        "scalar4",
        "constant",
    ] = "scalar1"
    """Initialization strategy for the limited-memory quasi-Newton approximation.

    Determines how the diagonal matrix B_0 as the first term in the limited-memory
    approximation should be computed. Possible values:

    - "scalar1": sigma = s^T y / s^T s
    - "scalar2": sigma = y^T y / s^T y
    - "scalar3": arithmetic average of scalar1 and scalar2
    - "scalar4": geometric average of scalar1 and scalar2
    - "constant": sigma = ``limited_memory_init_val``

    """

    limited_memory_init_val: PositiveFloat = 1
    """Value for B_0 in the low-rank update.

    The starting matrix in the low-rank update, B_0, is chosen to be this multiple
    of the identity in the first iteration (when no updates have been performed
    yet), and is constantly chosen as this value if
    ``limited_memory_initialization`` is "constant".

    """

    limited_memory_init_val_max: PositiveFloat = 1e8
    """Upper bound on the value for B_0 in the low-rank update."""

    limited_memory_init_val_min: PositiveFloat = 1e-8
    """Lower bound on the value for B_0 in the low-rank update."""

    limited_memory_max_skipping: PositiveInt = 2
    """Threshold for successive iterations where the quasi-Newton update is
    skipped.

    If the update is skipped more than this number of successive iterations, the
    quasi-Newton approximation is reset.

    """

    limited_memory_special_for_resto: YesNoBool = False
    """Whether the quasi-Newton updates should be special during the restoration
    phase.

    Until Nov 2010, Ipopt used a special update during the restoration phase, but
    it turned out that this does not work well. The new default uses the regular
    update procedure, which improves results. If for some reason you want to get
    back to the original update, set this option to True.

    """

    hessian_approximation: Literal[
        "limited-memory",
        "exact",
    ] = "limited-memory"
    """Indicates what Hessian information is to be used.

    "exact" uses second derivatives provided by the NLP; "limited-memory" performs
    a limited-memory quasi-Newton approximation of the Hessian of the Lagrangian.
    Since optimagic does not pass second derivatives to Ipopt, the default is
    "limited-memory".

    """

    hessian_approximation_space: Literal[
        "nonlinear-variables",
        "all-variables",
    ] = "nonlinear-variables"
    """Advanced feature. Indicates in which subspace the Hessian information is to
    be approximated.

    "nonlinear-variables" approximates only in the space of the nonlinear
    variables; "all-variables" approximates in the space of all variables (without
    slacks).

    """

    # linear solver
    linear_solver: Literal[
        "mumps", "ma27", "ma57", "ma77", "ma86", "ma97", "pardiso", "custom"
    ] = "mumps"
    """Linear solver used for the step computations.

    Determines which linear algebra package is used for the solution of the
    augmented linear system (for obtaining the search directions). The default in
    optimagic is "mumps" (the MUMPS package that ships with most cyipopt
    installations). The Harwell routines "ma27", "ma57", "ma77", "ma86" and "ma97"
    as well as "pardiso" are loaded from libraries at runtime and require these to
    be installed; "custom" selects a custom linear solver (expert use).

    """

    linear_solver_options: dict[str, Any] | None = None
    """Dictionary with the linear solver options, possibly including
    ``linear_system_scaling``, ``hsllib`` and ``pardisolib``.

    See the `Ipopt documentation
    <https://coin-or.github.io/Ipopt/OPTIONS.html>`_ for details. The linear
    solver options are not automatically converted to float at the moment.

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if not IS_CYIPOPT_INSTALLED:
            raise NotInstalledError(
                "The 'ipopt' algorithm requires the cyipopt package to be installed.\n"
                "You can install it with: `conda install -c conda-forge cyipopt`."
            )

        import cyipopt

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
