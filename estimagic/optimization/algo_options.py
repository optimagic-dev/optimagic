"""
=====================================================================================
Stopping Criteria
=====================================================================================
"""

RELATIVE_CRITERION_TOLERANCE = 2e-9
"""float: Stop when the relative improvement between two iterations is below this.

    The exact definition of relative improvement depends on the optimizer and should
    be documented there. To disable it, set it to 0.

    The default value is inspired by scipy L-BFGS-B defaults, but rounded.

"""

ABSOLUTE_CRITERION_TOLERANCE = 0
"""float: Stop when the absolute improvement between two iterations is below this.

    Disabled by default because it is very problem specific.

"""

ABSOLUTE_GRADIENT_TOLERANCE = 1e-5
"""float: Stop when the gradient are smaller than this.

    For some algorithms this criterion refers to all entries, for others to some norm.

    For bound constrained optimizers this typically refers to a projected gradient.
    The exact definition should be documented for each optimizer.

    The default is the same as scipy. To disable it, set it to zero.

"""

RELATIVE_GRADIENT_TOLERANCE = 1e-8
"""float: Stop when the gradient, divided by the absolute value of the criterion
    function is smaller than this. For some algorithms this criterion refers to
    all entries, for others to some norm.For bound constrained optimizers this
    typically refers to a projected gradient. The exact definition should be documented
    for each optimizer. To disable it, set it to zero.

"""

SCALED_GRADIENT_TOLERANCE = 1e-8
"""float: Stop when all entries (or for some algorithms the norm) of the gradient,
    divided by the norm of the gradient at start parameters is smaller than this.
    For bound constrained optimizers this typically refers to a projected gradient.
    The exact definition should be documented for each optimizer.
    To disable it, set it to zero.

"""

RELATIVE_PARAMS_TOLERANCE = 1e-5
"""float: Stop when the relative change in parameters is smaller than this.
    The exact definition of relative change and whether this refers to the maximum
    change or the average change depends on the algorithm and should be documented
    there. To disable it, set it to zero. The default is the same as in scipy.

"""

ABSOLUTE_PARAMS_TOLERANCE = 0
"""float: Stop when the absolute change in parameters between two iterations is smaller
    than this. Whether this refers to the maximum change or the average change depends
    on the algorithm and should be documented there.

    Disabled by default because it is very problem specific. To enable it, set it to a
    value larger than zero.

"""

NOISE_CORRECTED_CRITERION_TOLERANCE = 1.0
"""float: Stop when the evaluations on the set of interpolation points all fall within
    this factor of the noise level. The default is 1, i.e. when all evaluations are
    within the noise level. If you want to not use this criterion but still flag your
    criterion function as noisy, set this tolerance to 0.0.

    .. warning::
        Very small values, as in most other tolerances don't make sense here.

"""


MAX_CRITERION_EVALUATIONS = 1_000_000
"""int:
    If the maximum number of function evaluation is reached, the optimization stops
    but we do not count this as successful convergence. The function evaluations used
    to evaluate a numerical gradient do not count for this.

"""

MAX_ITERATIONS = 1_000_000
"""int:
    If the maximum number of iterations is reached, the
    optimization stops, but we do not count this as successful convergence.
    The difference to ``max_criterion_evaluations`` is that one iteration might
    need several criterion evaluations, for example in a line search or to determine
    if the trust region radius has to be decreased.

"""

SECOND_BEST_ABSOLUTE_CRITERION_TOLERANCE = 1e-08
"""float: absolute criterion tolerance estimagic requires if no other stopping
criterion apart from max iterations etc. is available
this is taken from scipy (SLSQP's value, smaller than Nelder-Mead).

"""

SECOND_BEST_ABSOLUTE_PARAMS_TOLERANCE = 1e-08
"""float: The absolute parameter tolerance estimagic requires if no other stopping
criterion apart from max iterations etc. is available. This is taken from pybobyqa.

"""

SLOW_IMPROVEMENT_TOLERANCE = {
    "threshold_for_insufficient_improvement": 1e-8,
    "n_insufficient_improvements_until_terminate": None,
    "comparison_period_for_insufficient_improvement": 5,
}
"""dict: Specification of when to terminate or restart the optimization because of only
    slow improvements. This is similar to an absolute criterion tolerance only that
    instead of a single improvement the average over several iterations must be small.

    Possible entries are:
        threshold_for_insufficient_improvement (float): Threshold whether an improvement
            is insufficient. Note: the improvement is divided by the
            ``comparison_period_for_insufficient_improvement``.
            So this is the required average improvement per iteration over the
            comparison period.
        n_insufficient_improvements_until_terminate (int): Number of consecutive
            insufficient improvements before termination (or restart). Default is
            ``20 * len(x)``.
        comparison_period_for_insufficient_improvement (int):
            How many iterations to go back to calculate the improvement.
            For example 5 would mean that each criterion evaluation is compared to the
            criterion value from 5 iterations before.

"""

"""
=====================================================================================
Other Common Tuning Parameters for Optimization Algorithms
=====================================================================================
"""

MAX_LINE_SEARCH_STEPS = 20
"""int: Inspired by scipy L-BFGS-B."""

LIMITED_MEMORY_STORAGE_LENGTH = 10
"""int: Taken from scipy L-BFGS-B."""

"""
-------------------------
Trust Region Parameters
-------------------------
"""

THRESHOLD_FOR_SAFETY_STEP = 0.5
r"""float: Threshold for when to call the safety step.

    :math:`\text{proposed step} \leq \text{threshold_for_safety_step} \cdot
    \text{current_trust_region_radius}`.

"""

TRUSTREGION_THRESHOLD_SUCCESSFUL = 0.1
"""float: Share of the predicted improvement that has to be achieved for a trust
    region iteration to count as successful.

"""

TRUSTREGION_THRESHOLD_VERY_SUCCESSFUL = 0.7
"""float: Share of predicted improvement that has to be achieved for a trust region
    iteration to count as very successful.``criterion_noisy``

"""

TRUSTREGION_SHRINKING_FACTOR_NOT_SUCCESSFUL = None
"""float: Ratio by which to decrease the trust region radius when realized improvement
    does not match the ``threshold_successful``. The default is 0.98 if the criterion
    is noisy and 0.5 else.

"""

TRUSTREGION_EXPANSION_FACTOR_SUCCESSFUL = 2.0
r"""float: Ratio by which to increase the upper trust region radius :math:`\Delta_k`
    in very successful iterations (:math:`\gamma_{inc}` in the notation of the paper).

"""

TRUSTREGION_EXPANSION_FACTOR_VERY_SUCCESSFUL = 4.0
r"""float: Ratio of the proposed step ($\|s_k\|$) by which to increase the upper trust
    region radius (:math:`\Delta_k`) in very successful iterations
    (:math:`\overline{\gamma}_{inc}` in the notation of the paper).

"""

TRUSTREGION_SHRINKING_FACTOR_LOWER_RADIUS = None
r"""float: Ratio by which to shrink the lower trust region radius (:math:`\rho_k`)
    (:math:`\alpha_1` in the notation of the paper). Default is 0.9 if
    the criterion is noisy and 0.1 else.

"""

TRUSTREGION_UPDATE_FROM_MIN_TRUST_REGION = None
r"""float: Ratio of the current lower trust region (:math:`\rho_k`) by which to shrink
    the upper trust region radius (:math:`\Delta_k`) when the lower one is reduced
    (:math:`\alpha_2` in the notation of the paper). Default is 0.95 if the
    criterion is noisy and 0.5 else."""

"""
---------------------------------------------
Numerical Algorithm Group Tuning Parameters
---------------------------------------------
"""

RANDOM_INITIAL_DIRECTIONS = False
"""bool: Whether to draw the initial directions randomly.

If `False` use the coordinate directions.

"""

RANDOM_DIRECTIONS_ORTHOGONAL = True
"""bool: Whether to make randomly drawn initial directions orthogonal."""


INTERPOLATION_ROUNDING_ERROR = 0.1
r"""float:

    Internally, all the NAG algorithms store interpolation points with respect
    to a base point :math:`x_b`; that is, we store :math:`\{y_t-x_b\}`,
    which reduces the risk of roundoff errors. We shift :math:`x_b` to :math:`x_k` when
    :math:`\text{proposed step} \leq \text{interpolation_rounding_error} \cdot
    \|x_k-x_b\|`.

"""

CLIP_CRITERION_IF_OVERFLOWING = True
"""bool: Whether to clip the criterion to avoid ``OverflowError``."""


TRUSTREGION_PRECONDITION_INTERPOLATION = True
"""bool: whether to scale the interpolation linear system to improve conditioning."""


RESTART_OPTIONS = {
    "use_restarts": None,
    "max_unsuccessful": 10,
    "min_trust_region_scaling_after": 1.0,
    "use_soft": True,
    "move_current_point_at_soft": True,
    "reuse_criterion_value_at_hard": True,
    "max_iterations_without_new_best_after_soft": None,
    "automatic_detection": True,
    "n_iterations_for_automatc_detection": 30,
    "min_model_slope_increase_for_automatic_detection": 0.015,
    "min_correlations_for_automatic_detection": 0.1,
    "points_to_move_at_soft": 3,
    # just bobyqa
    "max_unsuccessful_total": None,
    "trust_region_scaling_after_unsuccessful": None,
    # just dfols
    "max_interpolation_points": None,
    "n_extra_interpolation_points_per_soft_reset": 0,
    "n_extra_interpolation_points_per_hard_reset": 0,
}
r"""dict: Options for restarting the optimization.

    Possible entries are:

        use_restarts (bool): Whether to do restarts when the minimum trust
            region radius (:math:`\rho_k`) reaches the stopping criterion
            (:math:`\rho_{end}`), or (optionally) when all points are within noise
            level. Default is ``True`` if the criterion is noisy.
        max_unsuccessful (int): maximum number of consecutive unsuccessful
            restarts allowed (i.e. restarts which did not outperform the best known
            value from earlier runs).
        min_trust_region_scaling_after (float): Factor with which the trust region
            stopping criterion is multiplied at each restart.

        use_soft (bool): Whether to use soft or hard restarts.

        move_current_point_at_soft (bool): Whether to move the current
            evaluation point ($x_k$) to the best new point evaluated.
        points_to_move_at_soft (int): Number of interpolation points to move
            at each soft restart.
        reuse_criterion_value_at_hard (bool): Whether or not to recycle the
            criterion value at the best iterate found when performing a hard restart.
            This saves one criterion evaluation.
        max_iterations_without_new_best_after_soft (int):
            The maximum number of successful steps in a given run where the new
            criterion value is worse than the best value found in previous runs before
            terminating. Default is ``max_criterion_evaluations``.
        automatic_detection (bool): Whether or not to
            automatically determine when to restart. This is an additional condition
            and restarts can still be triggered by small trust region radius, etc.
            There are two criteria used: trust region radius decreases
            (no increases over the history, more decreases than no changes) and
            changes in the model Jacobian (consistently increasing trend as measured
            by slope and correlation coefficient of the line of best fit).
        n_iterations_for_automatc_detection (int):
            How many iterations of model changes and trust region radii to store.
        min_model_slope_increase_for_automatic_detection (float):
            Minimum rate of increase of the Jacobian over past iterations to cause a
            restart.
        min_correlations_for_automatic_detection (float):
            Minimum correlation of the Jacobian data set required to cause a restart.

    Only used when using nag_bobyqa:
        max_unsuccessful_total (int): number of total unsuccessful restarts
            allowed. Default is 20 if ``seek_global_optimum`` and else unrestricted.
        trust_region_scaling_after_unsuccessful (float): Factor by which to
            increase the initial trust region radius (:math:`\rho_{beg}`) after
            unsuccessful restarts. Default is 1.1 if ``seek_global_optimum`` else 1.

    Only used when usinge nag_dfols:
        max_interpolation_points (int): Maximum allowed value of the number of
            interpolation points. This is useful if the number of interpolation points
            increases with each restart, e.g. when
            ``n_interpolation_points_to_add_at_restart > 0``. The default is
            ``n_interpolation_points``.
        n_extra_interpolation_points_per_soft_reset (int): Number of points to add to
            the interpolation set with each soft reset.
        n_extra_interpolation_points_per_hard_reset (int): Number of points to add to
            the interpolation set with each hard reset.
"""


FAST_START_OPTIONS = {
    "min_inital_points": None,
    "strategy": "auto",
    "scaling_of_trust_region_step_perturbation": None,
    "scaling_jacobian_perturb_components": 1e-2,
    "scaling_jacobian_perturb_floor_of_singular_values": 1,  # not supported yet by NAG
    "jacobian_perturb_abs_floor_for_singular_values": 1e-6,
    "jacobian_perturb_max_condition_number": 1e8,
    "geometry_improving_steps": False,
    "safety_steps": True,
    "shrink_upper_radius_in_safety_steps": False,
    "full_geometry_improving_step": False,
    "reset_trust_region_radius_after": False,
    "reset_min_trust_region_radius_after": False,
    "trust_region_decrease": None,
    "n_search_directions_to_add_when_incomplete": 0,
}
r"""dict: Options to start the optimization while building the full trust region model.

    To activate this, set the number of points at which to evaluate the criterion
    before doing the first step, `min_initial_points`, to something smaller than the
    number of parameters.

    The following options can be specified:

        min_initial_points (int): Number of initial interpolation
            points in addition to the start point. This should only be changed to
            a value less than ``len(x)``, and only if the default setup cost
            of ``len(x) + 1`` evaluations of the criterion is impractical.
            If this is set to be less than the default, the input value of
            ``n_interpolation_points`` should be set to ``len(x)``.
            If the default is used, all the other parameters have no effect.
            Default is ``n_interpolation_points - 1``.
            If the default setup costs of the evaluations are very large, DF-OLS
            can start with less than ``len(x)`` points and add points to the trust
            region model with every iteration.
        strategy ("jacobian", "trust_region" or "auto"):
            When there are less interpolation points than ``len(x)`` the model is
            underdetermined. This can be fixed in two ways:
            If "jacobian", the interpolated Jacobian is perturbed to have full
            rank, allowing the trust region step to include components in the full
            search space. This is the default if
            ``len(x) \geq number of root contributions``.
            If "trust_region_step", the trust region step is perturbed by an
            orthogonal direction not yet searched. It is the default if
            ``len(x) < number of root contributions``.
        scaling_of_trust_region_step_perturbation (float):
            When adding new search directions, the length of the step is the trust
            region radius multiplied by this value. The default is 0.1 if
            ``fast_start_strategy == "perturb_trust_region_radius"`` else 1.
        scaling_jacobian_perturb_components (float): Magnitude of extra components
            added to the Jacobian. Default is 1e-2.
        scaling_jacobian_perturb_floor_of_singular_values (float): Floor singular
            values of the Jacobian at this factor of the last non zero value.
            As of version 1.2.1 this option is not yet supported by DF-OLS.
        jacobian_perturb_abs_floor_for_singular_values (float): Absolute floor on
            singular values of the Jacobian. Default is 1e-6.
        jacobian_perturb_max_condition_number (float): Cap on the condition number
            of Jacobian after applying floors to singular values
            (effectively another floor on the smallest singular value, since the
            largest singular value is fixed).
        geometry_improving_steps (bool): Whether to do geometry-improving steps in the
            trust region algorithm, as per the usual algorithm during the fast start.
        safety_steps (bool): Whether to perform safety steps.
        shrink_upper_radius_in_safety_steps (bool): During the fast start whether to
            reduce the upper trust region radius in safety steps.
        full_geometry_improving_step (bool): During the fast start whether to do a
            full geometry-improving step within safety steps (the same as the post fast
            start phase of the algorithm). Since this involves reducing the upper trust
            region radius, this can only be `True` if
            `shrink_upper_radius_in_safety_steps == False`.
        reset_trust_region_radius_after (bool):
            Whether or not to reset the trust region radius to its initial value
            at the end of the growing phase.
        reset_min_trust_region_radius_after (bool):
            Whether or not to reset the minimum trust region radius
            (:math:`\rho_k`) to its initial value at the end of the growing phase.
        trust_region_decrease (float):
            Ratio by which to decrease the trust region radius when realized
            improvement does not match the ``threshold_for_successful_iteration``
            during the growing phase.  By default it is the same as
            ``reduction_when_not_successful``.
        n_search_directions_to_add_when_incomplete (int): Number of new search
            directions to add with each iteration where we do not have a full set
            of search directions. This approach is not recommended! Default is 0.

"""
