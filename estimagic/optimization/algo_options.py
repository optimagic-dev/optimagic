"""
The order is the following:

1. Convergence and Stopping Criteria
2. Trust Region Parameters
3. Other Numerical Algorithm Group Tuning Parameters

"""
"""
=====================================================================================
1. Stopping Criteria
=====================================================================================
"""

CONVERGENCE_RELATIVE_CRITERION_TOLERANCE = 2e-9
"""float: Stop when the relative improvement between two iterations is below this.

    The exact definition of relative improvement depends on the optimizer and should
    be documented there. To disable it, set it to 0.

    The default value is inspired by scipy L-BFGS-B defaults, but rounded.

"""

CONVERGENCE_ABSOLUTE_CRITERION_TOLERANCE = 0
"""float: Stop when the absolute improvement between two iterations is below this.

    Disabled by default because it is very problem specific.

"""

CONVERGENCE_ABSOLUTE_GRADIENT_TOLERANCE = 1e-5
"""float: Stop when the gradient are smaller than this.

    For some algorithms this criterion refers to all entries, for others to some norm.

    For bound constrained optimizers this typically refers to a projected gradient.
    The exact definition should be documented for each optimizer.

    The default is the same as scipy. To disable it, set it to zero.

"""

CONVERGENCE_RELATIVE_GRADIENT_TOLERANCE = 1e-8
"""float: Stop when the gradient, divided by the absolute value of the criterion
    function is smaller than this. For some algorithms this criterion refers to
    all entries, for others to some norm.For bound constrained optimizers this
    typically refers to a projected gradient. The exact definition should be documented
    for each optimizer. To disable it, set it to zero.

"""

CONVERGENCE_SCALED_GRADIENT_TOLERANCE = 1e-8
"""float: Stop when all entries (or for some algorithms the norm) of the gradient,
    divided by the norm of the gradient at start parameters is smaller than this.
    For bound constrained optimizers this typically refers to a projected gradient.
    The exact definition should be documented for each optimizer.
    To disable it, set it to zero.

"""

CONVERGENCE_RELATIVE_PARAMS_TOLERANCE = 1e-5
"""float: Stop when the relative change in parameters is smaller than this.
    The exact definition of relative change and whether this refers to the maximum
    change or the average change depends on the algorithm and should be documented
    there. To disable it, set it to zero. The default is the same as in scipy.

"""

CONVERGENCE_ABSOLUTE_PARAMS_TOLERANCE = 0
"""float: Stop when the absolute change in parameters between two iterations is smaller
    than this. Whether this refers to the maximum change or the average change depends
    on the algorithm and should be documented there.

    Disabled by default because it is very problem specific. To enable it, set it to a
    value larger than zero.

"""

CONVERGENCE_NOISE_CORRECTED_CRITERION_TOLERANCE = 1.0
"""float: Stop when the evaluations on the set of interpolation points all fall within
    this factor of the noise level. The default is 1, i.e. when all evaluations are
    within the noise level. If you want to not use this criterion but still flag your
    criterion function as noisy, set this tolerance to 0.0.

    .. warning::
        Very small values, as in most other tolerances don't make sense here.

"""

CONVERGENCE_MINIMAL_TRUSTREGION_RADIUS_TOLERANCE = 1e-8
"""float: Stop when the lower trust region radius falls below this value."""


STOPPING_MAX_CRITERION_EVALUATIONS = 1_000_000
"""int:
    If the maximum number of function evaluation is reached, the optimization stops
    but we do not count this as successful convergence. The function evaluations used
    to evaluate a numerical gradient do not count for this.

"""


STOPPING_MAX_CRITERION_EVALUATIONS_GLOBAL = 1_000
"""int:
    If the maximum number of function evaluation is reached, the optimization stops
    but we do not count this as successful convergence. The function evaluations used
    to evaluate a numerical gradient do not count for this. Set to a lower number than
    STOPPING_MAX_CRITERION_EVALUATIONS for global optimizers.

"""


STOPPING_MAX_ITERATIONS = 1_000_000
"""int:
    If the maximum number of iterations is reached, the
    optimization stops, but we do not count this as successful convergence.
    The difference to ``max_criterion_evaluations`` is that one iteration might
    need several criterion evaluations, for example in a line search or to determine
    if the trust region radius has to be shrunk.

"""

CONVERGENCE_SECOND_BEST_ABSOLUTE_CRITERION_TOLERANCE = 1e-08
"""float: absolute criterion tolerance estimagic requires if no other stopping
criterion apart from max iterations etc. is available
this is taken from scipy (SLSQP's value, smaller than Nelder-Mead).

"""

CONVERGENCE_SECOND_BEST_ABSOLUTE_PARAMS_TOLERANCE = 1e-08
"""float: The absolute parameter tolerance estimagic requires if no other stopping
criterion apart from max iterations etc. is available. This is taken from pybobyqa.

"""

CONVERGENCE_SLOW_PROGRESS = {
    "threshold_to_characterize_as_slow": 1e-8,
    "max_insufficient_improvements": None,
    "comparison_period": 5,
}
"""dict: Specification of when to terminate or reset the optimization because of only
    slow improvements. This is similar to an absolute criterion tolerance only that
    instead of a single improvement the average over several iterations must be small.

    Possible entries are:
        threshold_to_characterize_as_slow (float): Threshold whether an improvement
            is insufficient. Note: the improvement is divided by the
            ``comparison_period``.
            So this is the required average improvement per iteration over the
            comparison period.
        max_insufficient_improvements (int): Number of consecutive
            insufficient improvements before termination (or reset). Default is
            ``20 * len(x)``.
        comparison_period (int):
            How many iterations to go back to calculate the improvement.
            For example 5 would mean that each criterion evaluation is compared to the
            criterion value from 5 iterations before.

"""

"""
=====================================================================================
2. Other Common Tuning Parameters for Optimization Algorithms
=====================================================================================
"""

MAX_LINE_SEARCH_STEPS = 20
"""int: Inspired by scipy L-BFGS-B."""

LIMITED_MEMORY_STORAGE_LENGTH = 10
"""int: Taken from scipy L-BFGS-B."""

THRESHOLD_FOR_SAFETY_STEP = 0.5
r"""float: Threshold for when to call the safety step (:math:`\gamma_s`).

    :math:`\text{proposed step} \leq \text{threshold_for_safety_step} \cdot
    \text{current_lower_trustregion_radius}`.

"""

"""
-------------------------
Trust Region Parameters
-------------------------
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
"""float: Ratio by which to shrink the upper trust region radius when realized
    improvement does not match the ``threshold_successful``. The default is 0.98
    if the criterion is noisy and 0.5 else.

"""

TRUSTREGION_EXPANSION_FACTOR_SUCCESSFUL = 2.0
r"""float: Ratio by which to expand the upper trust region radius :math:`\Delta_k`
    in very successful iterations (:math:`\gamma_{inc}` in the notation of the paper).

"""

TRUSTREGION_EXPANSION_FACTOR_VERY_SUCCESSFUL = 4.0
r"""float: Ratio of the proposed step ($\|s_k\|$) by which to expand the upper trust
    region radius (:math:`\Delta_k`) in very successful iterations
    (:math:`\overline{\gamma}_{inc}` in the notation of the paper).

"""

TRUSTREGION_SHRINKING_FACTOR_LOWER_RADIUS = None
r"""float: Ratio by which to shrink the lower trust region radius (:math:`\rho_k`)
    (:math:`\alpha_1` in the notation of the paper). Default is 0.9 if
    the criterion is noisy and 0.1 else.

"""

TRUSTREGION_SHRINKING_FACTOR_UPPER_RADIUS = None
r"""float: Ratio of the current lower trust region (:math:`\rho_k`) by which to shrink
    the upper trust region radius (:math:`\Delta_k`) when the lower one is shrunk
    (:math:`\alpha_2` in the notation of the paper). Default is 0.95 if the
    criterion is noisy and 0.5 else."""

"""
---------------------------------------------
Numerical Algorithm Group Tuning Parameters
---------------------------------------------
"""

INITIAL_DIRECTIONS = "coordinate"
"""string: How to draw the initial directions. Possible values are "coordinate" for
    coordinate directions (the default) or "random".

"""

RANDOM_DIRECTIONS_ORTHOGONAL = True
"""bool: Whether to make randomly drawn initial directions orthogonal."""


INTERPOLATION_ROUNDING_ERROR = 0.1
r"""float: Internally, all the NAG algorithms store interpolation points with respect
    to a base point :math:`x_b`; that is, we store :math:`\{y_t-x_b\}`,
    which reduces the risk of roundoff errors. We shift :math:`x_b` to :math:`x_k` when
    :math:`\text{proposed step} \leq \text{interpolation_rounding_error} \cdot
    \|x_k-x_b\|`.

"""

CLIP_CRITERION_IF_OVERFLOWING = True
"""bool: Whether to clip the criterion to avoid ``OverflowError``."""


TRUSTREGION_PRECONDITION_INTERPOLATION = True
"""bool: whether to scale the interpolation linear system to improve conditioning."""


RESET_OPTIONS = {
    "use_resets": None,
    "minimal_trustregion_radius_tolerance_scaling_at_reset": 1.0,
    "reset_type": "soft",
    "move_center_at_soft_reset": True,
    "reuse_criterion_value_at_hard_reset": True,
    "max_iterations_without_new_best_after_soft_reset": None,
    "auto_detect": True,
    "auto_detect_history": 30,
    "auto_detect_min_jacobian_increase": 0.015,
    "auto_detect_min_correlations": 0.1,
    "points_to_replace_at_soft_reset": 3,
    "max_consecutive_unsuccessful_resets": 10,
    # just bobyqa
    "max_unsuccessful_resets": None,
    "trust_region_scaling_at_unsuccessful_reset": None,
    # just dfols
    "max_interpolation_points": None,
    "n_extra_interpolation_points_per_soft_reset": 0,
    "n_extra_interpolation_points_per_hard_reset": 0,
    "n_additional_extra_points_to_replace_per_reset": 0,
}
r"""dict: Options for reseting the optimization.

    Possible entries are:

        use_resets (bool): Whether to do resets when the lower trust
            region radius (:math:`\rho_k`) reaches the stopping criterion
            (:math:`\rho_{end}`), or (optionally) when all interpolation points are
            within noise level. Default is ``True`` if the criterion is noisy.
        minimal_trustregion_radius_tolerance_scaling_at_reset (float): Factor with
            which the trust region stopping criterion is multiplied at each reset.

        reset_type (str): Whether to use "soft" or "hard" resets. Default is "soft".

        move_center_at_soft_reset (bool): Whether to move the trust region center
            ($x_k$) to the best new point evaluated in stead of keeping it constant.
        points_to_replace_at_soft_reset (int): Number of interpolation points to move
            at each soft reset.
        reuse_criterion_value_at_hard_reset (bool): Whether or not to recycle the
            criterion value at the best iterate found when performing a hard reset.
            This saves one criterion evaluation.
        max_iterations_without_new_best_after_soft_reset (int):
            The maximum number of successful steps in a given run where the new
            criterion value is worse than the best value found in previous runs before
            terminating. Default is ``max_criterion_evaluations``.
        auto_detect (bool): Whether or not to
            automatically determine when to reset. This is an additional condition
            and resets can still be triggered by small upper trust region radius, etc.
            There are two criteria used: upper trust region radius shrinkage
            (no increases over the history, more decreases than no changes) and
            changes in the model Jacobian (consistently increasing trend as measured
            by slope and correlation coefficient of the line of best fit).
        auto_detect_history (int):
            How many iterations of model changes and trust region radii to store.
        auto_detect_min_jacobian_increase (float):
            Minimum rate of increase of the Jacobian over past iterations to cause a
            reset.
        auto_detect_min_correlations (float):
            Minimum correlation of the Jacobian data set required to cause a reset.
        max_consecutive_unsuccessful_resets (int): maximum number of consecutive
            unsuccessful resets allowed (i.e. resets which did not outperform the
            best known value from earlier runs).

    Only used when using nag_bobyqa:

        max_unsuccessful_resets (int):
            number of total unsuccessful resets allowed.
            Default is 20 if ``seek_global_optimum`` and else unrestricted.
        trust_region_scaling_at_unsuccessful_reset (float): Factor by which to
            expand the initial lower trust region radius (:math:`\rho_{beg}`) after
            unsuccessful resets. Default is 1.1 if ``seek_global_optimum`` else 1.

    Only used when using nag_dfols:

        max_interpolation_points (int): Maximum allowed value of the number of
            interpolation points. This is useful if the number of interpolation points
            increases with each reset, e.g. when
            ``n_extra_interpolation_points_per_soft_reset > 0``. The default is
            ``n_interpolation_points``.
        n_extra_interpolation_points_per_soft_reset (int): Number of points to add to
            the interpolation set with each soft reset.
        n_extra_interpolation_points_per_hard_reset (int): Number of points to add to
            the interpolation set with each hard reset.
        n_additional_extra_points_to_replace_per_reset (int): This parameter modifies
            ``n_extra_points_to_replace_successful``. With each reset
            ``n_extra_points_to_replace_successful`` is increased by this number.

"""


TRUSTREGION_FAST_START_OPTIONS = {
    "min_inital_points": None,
    "method": "auto",
    "scale_of_trustregion_step_perturbation": None,
    "scale_of_jacobian_components_perturbation": 1e-2,
    # the following will be growing.full_rank.min_sing_val
    # but it not supported yet by DF-OLS.
    "floor_of_jacobian_singular_values": 1,
    "jacobian_max_condition_number": 1e8,
    "geometry_improving_steps": False,
    "safety_steps": True,
    "shrink_upper_radius_in_safety_steps": False,
    "full_geometry_improving_step": False,
    "reset_trustregion_radius_after_fast_start": False,
    "reset_min_trustregion_radius_after_fast_start": False,
    "shrinking_factor_not_successful": None,
    "n_extra_search_directions_per_iteration": 0,
}
r"""dict: Options to start the optimization while building the full trust region model.

    To activate this, set the number of interpolation points at which to evaluate the
    criterion before doing the first step, `min_initial_points`, to something smaller
    than the number of parameters.

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
            can start with less than ``len(x)`` interpolation points and add points
            to the trust region model with every iteration.
        method ("jacobian", "trustregion" or "auto"):
            When there are less interpolation points than ``len(x)`` the model is
            underdetermined. This can be fixed in two ways:
            If "jacobian", the interpolated Jacobian is perturbed to have full
            rank, allowing the trust region step to include components in the full
            search space. This is the default if
            ``len(x) \geq number of root contributions``.
            If "trustregion_step", the trust region step is perturbed by an
            orthogonal direction not yet searched. It is the default if
            ``len(x) < number of root contributions``.
        scale_of_trustregion_step_perturbation (float):
            When adding new search directions, the length of the step is the trust
            region radius multiplied by this value. The default is 0.1 if
            ``method == "trustregion"`` else 1.
        scale_of_jacobian_components_perturbation (float): Magnitude of extra
            components added to the Jacobian. Default is 1e-2.
        floor_of_jacobian_singular_values (float): Floor singular
            values of the Jacobian at this factor of the last non zero value.
            As of version 1.2.1 this option is not yet supported by DF-OLS!
        scale_of_jacobian_singular_value_floor (float):
            Floor singular values of the Jacobian at this factor of the last nonzero
            value.
        jacobian_max_condition_number (float): Cap on the condition number
            of Jacobian after applying floors to singular values
            (effectively another floor on the smallest singular value, since the
            largest singular value is fixed).
        geometry_improving_steps (bool): Whether to do geometry-improving steps in the
            trust region algorithm, as per the usual algorithm during the fast start.
        safety_steps (bool):
            Whether to perform safety steps.
        shrink_upper_radius_in_safety_steps (bool): During the fast start whether to
            shrink the upper trust region radius in safety steps.
        full_geometry_improving_step (bool): During the fast start whether to do a
            full geometry-improving step within safety steps (the same as the post fast
            start phase of the algorithm). Since this involves reducing the upper trust
            region radius, this can only be `True` if
            `shrink_upper_radius_in_safety_steps == False`.
        reset_trustregion_radius_after_fast_start (bool):
            Whether or not to reset the upper trust region radius to its initial value
            at the end of the fast start phase.
        reset_min_trustregion_radius_after_fast_start (bool):
            Whether or not to reset the minimum trust region radius
            (:math:`\rho_k`) to its initial value at the end of the fast start phase.
        shrinking_factor_not_successful (float):
            Ratio by which to shrink the trust region radius when realized
            improvement does not match the ``threshold_for_successful_iteration``
            during the fast start phase.  By default it is the same as
            ``reduction_when_not_successful``.
        n_extra_search_directions_per_iteration (int): Number of new search
            directions to add with each iteration where we do not have a full set
            of search directions. This approach is not recommended! Default is 0.

"""
