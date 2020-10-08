"""
=====================================================================================
Stopping Criteria
=====================================================================================
"""

RELATIVE_CRITERION_TOLERANCE = 2e-9
"""float: Inspired by scipy L-BFGS-B defaults, but rounded."""

ABSOLUTE_CRITERION_TOLERANCE = 0
"""float: Disabled by default because it is very problem specific."""

ABSOLUTE_GRADIENT_TOLERANCE = 1e-5
"""float: Same as scipy."""

RELATIVE_GRADIENT_TOLERANCE = 1e-8

SCALED_GRADIENT_TOLERANCE = 1e-8

RELATIVE_PARAMS_TOLERANCE = 1e-5
"""float: Same as scipy."""

ABSOLUTE_PARAMS_TOLERANCE = 0
"""float: Disabled by default because it is very problem specific."""

MAX_CRITERION_EVALUATIONS = 1_000_000
MAX_ITERATIONS = 1_000_000

SECOND_BEST_ABSOLUTE_CRITERION_TOLERANCE = 1e-08
"""float: absolute criterion tolerance estimagic requires if no other stopping
criterion apart from max iterations etc. is available
this is taken from scipy (SLSQP's value, smaller than Nelder-Mead).

"""

SECOND_BEST_ABSOLUTE_PARAMS_TOLERANCE = 1e-08
"""float: The absolute parameter tolerance estimagic requires if no other stopping
criterion apart from max iterations etc. is available. This is taken from pybobyqa.

"""

COMPARISON_PERIOD_FOR_INSUFFICIENT_IMPROVEMENT = 5
"""int: How many iterations to go back to calculate the improvement.

For example 5 would mean that each criterion evaluation is compared to the
criterion value from 5 iterations before.

"""


SLOW_IMPROVEMENT_TOLERANCE = {
    "threshold_for_insufficient_improvement": 1e-8,
    "n_insufficient_improvements_until_terminate": None,
    "comparison_period_for_insufficient_improvement": COMPARISON_PERIOD_FOR_INSUFFICIENT_IMPROVEMENT,  # noqa E501
}
"""dict: Specification of when to terminate or restart the optimization because of only
    slow improvements. This is similar to an absolute criterion tolerance only that
    instead of a single improvement the average over several iterations must be small.

    Possible entries are:
        threshold_for_insufficient_improvement (float): Threshold whether an improvement
            is insufficient. Note: the improvement is divided by the
            ``comparison_period_for_insufficient_improvement``.
            So this is the required average improvement per iteration over the
            comparison period. Default is 1e-8.
        n_insufficient_improvements_until_terminate (int): Number of consecutive
            insufficient improvements before termination (or restart). Default is
            ``20 * len(x)``.
        comparison_period_for_insufficient_improvement (int):
            How many iterations to go back to calculate the improvement.
            For example 5 would mean that each criterion evaluation is compared to the
            criterion value from 5 iterations before.

"""


CONVERGENCE_NOISE_CRITERION = {
    "noise_scale_factor_for_quit": 1.0,
    "active": None,
    "multiplicative_noise_level": None,
    "additive_noise_level": None,
}
"""dict: Arguments for converging when the evaluations in the trust region all fall
    within a scaled version of the noise at the point of interest. Entries are:

    active (bool): Flag to quit (or restart) if
        all criterion evaluations of the trust region are within the some scaled
        version of the noise level at the point of interest.
        Default is ``False`` for smooth problems or ``True`` for noisy problems.
        The remaining arguments determine this scaling.
    noise_scale_factor_for_quit (float): Factor of the noise level to use in
        termination criterion.
    multiplicative_noise_level (float): Multiplicative noise level in :math:`f`.
        Can only specify one of multiplicative or additive noise levels.
        Default is :code:`None`.
    additive_noise_level (float): Additive noise level in :math:`f`.
        Can only specify one of multiplicative or additive noise levels.
        Default is :code:`None`.

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

THRESHOLD_FOR_SUCCESSFUL_ITERATION = 0.1
"""float: minimum share of predicted improvement to count an iteration as successful.

"""

THRESHOLD_FOR_VERY_SUCCESFUL_ITERATION = 0.7
"""float: share of predicted improvement that has to be surpassed for an iteration to
count as very successful.

"""

THRESHOLD_FOR_SAFETY_STEP = 0.5
r"""float: Threshold for when to call the safety step.

    :math:`\text{proposed step} \leq \text{threshold_for_safety_step} \cdot
    \text{current_trust_region_radius}`.

"""

TRUST_REGION_INCREASE_AFTER_SUCCESS = 2.0
r"""float: Ratio by which to increase the trust region radius :math:`\Delta_k` in
very successful iterations (:math:`\gamma_{inc}`).

"""

TRUST_REGION_INCREASE_AFTER_LARGE_SUCCESS = 4.0
r"""float: Ratio of the proposed step ($\|s_k\|$) by which to increase the trust region
radius (:math:`\Delta_k`) in very successful iterations
(:math:`\overline{\gamma}_{inc}`).

"""

TRUST_REGION_OPTIONS = {
    "threshold_successful": THRESHOLD_FOR_SUCCESSFUL_ITERATION,
    "threshold_very_successful": THRESHOLD_FOR_VERY_SUCCESFUL_ITERATION,
    "reduction_when_not_successful": None,
    "increase_after_success": TRUST_REGION_INCREASE_AFTER_SUCCESS,
    "increase_after_large_success": TRUST_REGION_INCREASE_AFTER_LARGE_SUCCESS,
    "min_decrease": None,
    "update_from_min_trust_region": None,
}
r"""dict: Options how the trust region is contracted and expanded.

    Possible entries are:

        threshold_successful (float): Minimum share of the predicted
            improvement that has to be realized for an iteration to count as successful.
        threshold_very_succesful (float): Share of predicted improvement
            that has to be surpassed for an iteration to count as very successful.
        reduction_when_not_successful (float): Ratio by which to
            decrease the trust region radius when realized improvement does not match
            the ``threshold_for_successful_iteration``. The default is 0.98 if
            ``criterion_noisy`` and 0.5 else.
        increase_after_success (float): Ratio by which to increase
            the trust region radius :math:`\Delta_k` in very successful iterations
            (:math:`\gamma_{inc}` in the notation of the paper).
        increase_after_large_success (float):
            Ratio of the proposed step ($\|s_k\|$) by which to increase the
            trust region radius (:math:`\Delta_k`) in very successful iterations
            (:math:`\overline{\gamma}_{inc}` in the notation of the paper).
        min_decrease (float):
            Ratio by which to decrease the minimal trust region radius
            (:math:`\rho_k`) (:math:`\alpha_1` in the notation of the paper).
            Default is 0.9 if ``criterion_noisy`` and 0.1 else.
        update_from_min_trust_region (float):
            Ratio of the current minimum trust region (:math:`\rho_k`) by which
            to decrease the actual trust region radius (:math:`\Delta_k`)
            when the minimum is reduced (:math:`\alpha_2` in the notation of the paper).
            Default is 0.95 if ``criterion_noisy`` and 0.5 else.

"""

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

CRITERION_NOISY = False
"""bool: Whether the criterion function is noisy.

    Your function is noisy if it does not always return the same value when evaluated
    at the same parameter values.

"""

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


SCALE_INTERPOLATION_SYSTEM = True
"""bool: whether to scale the interpolation linear system to improve conditioning."""

MAX_UNSUCCESSFUL_RESTARTS = 10
"""int: maximum number of consecutive unsuccessful restarts allowed.

    i.e. The number of restarts which did not improve upon the best known function value
    up to this point.

"""

MIN_TRUST_REGION_SCALING_AFTER_RESTART = 1.0
"""float: Factor with which the trust region stopping criterion is multiplied at each
restart."""

USE_SOFT_RESTARTS = True
"""bool: Whether to use soft or hard restarts."""

POINTS_TO_MOVE_AT_SOFT_RESTART = 3
"""int: Number of interpolation points to move at each soft restart."""

MOVE_CURRENT_POINT_AT_SOFT_RESTART = True
"""bool: Whether to move the current evaluation point ($x_k$) to the best new point."""

REUSE_CRITERION_VALUE_AT_HARD_RESTART = True
"""Whether or not to recycle the
criterion value at the best iterate found when performing a hard restart.
This saves one objective evaluation."""

ADDITIONAL_AUTOMATIC_RESTART_DETECTION = True
"""bool: Whether or not to automatically determine when to restart.

    This is an extra condition, and restarts can still be triggered by small trust
    region radius, etc.. There are two criteria used: trust region radius decreases
    (no increases over the history, more decreases than no changes) and
    change in model Jacobian (consistently increasing trend as measured
    by slope and correlation coefficient of line of best fit).

"""

N_ITERATIONS_FOR_AUTOMATIC_RESTART_DETECTION = 30
"""int: How many iterations of model changes and trust region radii to store."""

MIN_MODEL_SLOPE_INCREASE_FOR_AUTOMATIC_RESTART = 0.015
"""float: Minimum rate of increase of log gradients and log Hessians or the Jacobian
over past iterations to cause a restart.

"""

MIN_CORRELATIONS_FOR_AUTOMATIC_RESTART = 0.1
"""float: Minimum correlation of the log Gradient and log Hessian datasets or the
Jacobian dataset required to cause a restart.

"""

RESTART_OPTIONS = {
    "use_restarts": None,
    "max_unsuccessful": MAX_UNSUCCESSFUL_RESTARTS,
    "min_trust_region_scaling_after": MIN_TRUST_REGION_SCALING_AFTER_RESTART,
    "use_soft": USE_SOFT_RESTARTS,
    "move_current_point_at_soft": MOVE_CURRENT_POINT_AT_SOFT_RESTART,
    "reuse_criterion_value_at_hard": REUSE_CRITERION_VALUE_AT_HARD_RESTART,
    "max_iterations_without_new_best_after_soft": None,
    "automatic_detection": ADDITIONAL_AUTOMATIC_RESTART_DETECTION,
    "n_iterations_for_automatc_detection": N_ITERATIONS_FOR_AUTOMATIC_RESTART_DETECTION,  # noqa: E501
    "min_model_slope_increase_for_automatic_detection": MIN_MODEL_SLOPE_INCREASE_FOR_AUTOMATIC_RESTART,  # noqa: E501
    "min_correlations_for_automatic_detection": MIN_CORRELATIONS_FOR_AUTOMATIC_RESTART,
    "points_to_move_at_soft": POINTS_TO_MOVE_AT_SOFT_RESTART,
    # just bobyqa
    "max_unsuccessful_total": None,
    "trust_region_scaling_after_unsuccessful": None,
    # just dfols
    "max_interpolation_points": None,
    "n_interpolation_points_to_add_at_restart": 0,
    "n_interpolation_points_to_add_at_hard_restart_additionally": None,
}
r"""dict: Options for restarting the optimization.

    Possible entries are:

        use_restarts (bool): Whether to do restarts when the minimum trust
            region radius (:math:`\rho_k`) reaches the stopping criterion
            (:math:`\rho_{end}`), or (optionally) when all points are within noise
            level. Default is ``True`` if ``criterion_noisy``.
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
        n_interpolation_points_to_add_at_restart (int): Number by which to increase the
            number of interpolation points by with each restart.
        n_interpolation_points_to_add_at_hard_restart_additionally (int):
            Number by which to increase ``n_initial_points_to_add`` with each hard
            restart. To avoid a growing phase, it is best to set it to the same value
            as ``n_interpolation_points_to_add_at_restart``.

"""


FAST_START_OPTIONS = {
    "min_inital_points": None,
    "strategy": "auto",
    "scaling_of_trust_region_step_perturbation": None,
    "scaling_jacobian_perturb_components": 1e-2,
    "scaling_jacobian_perturb_floor_of_singular_values": 1,
    "jacobian_perturb_abs_floor_for_singular_values": 1e-6,
    "jacobian_perturb_max_condition_number": 1e8,
    "geometry_improving_steps": False,
    "safety_steps": True,
    "reduce_trust_region_with_safety_steps": False,
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
            values of the Jacobian at this factor of the last nonzero value.
            As of version 1.2.1 scaling_jacobian_perturb_floor_of_singular_values
            was not yet supported by DF-OLS.
        jacobian_perturb_abs_floor_for_singular_values (float): Absolute floor on
            singular values of the Jacobian. Default is 1e-6.
        jacobian_perturb_max_condition_number (float): Cap on the condition number
            of Jacobian after applying floors to singular values
            (effectively another floor on the smallest singular value, since the
            largest singular value is fixed). Default is 1e8.
        geometry_improving_steps (bool):
            While still growing the initial set, whether to do geometry-improving
            steps in the trust region algorithm. Default is False.
        safety_steps (bool):
            While still growing the initial set, whether to perform safety steps,
            or the regular trust region steps. Default is True.
        reduce_trust_region_with_safety_steps (bool):
            While still growing the initial set, whether to reduce trust region
            radius in safety steps. Default is False.
        reset_trust_region_radius_after (bool):
            Whether or not to reset the trust region radius to its initial value
            at the end of the growing phase. Default is False.
        reset_min_trust_region_radius_after (bool):
            Whether or not to reset the minimum trust region radius
            (:math:`\rho_k`) to its initial value at the end of the growing phase.
            Default is False.
        trust_region_decrease (float):
            Trust region decrease parameter during the growing phase. The default
            is ``reduction_when_not_successful``.
        n_search_directions_to_add_when_incomplete (int): Number of new search
            directions to add with each iteration where we do not have a full set
            of search directions. This approach is not recommended! Default is 0.

"""
