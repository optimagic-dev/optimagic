""""Implement algorithms by the (Numerical Algorithms Group)[https://www.nag.com/]."""
import warnings
from functools import partial

import numpy as np

from estimagic.config import ADDITIONAL_AUTOMATIC_RESTART_DETECTION
from estimagic.config import CLIP_CRITERION_IF_OVERFLOWING
from estimagic.config import COMPARISON_PERIOD_FOR_INSUFFICIENT_IMPROVEMENT
from estimagic.config import CRITERION_NOISY
from estimagic.config import INTERPOLATION_ROUNDING_ERROR
from estimagic.config import IS_DFOLS_INSTALLED
from estimagic.config import IS_PYBOBYQA_INSTALLED
from estimagic.config import MAX_CRITERION_EVALUATIONS
from estimagic.config import MAX_UNSUCCESSFUL_RESTARTS
from estimagic.config import MIN_CORRELATIONS_FOR_AUTOMATIC_RESTART
from estimagic.config import MIN_MODEL_SLOPE_INCREASE_FOR_AUTOMATIC_RESTART
from estimagic.config import MIN_TRUST_REGION_SCALING_AFTER_RESTART
from estimagic.config import MOVE_CURRENT_POINT_AT_SOFT_RESTART
from estimagic.config import N_ITERATIONS_FOR_AUTOMATIC_RESTART_DETECTION
from estimagic.config import NOISE_SCALE_FACTOR_FOR_QUIT
from estimagic.config import POINTS_TO_MOVE_AT_SOFT_RESTART
from estimagic.config import RANDOM_DIRECTIONS_ORTHOGONAL
from estimagic.config import RANDOM_INITIAL_DIRECTIONS
from estimagic.config import REUSE_CRITERION_VALUE_AT_HARD_RESTART
from estimagic.config import SCALE_INTERPOLATION_SYSTEM
from estimagic.config import SECOND_BEST_ABSOLUTE_PARAMS_TOLERANCE
from estimagic.config import THRESHOLD_FOR_SAFETY_STEP
from estimagic.config import THRESHOLD_FOR_SUCCESSFUL_ITERATION
from estimagic.config import THRESHOLD_FOR_VERY_SUCCESFUL_ITERATION
from estimagic.config import TRUST_REGION_INCREASE_AFTER_LARGE_SUCCESS
from estimagic.config import TRUST_REGION_INCREASE_AFTER_SUCCESS
from estimagic.config import USE_SOFT_RESTARTS
from estimagic.optimization.utilities import calculate_initial_trust_region_radius

try:
    import pybobyqa
except ImportError:
    pass

try:
    import dfols
except ImportError:
    pass


def nag_dfols(
    criterion_and_derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    max_criterion_evaluations=MAX_CRITERION_EVALUATIONS,
    initial_trust_region_radius=None,
    n_interpolation_points=None,
    max_interpolation_points=None,
    absolute_params_tolerance=SECOND_BEST_ABSOLUTE_PARAMS_TOLERANCE,
    criterion_noisy=CRITERION_NOISY,
    n_evals_per_point=None,
    interpolation_rounding_error=INTERPOLATION_ROUNDING_ERROR,
    threshold_for_safety_step=THRESHOLD_FOR_SAFETY_STEP,
    clip_criterion_if_overflowing=CLIP_CRITERION_IF_OVERFLOWING,
    random_initial_directions=RANDOM_INITIAL_DIRECTIONS,
    random_directions_orthogonal=RANDOM_DIRECTIONS_ORTHOGONAL,
    threshold_for_successful_iteration=THRESHOLD_FOR_SUCCESSFUL_ITERATION,
    threshold_for_very_succesful_iteration=THRESHOLD_FOR_VERY_SUCCESFUL_ITERATION,
    trust_region_reduction_when_not_successful=None,
    trust_region_increase_after_success=TRUST_REGION_INCREASE_AFTER_SUCCESS,
    trust_region_increase_after_large_success=TRUST_REGION_INCREASE_AFTER_LARGE_SUCCESS,
    min_trust_region_decrease=None,
    trust_region_update_from_min_trust_region=None,
    absolute_criterion_value_tolerance=None,
    threshold_for_insufficient_improvement=1e-4,
    n_insufficient_improvements_until_terminate=None,
    comparison_period_for_insufficient_improvement=COMPARISON_PERIOD_FOR_INSUFFICIENT_IMPROVEMENT,  # noqa: E501
    quit_when_trust_evaluations_within_noise=None,
    noise_scale_factor_for_quit=NOISE_SCALE_FACTOR_FOR_QUIT,
    multiplicative_noise_level=None,
    additive_noise_level=None,
    scale_interpolation_system=SCALE_INTERPOLATION_SYSTEM,
    use_restarts=None,
    max_unsuccessful_restarts=MAX_UNSUCCESSFUL_RESTARTS,
    min_trust_region_scaling_after_restart=MIN_TRUST_REGION_SCALING_AFTER_RESTART,
    use_soft_restarts=USE_SOFT_RESTARTS,
    move_current_point_at_soft_restart=MOVE_CURRENT_POINT_AT_SOFT_RESTART,
    reuse_criterion_value_at_hard_restart=REUSE_CRITERION_VALUE_AT_HARD_RESTART,
    max_iterations_without_new_best_after_soft_restart=None,
    additional_automatic_restart_detection=ADDITIONAL_AUTOMATIC_RESTART_DETECTION,
    n_iterations_for_automatic_restart_detection=N_ITERATIONS_FOR_AUTOMATIC_RESTART_DETECTION,  # noqa: E501
    min_model_slope_increase_for_automatic_restart=MIN_MODEL_SLOPE_INCREASE_FOR_AUTOMATIC_RESTART,  # noqa: E501
    min_correlations_for_automatic_restart=MIN_CORRELATIONS_FOR_AUTOMATIC_RESTART,
    relative_to_start_value_criterion_tolerance=0.0,
    n_extra_points_to_move_when_sufficient_improvement=0,
    n_extra_points_to_add_at_restart=0,
    use_momentum_method_to_move_extra_points=False,
    points_to_move_at_soft_restart=POINTS_TO_MOVE_AT_SOFT_RESTART,
    n_interpolation_points_to_add_at_restart=0,
):
    r"""Minimize a function with least squares structure using DFO-LS.

    The DFO-LS algorithm (:cite:`Cartis2018b`) solves

    .. math::

       \min_{x\in\mathbb{R}^n}  &\quad  f(x) := \sum_{i=1}^{m}r_{i}(x)^2 \\
       \text{s.t.} &\quad  a \

    Args:
        max_criterion_evaluations (int): If the maximum number of function evaluation is
            reached, the optimization stops but we do not count this as convergence.
        initial_trust_region_radius (float): Initial value of the trust region radius.
        n_interpolation_points (int): The number of interpolation points to use.
            With $n=len(x)$ the default is $n + 1$. If using restarts, this is the
            number of points to use in the first run of the solver, before any restarts.
        max_interpolation_points (int): Maximum allowed value of the number of
            interpolation points, useful if increasing with each restart. The default
            is `n_interpolation_points`.
        absolute_params_tolerance (float): Minimum allowed value of the trust region
            radius, which determines when a successful termination occurs.
        criterion_noisy (bool): Whether the criterion function is noisy, i.e. whether
            it does not always return the same value when evaluated at the same
            parameters.
        n_evals_per_point (callable): How often to evaluate the criterion function at
            each point.
            This is only applicable for criterion functions with stochastic noise,
            when averaging multiple evaluations at the same point produces a more
            accurate value.
            The input parameters are the ``trust_region_radius`` (``delta``),
            the ``min_trust_region_radius`` (``rho``),
            how many iterations the algorithm has been running for, ``n_iterations``
            and how many restarts have been performed, ``n_restarts``.
            The function must return an integer.
            Default is no averaging (i.e. ``n_evals_per_point(...) = 1``).
        interpolation_rounding_error (float): Internally, all interpolation
            points are stored with respect to a base point $x_b$; that is,
            pybobyqa stores $\{y_t-x_b\}$, which reduces the risk of roundoff
            errors. We shift $x_b$ to $x_k$ when
            :math:`\|s_k\| \leq
            \text{interpolation_rounding_error} \cdot \|x_k-x_b\|`
        threshold_for_safety_step (float): Threshold for when to call the safety step,
            :math:`\|s_k\| \leq \text{threshold_for_safety_step} \cdot \rho_k`
        clip_criterion_if_overflowing (bool): Whether to clip the criterion if it would
            raise an ``OverflowError`` otherwise.
        random_initial_directions (bool): Whether to draw the initial directions
            randomly (as opposed to coordinate directions).
        random_directions_orthogonal (bool): Whether to make random initial directions
            orthogonal.
        threshold_for_successful_iteration (float): Minimum share of the predicted
            improvement that has to be realized for an iteration to count as successful.
        threshold_for_very_succesful_iteration (float): Share of predicted improvement
            that has to be surpassed for an iteration to count as very successful.
        trust_region_reduction_when_not_successful (float): Ratio by which to
            decrease the trust region radius when realized improvement does not match
            the ``threshold_for_successful_iteration``. The default is 0.98 if
            ``criterion_noisy`` and 0.5 else.
        trust_region_increase_after_success (float): Ratio by which to increase
            the trust region radius :math:`\Delta_k` in very successful iterations
            (:math:`\gamma_{inc}`).
        trust_region_increase_after_large_success (float):
            Ratio of the proposed step ($\|s_k\|$) by which to increase the
            trust region radius (:math:`\Delta_k`) in very successful iterations
            (:math:`\overline{\gamma}_{inc}`).
        min_trust_region_decrease (float):
            Ratio by which to decrease the minimal trust region radius
            (:math:`\rho_k`) (:math:`\alpha_1`).
            Default is 0.9 if ``criterion_noisy`` and 0.1 else.
        trust_region_update_from_min_trust_region (float):
            Ratio of the current minimum trust region (:math:`\rho_k`) by which
            to decrease the actual trust region radius (:math:`\Delta_k`)
            when the lower bound is reduced (:math:`\alpha_2`). Default is 0.95 if
            ``criterion_noisy`` and 0.5 else.
        absolute_criterion_value_tolerance (float): Terminate successfully if
            the criterion value falls below this threshold. This is currently not yet
            supported by nag_dfols.
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
        quit_when_trust_evaluations_within_noise (bool): Flag to quit
            (or restart) if all $f(y_t)$ are within noise level of
            $f(x_k)$. Default is ``True`` if ``noisy_criterion`` and
            ``False`` else.
        noise_scale_factor_for_quit (float): Factor of the noise level to use in
            termination criterion.
        multiplicative_noise_level (float): Multiplicative noise level in the
            criterion. You can only specify ``multiplicative_noise_level`` or
            ``additive_noise_level``.
        additive_noise_level (float): Additive noise level in the
            criterion. You can only specify ``multiplicative_noise_level`` or
            ``additive_noise_level``.
        scale_interpolation_system (bool): Whether or not to scale the interpolation
            linear system to improve conditioning.
        use_restarts (bool): Whether to do restarts when the lower bound on the trust
            region radius (:math:`\rho_k`) reaches the stopping criterion
            (:math:`\rho_{end}`), or (optionally) when all points are within noise
            level. Default is ``True`` if ``criterion_noisy`` or when
            ``seek_global_optimum``.
        max_unsuccessful_restarts (int): maximum number of consecutive unsuccessful
            restarts allowed (i.e. restarts which did not reduce the objective further)
        min_trust_region_scaling_after_restart (float): Factor with which
            the trust region stopping criterion is multiplied at each restart.
        use_soft_restarts (bool): Whether to use soft or hard restarts.
        move_current_point_at_soft_restart (bool): Whether to move the current
            evaluation point ($x_k$) to the best new point evaluated.
        points_to_move_at_soft_restart (int): Number of interpolation points to move
            at each soft restart.
        reuse_criterion_value_at_hard_restart (bool): Whether or not to recycle the
            criterion value at the best iterate found when performing a hard restart.
            This saves one objective evaluation.
        max_iterations_without_new_best_after_soft_restart (int):
            The maximum number of successful steps in a given run where the new
            objective value is worse than the best value found in previous runs before
            terminating. Default is ``max_criterion_evaluations``.
        additional_automatic_restart_detection (bool): Whether or not to
            automatically determine when to restart. This is an extra condition, and
            restarts can still be triggered by small trust region radius, etc.
            There are two criteria used: trust region radius decreases
            (no increases over the history, more decreases than no changes) and
            change in model Jacobian (consistently increasing trend as measured
            by slope and correlation coefficient of line of best fit).
        n_iterations_for_automatic_restart_detection (int):
            How many iterations of model changes and trust region radii to store.
        min_model_slope_increase_for_automatic_restart (float):
            Minimum rate of increase of the Jacobian over past iterations to cause a
            restart.
        min_correlations_for_automatic_restart (float):
            Minimum correlation of the Jacobian data set required to cause a restart.
        relative_to_start_value_criterion_tolerance (float):
            Terminate if a point is reached where the ratio of the criterion value
            to the criterion value at the start params is below this value, i.e. if
            :math:`f(x_k)/f(x_0) \leq
            \text{relative_to_start_value_criterion_tolerance}`. Note this is
            deactivated unless the lowest mathematically possible criterion value (0.0)
            is actually achieved.
        n_extra_points_to_move_when_sufficient_improvement (int): The number of extra
            points (other than accepting the trust region step) to move. Useful when
            n_interpolation points > number of parameters + 1.
        n_extra_points_to_add_at_restart (int): The number by which to increase
            `n_extra_points_to_move_when_sufficient_improvement` at each restart.
        use_momentum_method_to_move_extra_points (bool): If moving extra points in
            successful iterations, whether to use the 'momentum' method. If not,
            uses geometry-improving steps.
        n_interpolation_points_to_add_at_restart (int): Amount to increase the number
            of interpolation points by with each restart.

    Returns:
        results (dict): See :ref:`internal_optimizer_output` for details.

    """
    if not IS_DFOLS_INSTALLED:
        raise NotImplementedError(
            "The dfols package is not installed and required for 'nag_dfols'. "
            "You can install it with 'pip install DFOLS'. "
            "For additional installation instructions visit: ",
            r"https://numericalalgorithmsgroup.github.io/dfols/build/html/install.html",
        )

    if initial_trust_region_radius is None:
        initial_trust_region_radius = calculate_initial_trust_region_radius(x)
    # -np.inf as a default leads to errors when building the documentation with sphinx.
    if absolute_criterion_value_tolerance is not None:
        warnings.warn(
            "absolute_criterion_value_tolerance is currently not yet supported by "
            "nag_dfols so this is option is ignored for the moment."
        )
    if n_evals_per_point is not None:

        def adjusted_n_evals_per_point(delta, rho, iter, nrestarts):  # noqa: A002
            return n_evals_per_point(
                trust_region_radius=delta,
                min_trust_region=rho,
                n_iterations=iter,
                n_restarts=nrestarts,
            )

    else:
        adjusted_n_evals_per_point = None

    algo_info = {
        "name": "nag_dfols",
        "primary_criterion_entry": "root_contributions",
        "parallelizes": False,
        "needs_scaling": False,
    }
    advanced_options = {
        "general.rounding_error_constant": interpolation_rounding_error,
        "general.safety_step_thresh": threshold_for_safety_step,
        "general.check_objfun_for_overflow": clip_criterion_if_overflowing,
        "init.random_initial_directions": random_initial_directions,
        "init.random_directions_make_orthogonal": random_directions_orthogonal,
        "tr_radius.eta1": threshold_for_successful_iteration,
        "tr_radius.eta2": threshold_for_very_succesful_iteration,
        "tr_radius.gamma_dec": trust_region_reduction_when_not_successful,
        "tr_radius.gamma_inc": trust_region_increase_after_success,
        "tr_radius.gamma_inc_overline": trust_region_increase_after_large_success,
        "tr_radius.alpha1": min_trust_region_decrease,
        "tr_radius.alpha2": trust_region_update_from_min_trust_region,
        "slow.thresh_for_slow": threshold_for_insufficient_improvement,
        "slow.max_slow_iters": n_insufficient_improvements_until_terminate,
        "slow.history_for_slow": comparison_period_for_insufficient_improvement,
        "noise.quit_on_noise_level": quit_when_trust_evaluations_within_noise,
        "noise.scale_factor_for_quit": noise_scale_factor_for_quit,
        "noise.multiplicative_noise_level": multiplicative_noise_level,
        "noise.additive_noise_level": additive_noise_level,
        "interpolation.precondition": scale_interpolation_system,
        "restarts.use_restarts": use_restarts,
        "restarts.max_unsuccessful_restarts": max_unsuccessful_restarts,
        "restarts.rhoend_scale": min_trust_region_scaling_after_restart,
        "restarts.use_soft_restarts": use_soft_restarts,
        "restarts.soft.move_xk": move_current_point_at_soft_restart,
        "restarts.hard.use_old_rk": reuse_criterion_value_at_hard_restart,
        "restarts.soft.max_fake_successful_steps": max_iterations_without_new_best_after_soft_restart,  # noqa: E501
        "restarts.auto_detect": additional_automatic_restart_detection,
        "restarts.auto_detect.history": n_iterations_for_automatic_restart_detection,  # noqa: E501
        "restarts.auto_detect.min_chgJ_slope": min_model_slope_increase_for_automatic_restart,  # noqa: E501
        "restarts.auto_detect.min_correl": min_correlations_for_automatic_restart,
        "model.rel_tol": relative_to_start_value_criterion_tolerance,
        "regression.num_extra_steps": n_extra_points_to_move_when_sufficient_improvement,  # noqa: E501
        "regression.increase_num_extra_steps_with_restart": n_extra_points_to_add_at_restart,  # noqa: E501
        "regression.momentum_extra_steps": use_momentum_method_to_move_extra_points,
        "restarts.soft.num_geom_steps": points_to_move_at_soft_restart,
        "restarts.increase_npt": n_interpolation_points_to_add_at_restart > 0,
        "restarts.increase_npt_amt": n_interpolation_points_to_add_at_restart,
        "restarts.max_npt": max_interpolation_points,
    }
    criterion = partial(
        criterion_and_derivative, task="criterion", algorithm_info=algo_info
    )

    res = dfols.solve(
        criterion,
        x0=x,
        bounds=(lower_bounds, upper_bounds),
        maxfun=max_criterion_evaluations,
        rhobeg=initial_trust_region_radius,
        npt=n_interpolation_points,
        rhoend=absolute_params_tolerance,
        nsamples=adjusted_n_evals_per_point,
        objfun_has_noise=criterion_noisy,
        scaling_within_bounds=False,
        do_logging=False,
        print_progress=False,
        user_params=advanced_options,
    )

    return _process_nag_result(res, len(x))


def nag_pybobyqa(
    criterion_and_derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    max_criterion_evaluations=MAX_CRITERION_EVALUATIONS,
    absolute_params_tolerance=SECOND_BEST_ABSOLUTE_PARAMS_TOLERANCE,
    initial_trust_region_radius=None,
    seek_global_optimum=False,
    random_initial_directions=RANDOM_INITIAL_DIRECTIONS,
    random_directions_orthogonal=RANDOM_DIRECTIONS_ORTHOGONAL,
    n_interpolation_points=None,
    criterion_noisy=CRITERION_NOISY,
    n_evals_per_point=None,
    interpolation_rounding_error=INTERPOLATION_ROUNDING_ERROR,
    threshold_for_safety_step=THRESHOLD_FOR_SAFETY_STEP,
    threshold_for_successful_iteration=THRESHOLD_FOR_SUCCESSFUL_ITERATION,
    threshold_for_very_succesful_iteration=THRESHOLD_FOR_VERY_SUCCESFUL_ITERATION,
    trust_region_reduction_when_not_successful=None,
    clip_criterion_if_overflowing=CLIP_CRITERION_IF_OVERFLOWING,
    absolute_criterion_value_tolerance=None,
    trust_region_increase_after_success=TRUST_REGION_INCREASE_AFTER_SUCCESS,
    trust_region_increase_after_large_success=TRUST_REGION_INCREASE_AFTER_LARGE_SUCCESS,
    min_trust_region_decrease=None,
    trust_region_update_from_min_trust_region=None,
    threshold_for_insufficient_improvement=1e-8,
    n_insufficient_improvements_until_terminate=None,
    comparison_period_for_insufficient_improvement=COMPARISON_PERIOD_FOR_INSUFFICIENT_IMPROVEMENT,  # noqa E501
    quit_when_trust_evaluations_within_noise=None,
    noise_scale_factor_for_quit=NOISE_SCALE_FACTOR_FOR_QUIT,
    multiplicative_noise_level=None,
    additive_noise_level=None,
    scale_interpolation_system=SCALE_INTERPOLATION_SYSTEM,
    frobenius_for_interpolation_problem=True,
    use_restarts=None,
    max_unsuccessful_restarts=MAX_UNSUCCESSFUL_RESTARTS,
    max_unsuccessful_restarts_total=None,
    trust_region_scaling_after_unsuccessful_restart=None,
    min_trust_region_scaling_after_restart=MIN_TRUST_REGION_SCALING_AFTER_RESTART,
    use_soft_restarts=USE_SOFT_RESTARTS,
    points_to_move_at_soft_restart=POINTS_TO_MOVE_AT_SOFT_RESTART,
    move_current_point_at_soft_restart=MOVE_CURRENT_POINT_AT_SOFT_RESTART,
    reuse_criterion_value_at_hard_restart=REUSE_CRITERION_VALUE_AT_HARD_RESTART,
    max_iterations_without_new_best_after_soft_restart=None,
    additional_automatic_restart_detection=ADDITIONAL_AUTOMATIC_RESTART_DETECTION,
    n_iterations_for_automatic_restart_detection=N_ITERATIONS_FOR_AUTOMATIC_RESTART_DETECTION,  # noqa: E501
    min_model_slope_increase_for_automatic_restart=MIN_MODEL_SLOPE_INCREASE_FOR_AUTOMATIC_RESTART,  # noqa: E501
    min_correlations_for_automatic_restart=MIN_CORRELATIONS_FOR_AUTOMATIC_RESTART,
):
    r"""Minimize a function using the BOBYQA algorithm.

    BOBYQA (:cite:`Powell2009`, :cite:`Cartis2018`, :cite:`Cartis2018a`) is a
    derivative-free trust-region method. It is designed to solve nonlinear local
    minimization problems.

    There are two main situations when using a derivative-free algorithm like BOBYQA
    is preferable to derivative-based algorithms:

    1. The criterion function is not deterministic, i.e. if we evaluate the criterion
       function multiple times at the same parameter vector we get different results.

    2. The criterion function is very expensive to evaluate and only finite differences
       are available to calculate its derivative.

    The detailed documentation of the algorithm can be found `here
    <https://numericalalgorithmsgroup.github.io/pybobyqa/build/html/index.html>`_.

    There are four possible convergence criteria:

    1. when the trust region radius is reduced below a lower bound. This is
       approximately equivalent to an absolute parameter tolerance.

    2. when the criterion value falls below an absolute, user-specified value,
       the optimization terminates successfully.

    3. when insufficient improvements have been gained over a certain number of
       iterations. The (absolute) threshold for what constitutes an insufficient
       improvement, how many iterations have to be insufficient and with which
       iteration to compare can all be specified by the user.

    4. when all evaluations on the trust region points fall within a scaled version of
       the noise level of the criterion function. This is only applicable if the
       criterion function is noisy.

    Args:
        absolute_params_tolerance (float): Minimum allowed value of the trust region
            radius, which determines when a successful termination occurs.
        max_criterion_evaluations (int): If the maximum number of function evaluation is
            reached, the optimization stops but we do not count this as convergence.
        initial_trust_region_radius (float): Initial value of the trust region radius.
        seek_global_optimum (bool): whether to apply the heuristic to escape local
            minima presented in :cite:`Cartis2018a`. Only applies for noisy criterion
            functions.
        random_initial_directions (bool): Whether to draw the initial directions
            randomly (as opposed to coordinate directions).
        random_directions_orthogonal (bool): Whether to make random initial directions
            orthogonal.
        n_interpolation_points (int): The number of interpolation points to use.
            With $n=len(x)$ the default is $2n+1$ if not ``criterion_noisy``.
            Otherwise, it is set to $(n+1)(n+2)/2)$.

            Larger values are particularly useful for noisy problems.
            Py-BOBYQA requires

            .. math::
                n + 1 \leq \text{n_interpolation_points} \leq (n+1)(n+2)/2.

        criterion_noisy (bool): Whether the criterion function is noisy, i.e. whether
            it does not always return the same value when evaluated at the same
            parameters.
        n_evals_per_point (callable): How often to evaluate the criterion function at
            each point.
            This is only applicable for criterion functions with stochastic noise,
            when averaging multiple evaluations at the same point produces a more
            accurate value.
            The input parameters are the ``trust_region_radius`` (``delta``),
            the ``min_trust_region_radius`` (``rho``),
            how many iterations the algorithm has been running for, ``n_iterations``
            and how many restarts have been performed, ``n_restarts``.
            The function must return an integer.
            Default is no averaging (i.e. ``n_evals_per_point(...) = 1``).
        interpolation_rounding_error (float): Internally, all interpolation
            points are stored with respect to a base point $x_b$; that is,
            pybobyqa stores $\{y_t-x_b\}$, which reduces the risk of roundoff
            errors. We shift $x_b$ to $x_k$ when
            :math:`\|s_k\| \leq
            \text{interpolation_rounding_error} \cdot \|x_k-x_b\|` where $s_k$ is the
            proposed step size.
        threshold_for_safety_step (float): Threshold for when to call the safety step,
            :math:`\|s_k\| \leq \text{threshold_for_safety_step} \cdot \rho_k`
            where $s_k$ is the proposed step size and :math:`\rho_k` the current trust
            region radius.
        threshold_for_successful_iteration (float): Minimum share of the predicted
            improvement that has to be realized for an iteration to count as successful.
        threshold_for_very_succesful_iteration (float): Share of predicted improvement
            that has to be surpassed for an iteration to count as very successful.
        trust_region_reduction_when_not_successful (float): Ratio by which to
            decrease the trust region radius when realized improvement does not match
            the ``threshold_for_successful_iteration``. The default is 0.98 if
            ``criterion_noisy`` and 0.5 else.
        clip_criterion_if_overflowing (bool): Whether to clip the criterion if it would
            raise an ``OverflowError`` otherwise.
        absolute_criterion_value_tolerance (float): Terminate successfully if
            the criterion value falls below this threshold. This is deactivated
            (i.e. set to -inf) by default.
        trust_region_increase_after_success (float): Ratio by which to increase
            the trust region radius :math:`\Delta_k` in very successful iterations
            (:math:`\gamma_{inc}`).
        trust_region_increase_after_large_success (float):
            Ratio of the proposed step ($\|s_k\|$) by which to increase the
            trust region radius (:math:`\Delta_k`) in very successful iterations
            (:math:`\overline{\gamma}_{inc}`).
        min_trust_region_decrease (float):
            Ratio by which to decrease the minimal trust region radius
            (:math:`\rho_k`) (:math:`\alpha_1`).
            Default is 0.9 if ``criterion_noisy`` and 0.1 else.
        trust_region_update_from_min_trust_region (float):
            Ratio of the current minimum trust region (:math:`\rho_k`) by which
            to decrease the actual trust region radius (:math:`\Delta_k`)
            when the lower bound is reduced (:math:`\alpha_2`). Default is 0.95 if
            ``criterion_noisy`` and 0.5 else.
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
        quit_when_trust_evaluations_within_noise (bool): Flag to quit
            (or restart) if all $f(y_t)$ are within noise level of
            $f(x_k)$. Default is ``True`` if ``noisy_criterion`` and
            ``False`` else.
        noise_scale_factor_for_quit (float): Factor of the noise level to use in
            termination criterion.
        multiplicative_noise_level (float): Multiplicative noise level in the
            criterion. You can only specify ``multiplicative_noise_level`` or
            ``additive_noise_level``.
        additive_noise_level (float): Additive noise level in the
            criterion. You can only specify ``multiplicative_noise_level`` or
            ``additive_noise_level``.
        scale_interpolation_system (bool): Whether or not to scale the interpolation
            linear system to improve conditioning.
        frobenius_for_interpolation_problem (bool): Whether to solve the
            underdetermined quadratic interpolation problem by minimizing the Frobenius
            norm of the Hessian, or change in Hessian.
        use_restarts (bool): Whether to do restarts when the lower bound on the trust
            region radius (:math:`\rho_k`) reaches the stopping criterion
            (:math:`\rho_{end}`), or (optionally) when all points are within noise
            level. Default is ``True`` if ``criterion_noisy`` or when
            ``seek_global_optimum``.
        max_unsuccessful_restarts (int): maximum number of consecutive unsuccessful
            restarts allowed (i.e. restarts which did not reduce the objective further)
        max_unsuccessful_restarts_total (int): number of total unsuccessful restarts
            allowed. Default is 20 if ``seek_global_optimum`` and else unrestricted.
        trust_region_scaling_after_unsuccessful_restart (float): Factor by which to
            increase the initial trust region radius (:math:`\rho_{beg}`) after
            unsuccessful restarts. Default is 1.1 if ``seek_global_optimum`` else 1.
        min_trust_region_scaling_after_restart (float): Factor with which
            the trust region stopping criterion is multiplied at each restart.
        use_soft_restarts (bool): Whether to use soft or hard restarts.
        points_to_move_at_soft_restart (int): Number of interpolation points to
            move at each soft restart.
        move_current_point_at_soft_restart (bool): Whether to move the current
            evaluation point ($x_k$) to the best new point evaluated.
        reuse_criterion_value_at_hard_restart (bool): Whether or not to recycle the
            criterion value at the best iterate found when performing a hard restart.
            This saves one objective evaluation.
        max_iterations_without_new_best_after_soft_restart (int):
            The maximum number of successful steps in a given run where the new
            objective value is worse than the best value found in previous runs before
            terminating. Default is ``max_criterion_evaluations``.
        additional_automatic_restart_detection (bool): Whether or not to
            automatically determine when to restart. This is an extra condition, and
            restarts can still be triggered by small trust region radius, etc.
            There are two criteria used: trust region radius decreases
            (no increases over the history, more decreases than no changes) and
            change in model Jacobian (consistently increasing trend as measured
            by slope and correlation coefficient of line of best fit).
        n_iterations_for_automatic_restart_detection (int):
            How many iterations of model changes and trust region radii to store.
        min_model_slope_increase_for_automatic_restart (float):
            Minimum rate of increase of log gradients and log Hessians over past
            iterations to cause a restart.
        min_correlations_for_automatic_restart (float):
            Minimum correlation of the log Gradient and log Hessian datasets
            required to cause a restart.

    Returns:
        results (dict): See :ref:`internal_optimizer_output` for details.

    """
    if not IS_PYBOBYQA_INSTALLED:
        raise NotImplementedError(
            "The pybobyqa package is not installed and required for 'nag_pybobyqa'. "
            "You can install it with 'pip install Py-BOBYQA'. "
            "For additional installation instructions visit: ",
            r"https://numericalalgorithmsgroup.github.io/pybobyqa/build/html/"
            "install.html",
        )

    if initial_trust_region_radius is None:
        initial_trust_region_radius = calculate_initial_trust_region_radius(x)
    # -np.inf as a default leads to errors when building the documentation with sphinx.
    if absolute_criterion_value_tolerance is None:
        absolute_criterion_value_tolerance = -np.inf
    if n_evals_per_point is not None:

        def adjusted_n_evals_per_point(delta, rho, iter, nrestarts):  # noqa: A002
            return n_evals_per_point(
                trust_region_radius=delta,
                min_trust_region=rho,
                n_iterations=iter,
                n_restarts=nrestarts,
            )

    else:
        adjusted_n_evals_per_point = None

    algo_info = {
        "name": "nag_pybobyqa",
        "primary_criterion_entry": "value",
        "parallelizes": False,
        "needs_scaling": False,
    }
    criterion = partial(
        criterion_and_derivative, task="criterion", algorithm_info=algo_info
    )

    advanced_options = {
        "general.rounding_error_constant": interpolation_rounding_error,
        "general.safety_step_thresh": threshold_for_safety_step,
        "general.check_objfun_for_overflow": clip_criterion_if_overflowing,
        "init.random_initial_directions": random_initial_directions,
        "init.random_directions_make_orthogonal": random_directions_orthogonal,
        "tr_radius.eta1": threshold_for_successful_iteration,
        "tr_radius.eta2": threshold_for_very_succesful_iteration,
        "tr_radius.gamma_dec": trust_region_reduction_when_not_successful,
        "tr_radius.gamma_inc": trust_region_increase_after_success,
        "tr_radius.gamma_inc_overline": trust_region_increase_after_large_success,
        "tr_radius.alpha1": min_trust_region_decrease,
        "tr_radius.alpha2": trust_region_update_from_min_trust_region,
        "model.abs_tol": absolute_criterion_value_tolerance,
        "slow.thresh_for_slow": threshold_for_insufficient_improvement,
        "slow.max_slow_iters": n_insufficient_improvements_until_terminate,
        "slow.history_for_slow": comparison_period_for_insufficient_improvement,
        "noise.quit_on_noise_level": quit_when_trust_evaluations_within_noise,
        "noise.scale_factor_for_quit": noise_scale_factor_for_quit,
        "noise.multiplicative_noise_level": multiplicative_noise_level,
        "noise.additive_noise_level": additive_noise_level,
        "interpolation.precondition": scale_interpolation_system,
        "interpolation.minimum_change_hessian": frobenius_for_interpolation_problem,
        "restarts.use_restarts": use_restarts,
        "restarts.max_unsuccessful_restarts": max_unsuccessful_restarts,
        "restarts.max_unsuccessful_restarts_total": max_unsuccessful_restarts_total,
        "restarts.rhobeg_scale_after_unsuccessful_restart": trust_region_scaling_after_unsuccessful_restart,  # noqa E501
        "restarts.rhoend_scale": min_trust_region_scaling_after_restart,
        "restarts.use_soft_restarts": use_soft_restarts,
        "restarts.soft.num_geom_steps": points_to_move_at_soft_restart,
        "restarts.soft.move_xk": move_current_point_at_soft_restart,
        "restarts.hard.use_old_fk": reuse_criterion_value_at_hard_restart,
        "restarts.soft.max_fake_successful_steps": max_iterations_without_new_best_after_soft_restart,  # noqa: E501
        "restarts.auto_detect": additional_automatic_restart_detection,
        "restarts.auto_detect.history": n_iterations_for_automatic_restart_detection,  # noqa: E501
        "restarts.auto_detect.min_chg_model_slope": min_model_slope_increase_for_automatic_restart,  # noqa: E501
        "restarts.auto_detect.min_correl": min_correlations_for_automatic_restart,
    }

    res = pybobyqa.solve(
        criterion,
        x0=x,
        bounds=(lower_bounds, upper_bounds),
        maxfun=max_criterion_evaluations,
        rhobeg=initial_trust_region_radius,
        user_params=advanced_options,
        scaling_within_bounds=False,
        do_logging=False,
        print_progress=False,
        objfun_has_noise=criterion_noisy,
        nsamples=adjusted_n_evals_per_point,
        npt=n_interpolation_points,
        rhoend=absolute_params_tolerance,
        seek_global_minimum=seek_global_optimum,
    )

    return _process_nag_result(res, len(x))


def _process_nag_result(nag_result_obj, len_x):
    processed = {
        "solution_criterion": nag_result_obj.f,
        "n_iterations": nag_result_obj.nf,
        "n_criterion_evaluations": nag_result_obj.nx,
        "message": nag_result_obj.msg,
        "success": nag_result_obj.flag == nag_result_obj.EXIT_SUCCESS,
        "reached_convergence_criterion": None,
    }
    if nag_result_obj.x is not None:
        processed["solution_x"] = nag_result_obj.x
    else:
        processed["solution_x"] = np.array([np.nan] * len_x)
    try:
        processed["solution_derivative"] = nag_result_obj.gradient
    except AttributeError:
        pass
    try:
        processed["solution_hessian"] = nag_result_obj.hessian
    except AttributeError:
        pass
    return processed
