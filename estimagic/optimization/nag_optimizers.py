""""Implement algorithms by the (Numerical Algorithms Group)[https://www.nag.com/]."""
from functools import partial

import numpy as np

from estimagic.config import CRITERION_NOISY
from estimagic.config import IS_PYBOBYQA_INSTALLED
from estimagic.config import MAX_CRITERION_EVALUATIONS
from estimagic.config import MIN_IMPROVEMENT_FOR_SUCCESSFUL_ITERATION
from estimagic.config import RANDOM_DIRECTIONS_ORTHOGONAL
from estimagic.config import RANDOM_INITIAL_DIRECTIONS
from estimagic.config import SECOND_BEST_ABSOLUTE_PARAMS_TOLERANCE
from estimagic.config import THRESHOLD_FOR_VERY_SUCCESFUL_ITERATION
from estimagic.optimization.utilities import calculate_initial_trust_region_radius

try:
    import pybobyqa
except ImportError:
    pass


def nag_pybobyqa(
    criterion_and_derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    max_criterion_evaluations=MAX_CRITERION_EVALUATIONS,
    absolute_params_tolerance=SECOND_BEST_ABSOLUTE_PARAMS_TOLERANCE,
    initial_trust_region_radius=None,
    seek_global_minimum=False,
    random_initial_directions=RANDOM_INITIAL_DIRECTIONS,
    random_directions_orthogonal=RANDOM_DIRECTIONS_ORTHOGONAL,
    nr_interpolation_points=None,
    criterion_noisy=CRITERION_NOISY,
    nr_evals_per_point=None,
    interpolation_rounding_error_constant=0.1,
    threshold_for_safety_step=0.5,
    intermediate_processing_of_initial_points=True,
    min_improvement_for_successful_iteration=MIN_IMPROVEMENT_FOR_SUCCESSFUL_ITERATION,
    threshold_for_very_succesful_iteration=THRESHOLD_FOR_VERY_SUCCESFUL_ITERATION,
    trust_region_radius_reduction_when_not_successful=None,
    clip_criterion_if_overflowing=True,
    absolute_criterion_value_stopping_criterion=None,
    trust_region_increase_factor_after_success=2.0,
    trust_region_increase_factor_after_large_success=4.0,
    trust_region_lower_bound_decrease_factor=None,
    threshold_for_insufficient_improvement=1e-8,
    nr_of_insufficient_improvements_to_terminate=None,
    comparison_period_for_insufficient_improvement=5,
    quit_when_trust_evaluations_within_noise_level=None,
    noise_scale_factor_for_quit=1.0,
    multiplicative_noise_level=None,
    additive_noise_level=None,
    scale_interpolation_system=True,
    frobenius_for_interpolation_problem=True,
    use_restarts=None,
    max_unsuccessful_restarts=10,
    max_unsuccessful_restarts_total=None,
    trust_region_rescaling_after_unsuccessful_restart=None,
    trust_region_stop_criterion_scaling_after_restart=1.0,
    use_soft_restarts=True,
    nr_points_to_move_at_soft_restart=3,
    move_current_point_at_soft_restart=True,
    reuse_criterion_value_at_hard_restart=True,
    max_nr_only_local_improvements_after_soft_restart=None,
    additional_automatic_restart_detection=True,
    iterations_to_save_for_automatic_restart_detection=30,
    min_model_slope_increase_for_automatic_restart=0.015,
    min_correlations_for_automatic_restart=0.1,
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
        seek_global_minimum (bool): whether to apply the heuristic to escape local
            minima presented in :cite:`Cartis2018a`. Only applies for noisy criterion
            functions.
        random_initial_directions (bool): Whether to draw the initial directions
            randomly (as opposed to coordinate directions).
        random_directions_orthogonal (bool): Whether to make random initial directions
            orthogonal.
        nr_interpolation_points (int): The number of interpolation points to use.
            With $n=len(x)$ the default is $2n+1$ if not ``criterion_noisy``.
            Otherwise, it is set to $(n+1)(n+2)/2)$.

            Larger values are particularly useful for noisy problems.
            Py-BOBYQA requires

            .. math::
                n + 1 \leq \text{nr_interpolation_points} \leq (n+1)(n+2)/2.

        criterion_noisy (bool): Whether the criterion function is noisy, i.e. whether
            it does not always return the same value when evaluated at the same
            parameters.
        nr_evals_per_point (callable): How often to evaluate the criterion function at
            each point. The function must take ``delta``, ``rho``, ``iter`` and
            ``nrestarts`` as arguments and return an integer.
            This is only applicable for criterion functions with stochastic noise,
            when averaging multiple evaluations at the same point produces a more
            accurate value.
            The input parameters are the trust region radius (``delta``),
            the lower bound on the trust region radius (``rho``),
            how many iterations the algorithm has been running for (``iter``)
            and how many restarts have been performed (``nrestarts``).
            Default is no averaging (i.e.
            ``nr_evals_per_point(delta, rho, iter, nrestarts) = 1``).
        interpolation_rounding_error_constant (float): Internally, all interpolation
            points are stored with respect to a base point $x_b$; that is,
            pybobyqa stores $\{y_t-x_b\}$, which reduces the risk of roundoff
            errors. We shift $x_b$ to $x_k$ when
            :math:`\|s_k\| \leq
            \text{interpolation_rounding_error_constant} \cdot \|x_k-x_b\|`
        threshold_for_safety_step (float): Threshold for when to call the safety step,
            :math:`\|s_k\| \leq \text{threshold_for_safety_step} \cdot \rho_k`
        intermediate_processing_of_initial_points (bool): If using random directions,
            whether to do intermediate processing between point evaluations.
        min_improvement_for_successful_iteration (float): Minimum share of the predicted
            improvement that has to be realized for an iteration to count as successful.
        threshold_for_very_succesful_iteration (float): Share of predicted improvement
            that has to be surpassed for an iteration to count as very successful.
        trust_region_reduction_ratio_when_not_successful (float): Ratio by which to
            decrease the trust region radius when realized improvement does not match
            the ``min_improvement_for_successful_iteration``. The default is 0.98 if
            ``criterion_noisy`` and 0.5 else.
        clip_criterion_if_overflowing (bool): Whether to clip the criterion if it would
            raise an ``OverflowError`` otherwise.
        absolute_criterion_value_stopping_criterion (float): Terminate successfully if
            the criterion value falls below this threshold. This is deactivated
            (i.e. set to -inf) by default.
        trust_region_increase_factor_after_success (float): Ratio by which to increase
            the trust region radius :math:`\Delta_k` in very successful iterations
            (:math:`\gamma_{inc}`).
        trust_region_increase_factor_after_large_success (float):
            Ratio of the proposed step ($\|s_k\|$) by which to increase the
            trust region radius (:math:`\Delta_k`) in very successful iterations
            (:math:`\overline{\gamma}_{inc}`).
        trust_region_lower_bound_decrease_factor (float):
            Ratio by which to decrease the lower bound on the trust region radius
            (:math:`\rho_k`) (:math:`\alpha_1`).
            Default is 0.9 if ``criterion_noisy`` and 0.1 else.
        trust_region_update_from_lower_bound_to_trust_region_factor (float):
            Ratio of the lower bound on the trust region (:math:`\rho_k`) by which
            to decrease the actual trust region radius (:math:`\Delta_k`)
            when the lower bound is reduced (:math:`\alpha_2`). Default is 0.95 if
            ``criterion_noisy`` and 0.5 else.
        threshold_for_insufficient_improvement (float): Threshold whether an improvement
            is insufficient. Note: the improvement is divided by the
            ``comparison_period_for_insufficient_improvement``.
            So this is the required average improvement per iteration over the
            comparison period.
        nr_of_insufficient_improvements_to_terminate (int): Number of consecutive
            insufficient improvements before termination (or restart). Default is
            ``20 * len(x)``.
        comparison_period_for_insufficient_improvement (int):
            How many iterations to go back to calculate the improvement.
            For example 5 would mean that each criterion evaluation is compared to the
            criterion value from 5 iterations before.
        quit_when_trust_evaluations_within_noise_level (bool): Flag to quit
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
        scale_interpolation_system (bool): whether or not to scale the interpolation
            linear system to improve conditioning.
        frobenius_for_interpolation_problem (bool): whether to solve the
            underdetermined quadratic interpolation problem by minimizing the Frobenius
            norm of the Hessian, or change in Hessian.
        use_restarts (bool): whether to do restarts when the lower bound on the trust
            region radius (:math:`\rho_k`) reaches the stopping criterion
            (:math:`\rho_{end}`), or (optionally) when all points are within noise
            level. Default is ``True`` if ``criterion_noisy`` or when
            ``seek_global_minimum``.
        max_unsuccessful_restarts (int): maximum number of consecutive unsuccessful
            restarts allowed (i.e. restarts which did not reduce the objective further)
        max_unsuccessful_restarts_total (int): number of total unsuccessful restarts
            allowed. Default is 20 if ``seek_global_optimum`` and else unrestricted.
        trust_region_rescaling_after_unsuccessful_restart (float): Factor by which to
            increase the initial trust region radius (:math:`\rho_{beg}`) after
            unsuccessful restarts. Default is 1.1 if ``seek_global_optimum`` else 1.
        trust_region_stop_criterion_scaling_after_restart (float): Factor with which
            the trust region stopping criterion is multiplied with at each restart.
        use_soft_restarts (bool): Whether to use soft or hard restarts.
        nr_points_to_move_at_soft_restart (int): Number of interpolation points to
            move at each soft restart.
        move_current_point_at_soft_restart (bool): Whether to move the current
            evaluation point ($x_k$) to the best new point evaluated.
        reuse_criterion_value_at_hard_restart (bool): whether or not to recycle the
            criterion value at the best iterate found when performing a hard restart.
            This saves one objective evaluation.
        max_nr_only_local_improvements_after_soft_restart (int):
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
        iterations_to_save_for_automatic_restart_detection (int):
            How many iterations of model changes and trust region radii to store.
        min_model_slope_increase_for_automatic_restart (float):
            Minimum rate of increase of $\log(\|g_k-g_{k-1}\|)$ and
            $\log(\|H_k-H_{k-1}\|_F)$ over past iterations to cause a restart.
        min_correlations_for_automatic_restart (float):
            Minimum correlation of the data sets $(k, \log(\|g_k-g_{k-1}\|))$ and
            $(k, \log(\|H_k-H_{k-1}\|_F))$ required to cause a restart.

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
    if absolute_criterion_value_stopping_criterion is None:
        absolute_criterion_value_stopping_criterion = (-np.inf,)

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
        "init.random_initial_directions": random_initial_directions,
        "init.random_directions_make_orthogonal": random_directions_orthogonal,
        "init.run_in_parallel": not intermediate_processing_of_initial_points,
        "general.rounding_error_constant": interpolation_rounding_error_constant,
        "general.safety_step_thresh": threshold_for_safety_step,
        "general.check_objfun_for_overflow": clip_criterion_if_overflowing,
        "tr_radius.eta1": min_improvement_for_successful_iteration,
        "tr_radius.eta2": threshold_for_very_succesful_iteration,
        "tr_radius.gamma_inc": trust_region_increase_factor_after_success,
        "tr_radius.gamma_inc_overline": trust_region_increase_factor_after_large_success,  # noqa E501
        "tr_radius.alpha1": trust_region_lower_bound_decrease_factor,
        "model.abs_tol": absolute_criterion_value_stopping_criterion,
        "slow.thresh_for_slow": threshold_for_insufficient_improvement,
        "slow.max_slow_iters": nr_of_insufficient_improvements_to_terminate,
        "slow.history_for_slow": comparison_period_for_insufficient_improvement,
        "noise.quit_on_noise_level": quit_when_trust_evaluations_within_noise_level,
        "noise.scale_factor_for_quit": noise_scale_factor_for_quit,
        "noise.multiplicative_noise_level": multiplicative_noise_level,
        "noise.additive_noise_level": additive_noise_level,
        "interpolation.precondition": scale_interpolation_system,
        "interpolation.minimum_change_hessian": frobenius_for_interpolation_problem,
        "restarts.use_restarts": use_restarts,
        "restarts.max_unsuccessful_restarts": max_unsuccessful_restarts,
        "restarts.max_unsuccessful_restarts_total": max_unsuccessful_restarts_total,
        "restarts.rhobeg_scale_after_unsuccessful_restart": trust_region_rescaling_after_unsuccessful_restart,  # noqa E501
        "restarts.rhoend_scale": trust_region_stop_criterion_scaling_after_restart,
        "restarts.use_soft_restarts": use_soft_restarts,
        "restarts.soft.num_geom_steps": nr_points_to_move_at_soft_restart,
        "restarts.soft.move_xk": move_current_point_at_soft_restart,
        "restarts.hard.use_old_fk": reuse_criterion_value_at_hard_restart,
        "restarts.soft.max_fake_successful_steps": max_nr_only_local_improvements_after_soft_restart,  # noqa: E501
        "restarts.auto_detect": additional_automatic_restart_detection,
        "restarts.auto_detect.history": iterations_to_save_for_automatic_restart_detection,  # noqa: E501
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
        nsamples=nr_evals_per_point,
        npt=nr_interpolation_points,
        rhoend=absolute_params_tolerance,
        seek_global_minimum=seek_global_minimum,
    )

    return _process_nag_result(res)


def _process_nag_result(nag_result_obj):
    processed = {
        "solution_x": nag_result_obj.x,
        "solution_criterion": nag_result_obj.f,
        "solution_derivative": nag_result_obj.gradient,
        "solution_hessian": nag_result_obj.hessian,
        "n_iterations": nag_result_obj.nf,
        "n_criterion_evaluations": nag_result_obj.nx,
        "message": nag_result_obj.msg,
        "success": nag_result_obj.flag == nag_result_obj.EXIT_SUCCESS,
        "reached_convergence_criterion": None,
    }
    return processed
