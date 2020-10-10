"""Implement algorithms by the (Numerical Algorithms Group)[https://www.nag.com/]."""
from functools import partial

import numpy as np

from estimagic.config import IS_DFOLS_INSTALLED
from estimagic.config import IS_PYBOBYQA_INSTALLED
from estimagic.optimization.algo_options import CLIP_CRITERION_IF_OVERFLOWING
from estimagic.optimization.algo_options import CONVERGENCE_NOISE_CRITERION
from estimagic.optimization.algo_options import CRITERION_NOISY
from estimagic.optimization.algo_options import FAST_START_OPTIONS
from estimagic.optimization.algo_options import INTERPOLATION_ROUNDING_ERROR
from estimagic.optimization.algo_options import MAX_CRITERION_EVALUATIONS
from estimagic.optimization.algo_options import RANDOM_DIRECTIONS_ORTHOGONAL
from estimagic.optimization.algo_options import RANDOM_INITIAL_DIRECTIONS
from estimagic.optimization.algo_options import RESTART_OPTIONS
from estimagic.optimization.algo_options import SCALE_INTERPOLATION_SYSTEM
from estimagic.optimization.algo_options import SECOND_BEST_ABSOLUTE_PARAMS_TOLERANCE
from estimagic.optimization.algo_options import SLOW_IMPROVEMENT_TOLERANCE
from estimagic.optimization.algo_options import THRESHOLD_FOR_SAFETY_STEP
from estimagic.optimization.algo_options import TRUST_REGION_OPTIONS
from estimagic.optimization.utilities import calculate_trustregion_initial_radius

if IS_PYBOBYQA_INSTALLED:
    import pybobyqa

if IS_DFOLS_INSTALLED:
    import dfols


def nag_dfols(
    criterion_and_derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    criterion_noisy=CRITERION_NOISY,
    noise_n_evals_per_point=None,
    clip_criterion_if_overflowing=CLIP_CRITERION_IF_OVERFLOWING,
    max_criterion_evaluations=MAX_CRITERION_EVALUATIONS,
    absolute_params_tolerance=SECOND_BEST_ABSOLUTE_PARAMS_TOLERANCE,
    trustregion_initial_radius=None,
    random_initial_directions=RANDOM_INITIAL_DIRECTIONS,
    random_directions_orthogonal=RANDOM_DIRECTIONS_ORTHOGONAL,
    n_interpolation_points=None,
    interpolation_rounding_error=INTERPOLATION_ROUNDING_ERROR,
    threshold_for_safety_step=THRESHOLD_FOR_SAFETY_STEP,
    scale_interpolation_system=SCALE_INTERPOLATION_SYSTEM,
    trust_region_options=None,
    convergence_noise_criterion=CONVERGENCE_NOISE_CRITERION,
    restart_options=None,
    slow_improvement_tolerance=None,
    relative_to_start_value_criterion_tolerance=0.0,
    n_extra_points_to_move_when_sufficient_improvement=0,
    use_momentum_method_to_move_extra_points=False,
    n_increase_move_points_at_restart=0,
    fast_start_options=None,
):
    r"""Minimize a function with least squares structure using DFO-LS.

    The DFO-LS algorithm :cite:`Cartis2018b` is designed to solve the nonlinear
    least-squares minimization problem (with optional bound constraints)

    .. math::

        \min_{x\in\mathbb{R}^n}  &\quad  f(x) := \sum_{i=1}^{m}r_{i}(x)^2 \\
        \text{s.t.} &\quad  \text{lower_bounds} \leq x \leq \text{upper_bounds}

    The :math:`r_{i}` are called root contributions in estimagic.

    DFO-LS is a derivative-free optimization algorithm, which means it does not require
    the user to provide the derivatives of f(x) or :math:`r_{i}(x)`, nor does it
    attempt to estimate them internally (by using finite differencing, for instance).

    There are two main situations when using a derivative-free algorithm
    (such as DFO-LS) is preferable to a derivative-based algorithm (which is the vast
    majority of least-squares solvers):

    1. If the residuals are noisy, then calculating or even estimating their derivatives
       may be impossible (or at least very inaccurate). By noisy, we mean that if we
       evaluate :math:`r_{i}(x)` multiple times at the same value of x, we get different
       results. This may happen when a Monte Carlo simulation is used, for instance.

    2. If the residuals are expensive to evaluate, then estimating derivatives
       (which requires n evaluations of each :math:`r_{i}(x)` for every point of
       interest x) may be prohibitively expensive. Derivative-free methods are designed
       to solve the problem with the fewest number of evaluations of the criterion as
       possible.

    To read the detailed documentation of the algorithm `click here
    <https://numericalalgorithmsgroup.github.io/dfols/>`_.

    There are four possible convergence criteria:

    1. when the trust region radius is reduced below a minimum
       (``absolute parameter tolerance``).

    2. when the improvements of iterations become very small. This is very similar to
       ``relative_criterion_tolerance`` but ``slow_improvement_tolerance`` is more
       general allowing to specify not only the threshold for convergence but also
       a period over which the improvements must have been very small.

    3. when a sufficient reduction to the criterion value at the start parameters
       has been reached, i.e. when
       :math:`\frac{f(x)}{f(x_0)} \leq
       \text{relative_to_start_value_criterion_tolerance}`

    4. when all evaluations on the trust region points fall within a scaled version of
       the noise level of the criterion function. This is only applicable if the
       criterion function is noisy. You can specify this criterion with
       ``convergence_noise_criterion``.

    DF-OLS supports restarting the optimization and dynamically growing the initial set.
    For more information see
    `their detailed documentation <https://numericalalgorithmsgroup.github.io/dfols/>`_
    and :cite:`Cartis2018b`.

    Args:
        criterion_noisy (bool): Whether the criterion function is noisy, i.e. whether
            it does not always return the same value when evaluated at the same
            parameters.
        noise_n_evals_per_point (callable): How often to evaluate the criterion
            function at each point.
            This is only applicable for criterion functions with stochastic noise,
            when averaging multiple evaluations at the same point produces a more
            accurate value.
            The input parameters are the ``trust_region_radius`` (:math:`\Delta`),
            the ``min_trust_region_radius`` (:math:`\rho`),
            how many iterations the algorithm has been running for, ``n_iterations``
            and how many restarts have been performed, ``n_restarts``.
            The function must return an integer.
            Default is no averaging (i.e. ``noise_n_evals_per_point(trust_region_radius,
            min_trust_region_radius, n_iterations, n_restarts) = 1``).
        clip_criterion_if_overflowing (bool): Whether to clip the criterion if it would
            raise an ``OverflowError`` otherwise.
        max_criterion_evaluations (int): If the maximum number of function evaluation is
            reached, the optimization stops but we do not count this as convergence.
        absolute_params_tolerance (float): Minimum allowed value of the trust region
            radius, which is one criterion for successful termination.
        trustregion_initial_radius (float): Initial value of the trust region radius.
        random_initial_directions (bool): Whether to draw the initial directions
            randomly (as opposed to coordinate directions).
        random_directions_orthogonal (bool): Whether to make random initial directions
            orthogonal.
        n_interpolation_points (int): The number of interpolation points to use.
            The default is :code:`len(x) + 1`. If using restarts, this is the
            number of points to use in the first run of the solver, before any restarts.
        interpolation_rounding_error (float): Internally, all interpolation
            points are stored with respect to a base point $x_b$; that is,
            DF-OLS stores $\{y_t-x_b\}$, which reduces the risk of roundoff
            errors. We shift $x_b$ to $x_k$ when
            :math:`\|s_k\| \leq
            \text{interpolation_rounding_error} \cdot \|x_k-x_b\|`, where
            :math:`\|s_k\|` is the proposed step size.
        threshold_for_safety_step (float): Threshold for when to call the safety step,
            :math:`\|s_k\| \leq \text{threshold_for_safety_step} \cdot \rho_k`, where
            :math:`\|s_k\|` is the proposed step size.
        trust_region_options (dict): Options how the trust region is contracted and
            expanded.
        slow_improvement_tolerance (dict): Arguments for converging when the evaluations
            over several iterations only yield small improvements on average.
        convergence_noise_criterion (dict): Arguments for converging when the
            evaluations in the trust region all fall within a scaled version of the
            noise at the point of interest.
        scale_interpolation_system (bool): Whether or not to scale the interpolation
            system to improve conditioning.
        relative_to_start_value_criterion_tolerance (float):
            Terminate if a point is reached where the ratio of the criterion value
            to the criterion value at the start params is below this value, i.e. if
            :math:`f(x_k)/f(x_0) \leq
            \text{relative_to_start_value_criterion_tolerance}`. Note this is
            deactivated unless the lowest mathematically possible criterion value (0.0)
            is actually achieved.
        n_extra_points_to_move_when_sufficient_improvement (int): The number of extra
            points (other than accepting the trust region step) to move. Useful when
            ``n_interpolation_points > len(x) + 1``.
        n_increase_move_points_at_restart (int): The number by which to increase
            ``n_extra_points_to_move_when_sufficient_improvement`` at each restart.
        use_momentum_method_to_move_extra_points (bool): If moving extra points in
            ``n_extra_points_to_move_when_sufficient_improvement`` at each restart.
            successful iterations, whether to use the 'momentum' method. If not,
            uses geometry-improving steps.
        fast_start_options (dict): Options to start the optimization while building the
            full size of the trust region model.

    Returns:
        results (dict): See :ref:`internal_optimizer_output` for details.

    """
    if not IS_DFOLS_INSTALLED:
        raise NotImplementedError(
            "The dfols package is not installed and required for 'nag_dfols'. "
            "You can install it with 'pip install DFO-LS'. "
            "For additional installation instructions visit: ",
            r"https://numericalalgorithmsgroup.github.io/dfols/build/html/install.html",
        )

    algo_info = {
        "name": "nag_dfols",
        "primary_criterion_entry": "root_contributions",
        "parallelizes": False,
        "needs_scaling": False,
    }

    advanced_options, restart_options = _create_nag_advanced_options(
        x=x,
        trustregion_initial_radius=trustregion_initial_radius,
        noise_n_evals_per_point=noise_n_evals_per_point,
        convergence_noise_criterion=convergence_noise_criterion,
        restart_options=restart_options,
        slow_improvement_tolerance=slow_improvement_tolerance,
        interpolation_rounding_error=interpolation_rounding_error,
        threshold_for_safety_step=threshold_for_safety_step,
        clip_criterion_if_overflowing=clip_criterion_if_overflowing,
        random_initial_directions=random_initial_directions,
        random_directions_orthogonal=random_directions_orthogonal,
        scale_interpolation_system=scale_interpolation_system,
        trust_region_options=trust_region_options,
    )

    fast_start_options = _build_options_dict(
        user_input=fast_start_options, default_options=FAST_START_OPTIONS,
    )
    if not fast_start_options["step_type"] in ["safety", "regular"]:
        raise ValueError(
            "step_type of the fast_start_options must be 'safety' or 'regular'"
        )
    if (
        fast_start_options["shrink_upper_radius_in_safety_steps"]
        and fast_start_options["full_geometry_improving_step"]
    ):
        raise ValueError(
            "full_geometry_improving_step of the fast_start_options can only be True "
            "if shrink_upper_radius_in_safety_steps is False."
        )
    (
        perturb_jacobian,
        perturb_trust_region,
    ) = _get_fast_start_strategy_from_user_value(fast_start_options["strategy"])
    if (
        restart_options["n_extra_interpolation_points_per_soft_reset"]
        < restart_options["n_extra_interpolation_points_per_soft_reset"]
    ):
        raise ValueError(
            "In the restart options 'n_extra_interpolation_points_per_soft_reset must "
            "be larger or the same as n_extra_interpolation_points_per_hard_reset."
        )

    dfols_options = {
        "growing.full_rank.use_full_rank_interp": perturb_jacobian,
        "growing.perturb_trust_region_step": perturb_trust_region,
        "restarts.hard.use_old_rk": restart_options["reuse_criterion_value_at_hard"],
        "restarts.auto_detect.min_chgJ_slope": restart_options[
            "min_model_slope_increase_for_automatic_detection"
        ],  # noqa: E501
        "restarts.max_npt": restart_options["max_interpolation_points"],
        "restarts.increase_npt": restart_options[
            "n_extra_interpolation_points_per_soft_reset"
        ]
        > 0,
        "restarts.increase_npt_amt": restart_options[
            "n_extra_interpolation_points_per_soft_reset"
        ],
        "restarts.hard.increase_ndirs_initial_amt": restart_options[
            "n_extra_interpolation_points_per_hard_reset"
        ]
        - restart_options["n_extra_interpolation_points_per_soft_reset"],
        "model.rel_tol": relative_to_start_value_criterion_tolerance,
        "regression.num_extra_steps": n_extra_points_to_move_when_sufficient_improvement,  # noqa: E501
        "regression.momentum_extra_steps": use_momentum_method_to_move_extra_points,
        "regression.increase_num_extra_steps_with_restart": n_increase_move_points_at_restart,  # noqa: E501
        "growing.ndirs_initial": fast_start_options["min_inital_points"],
        "growing.delta_scale_new_dirns": fast_start_options[
            "scaling_of_trust_region_step_perturbation"
        ],
        "growing.full_rank.scale_factor": fast_start_options[
            "scaling_jacobian_perturb_components"
        ],
        "growing.full_rank.min_sing_val": fast_start_options[
            "jacobian_perturb_abs_floor_for_singular_values"
        ],  # noqa: E501
        "growing.full_rank.svd_max_jac_cond": fast_start_options[
            "jacobian_perturb_max_condition_number"
        ],
        "growing.do_geom_steps": fast_start_options["step_type"] == "regular",
        "growing.safety.do_safety_step": fast_start_options["step_type"] == "safety",
        "growing.safety.reduce_delta": fast_start_options[
            "shrink_upper_radius_in_safety_steps"
        ],
        "growing.safety.full_geom_step": fast_start_options[
            "full_geometry_improving_step"
        ],
        "growing.reset_delta": fast_start_options["reset_trust_region_radius_after"],
        "growing.reset_rho": fast_start_options["reset_min_trust_region_radius_after"],
        "growing.gamma_dec": fast_start_options["trust_region_decrease"],
        "growing.num_new_dirns_each_iter": fast_start_options[
            "n_search_directions_to_add_when_incomplete"
        ],
    }

    advanced_options.update(dfols_options)

    criterion = partial(
        criterion_and_derivative, task="criterion", algorithm_info=algo_info
    )

    res = dfols.solve(
        criterion,
        x0=x,
        bounds=(lower_bounds, upper_bounds),
        maxfun=max_criterion_evaluations,
        rhobeg=trustregion_initial_radius,
        npt=n_interpolation_points,
        rhoend=absolute_params_tolerance,
        nsamples=noise_n_evals_per_point,
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
    criterion_noisy=CRITERION_NOISY,
    clip_criterion_if_overflowing=CLIP_CRITERION_IF_OVERFLOWING,
    restart_options=None,
    seek_global_optimum=False,
    max_criterion_evaluations=MAX_CRITERION_EVALUATIONS,
    absolute_params_tolerance=SECOND_BEST_ABSOLUTE_PARAMS_TOLERANCE,
    absolute_criterion_value_tolerance=None,
    convergence_noise_criterion=None,
    slow_improvement_tolerance=None,
    noise_n_evals_per_point=None,
    trustregion_initial_radius=None,
    random_initial_directions=RANDOM_INITIAL_DIRECTIONS,
    random_directions_orthogonal=RANDOM_DIRECTIONS_ORTHOGONAL,
    n_interpolation_points=None,
    interpolation_rounding_error=INTERPOLATION_ROUNDING_ERROR,
    threshold_for_safety_step=THRESHOLD_FOR_SAFETY_STEP,
    scale_interpolation_system=SCALE_INTERPOLATION_SYSTEM,
    trust_region_options=None,
    frobenius_for_interpolation_problem=True,
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
    <https://numericalalgorithmsgroup.github.io/pybobyqa/>`_.

    There are four possible convergence criteria:

    1. when the trust region radius is reduced below a minimum. This is
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
        trustregion_initial_radius (float): Initial value of the trust region radius.
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
        noise_n_evals_per_point (callable): How often to evaluate the criterion
            function at each point.
            This is only applicable for criterion functions with stochastic noise,
            when averaging multiple evaluations at the same point produces a more
            accurate value.
            The input parameters are the ``trust_region_radius`` (``delta``),
            the ``min_trust_region_radius`` (``rho``),
            how many iterations the algorithm has been running for, ``n_iterations``
            and how many restarts have been performed, ``n_restarts``.
            The function must return an integer.
            Default is no averaging (i.e. ``noise_n_evals_per_point(...) = 1``).
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
        trust_region_options (dict): Options how the trust region is contracted and
            expanded.
        clip_criterion_if_overflowing (bool): Whether to clip the criterion if it would
            raise an ``OverflowError`` otherwise.
        absolute_criterion_value_tolerance (float): Terminate successfully if
            the criterion value falls below this threshold. This is deactivated
            (i.e. set to -inf) by default.
        convergence_noise_criterion (dict): Arguments for converging when the
            evaluations in the trust region all fall within a scaled version of the
            noise at the point of interest.
        slow_improvement_tolerance (dict): Arguments for converging when the evaluations
            over several iterations only yield small improvements on average.
        scale_interpolation_system (bool): Whether or not to scale the interpolation
            system to improve conditioning.
        frobenius_for_interpolation_problem (bool): Whether to solve the
            underdetermined quadratic interpolation problem by minimizing the Frobenius
            norm of the Hessian, or change in Hessian.
        restart_options (dict): Options for restarting the optimization.

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

    if absolute_criterion_value_tolerance is None:
        absolute_criterion_value_tolerance = -np.inf

    algo_info = {
        "name": "nag_pybobyqa",
        "primary_criterion_entry": "value",
        "parallelizes": False,
        "needs_scaling": False,
    }
    criterion = partial(
        criterion_and_derivative, task="criterion", algorithm_info=algo_info
    )
    advanced_options, restart_options = _create_nag_advanced_options(
        x=x,
        trustregion_initial_radius=trustregion_initial_radius,
        noise_n_evals_per_point=noise_n_evals_per_point,
        convergence_noise_criterion=convergence_noise_criterion,
        restart_options=restart_options,
        slow_improvement_tolerance=slow_improvement_tolerance,
        interpolation_rounding_error=interpolation_rounding_error,
        threshold_for_safety_step=threshold_for_safety_step,
        clip_criterion_if_overflowing=clip_criterion_if_overflowing,
        random_initial_directions=random_initial_directions,
        random_directions_orthogonal=random_directions_orthogonal,
        scale_interpolation_system=scale_interpolation_system,
        trust_region_options=trust_region_options,
    )

    pybobyqa_options = {
        "model.abs_tol": absolute_criterion_value_tolerance,
        "interpolation.minimum_change_hessian": frobenius_for_interpolation_problem,
        "restarts.max_unsuccessful_restarts_total": restart_options[
            "max_unsuccessful_total"
        ],
        "restarts.rhobeg_scale_after_unsuccessful_restart": restart_options[
            "trust_region_scaling_after_unsuccessful"
        ],  # noqa E501
        "restarts.hard.use_old_fk": restart_options["reuse_criterion_value_at_hard"],
        "restarts.auto_detect.min_chg_model_slope": restart_options[
            "min_model_slope_increase_for_automatic_detection"
        ],  # noqa: E501
    }

    advanced_options.update(pybobyqa_options)

    res = pybobyqa.solve(
        criterion,
        x0=x,
        bounds=(lower_bounds, upper_bounds),
        maxfun=max_criterion_evaluations,
        rhobeg=trustregion_initial_radius,
        user_params=advanced_options,
        scaling_within_bounds=False,
        do_logging=False,
        print_progress=False,
        objfun_has_noise=criterion_noisy,
        nsamples=noise_n_evals_per_point,
        npt=n_interpolation_points,
        rhoend=absolute_params_tolerance,
        seek_global_minimum=seek_global_optimum,
    )

    return _process_nag_result(res, len(x))


def _process_nag_result(nag_result_obj, len_x):
    """Convert the NAG result object to our result dictionary.

    Args:
        nag_result_obj: NAG result object
        len_x (int): length of the supplied parameters, i.e. the dimensionality of the
            problem.


    Returns:
        results (dict): See :ref:`internal_optimizer_output` for details.
    """
    processed = {
        "solution_criterion": nag_result_obj.f,
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


def _create_nag_advanced_options(
    x,
    trustregion_initial_radius,
    noise_n_evals_per_point,
    convergence_noise_criterion,
    restart_options,
    slow_improvement_tolerance,
    interpolation_rounding_error,
    threshold_for_safety_step,
    clip_criterion_if_overflowing,
    random_initial_directions,
    random_directions_orthogonal,
    scale_interpolation_system,
    trust_region_options,
):
    if trustregion_initial_radius is None:
        trustregion_initial_radius = calculate_trustregion_initial_radius(x)
    # -np.inf as a default leads to errors when building the documentation with sphinx.
    noise_n_evals_per_point = _change_evals_per_point_interface(noise_n_evals_per_point)
    convergence_noise_criterion = _build_options_dict(
        user_input=convergence_noise_criterion,
        default_options=CONVERGENCE_NOISE_CRITERION,
    )
    trust_region_options = _build_options_dict(
        user_input=trust_region_options, default_options=TRUST_REGION_OPTIONS,
    )
    restart_options = _build_options_dict(
        user_input=restart_options, default_options=RESTART_OPTIONS,
    )
    slow_improvement_tolerance = _build_options_dict(
        user_input=slow_improvement_tolerance,
        default_options=SLOW_IMPROVEMENT_TOLERANCE,
    )

    advanced_options = {
        "general.rounding_error_constant": interpolation_rounding_error,
        "general.safety_step_thresh": threshold_for_safety_step,
        "general.check_objfun_for_overflow": clip_criterion_if_overflowing,
        "init.random_initial_directions": random_initial_directions,
        "init.random_directions_make_orthogonal": random_directions_orthogonal,
        "tr_radius.eta1": trust_region_options["threshold_successful"],
        "tr_radius.eta2": trust_region_options["threshold_very_successful"],
        "tr_radius.gamma_dec": trust_region_options["reduction_when_not_successful"],
        "tr_radius.gamma_inc": trust_region_options["increase_after_success"],
        "tr_radius.gamma_inc_overline": trust_region_options[
            "increase_after_large_success"
        ],
        "tr_radius.alpha1": trust_region_options["min_decrease"],
        "tr_radius.alpha2": trust_region_options["update_from_min_trust_region"],
        "general.rounding_error_constant": interpolation_rounding_error,
        "general.safety_step_thresh": threshold_for_safety_step,
        "general.check_objfun_for_overflow": clip_criterion_if_overflowing,
        "init.random_initial_directions": random_initial_directions,
        "init.random_directions_make_orthogonal": random_directions_orthogonal,
        "slow.thresh_for_slow": slow_improvement_tolerance[
            "threshold_for_insufficient_improvement"
        ],
        "slow.max_slow_iters": slow_improvement_tolerance[
            "n_insufficient_improvements_until_terminate"
        ],
        "slow.history_for_slow": slow_improvement_tolerance[
            "comparison_period_for_insufficient_improvement"
        ],
        "noise.quit_on_noise_level": convergence_noise_criterion["active"],
        "noise.scale_factor_for_quit": convergence_noise_criterion[
            "noise_scale_factor_for_quit"
        ],
        "noise.multiplicative_noise_level": convergence_noise_criterion[
            "multiplicative_noise_level"
        ],
        "noise.additive_noise_level": convergence_noise_criterion[
            "additive_noise_level"
        ],
        "interpolation.precondition": scale_interpolation_system,
        "restarts.use_restarts": restart_options["use_restarts"],
        "restarts.max_unsuccessful_restarts": restart_options["max_unsuccessful"],
        "restarts.rhoend_scale": restart_options["min_trust_region_scaling_after"],
        "restarts.use_soft_restarts": restart_options["use_soft"],
        "restarts.soft.move_xk": restart_options["move_current_point_at_soft"],
        "restarts.soft.max_fake_successful_steps": restart_options[
            "max_iterations_without_new_best_after_soft"
        ],  # noqa: E501
        "restarts.auto_detect": restart_options["automatic_detection"],
        "restarts.auto_detect.history": restart_options[
            "n_iterations_for_automatc_detection"
        ],  # noqa: E501
        "restarts.auto_detect.min_correl": restart_options[
            "min_correlations_for_automatic_detection"
        ],
        "restarts.soft.num_geom_steps": restart_options["points_to_move_at_soft"],
    }

    return advanced_options, restart_options


def _change_evals_per_point_interface(func):
    """Change the interface of the user supplied function to the one expected
    by NAG.

    Args:
        func (callable or None): function mapping from our names to
            noise_n_evals_per_point.

    Returns:
        adjusted_noise_n_evals_per_point (callable): function mapping from the
            argument names expected by pybobyqa and df-ols to noise_n_evals_per_point.
    """
    if func is not None:

        def adjusted_noise_n_evals_per_point(delta, rho, iter, nrestarts):  # noqa: A002
            return func(
                trust_region_radius=delta,
                min_trust_region=rho,
                n_iterations=iter,
                n_restarts=nrestarts,
            )

        return adjusted_noise_n_evals_per_point


def _build_options_dict(user_input, default_options):
    """Create the full dictionary of trust region fast start options from user input.

    Args:
        user_input (dict or None): dictionary to update the default options with.
            May only contain keys present in the default options.
        default_options (dict): the default values.

    Returns:
        full_options (dict)

    """
    full_options = default_options.copy()
    user_input = {} if user_input is None else user_input
    invalid = [x for x in user_input if x not in full_options]
    if len(invalid) > 0:
        raise ValueError(
            f"You specified illegal options {', '.join(invalid)}. Allowed are: "
            ", ".join(full_options.keys())
        )
    full_options.update(user_input)
    return full_options


def _get_fast_start_strategy_from_user_value(user_value):
    """Get perturb_jacobian and perturb_trust_region_step from user value."""
    allowed_perturb_values = ["auto", "jacobian", "trust_region"]
    if user_value not in allowed_perturb_values:
        raise ValueError(
            "`perturb_jacobian_or_trust_region_step` must be one of "
            f"{allowed_perturb_values}. You provided {user_value}."
        )
    if user_value == "auto":
        perturb_jacobian = None
        perturb_trust_region_step = None
    else:
        perturb_jacobian = user_value == "jacobian"
        perturb_trust_region_step = not perturb_jacobian

    return perturb_jacobian, perturb_trust_region_step
