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
    trust_region_increase_factor_after_success=2.0,
    trust_region_increase_factor_after_large_success=4.0,
    trust_region_lower_bound_decrease_factor=None,
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

    Args:
        absolute_params_tolerance (float): minimum allowed value of the trust region
            radius, which determines when a successful termination occurs.
        max_criterion_evaluations (int): If the maximum number of function evaluation is
            reached, the optimization stops but we do not count this as convergence.
        initial_trust_region_radius (float): initial value of the trust region radius.
        seek_global_minimum (bool): Whether to apply the heuristic to escape local
            minima presented in :cite:`Cartis2018a`. Only applies for noisy criterion
            functions.r
        random_initial_directions (bool): If True, the initial directions are drawn
            randomly (as opposed to coordinate directions).
        random_directions_orthogonal (bool): If True and random initial directions are
            drawn, the drawn directions are made orthogonal.
        nr_interpolation_points (int): the number of interpolation points to use.
            default is $ 2n+1 $ for a problem with $ len(x)=n $ if not
            ``criterion_noisy``. Otherwise it is set to $ (n+1)(n+2)/2) $.
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
            points are stored with respect to a base point :math:`x_b`; that is,
            pybobyqa stores :math:`\{y_t-x_b\}`, which reduces the risk of roundoff
            errors. We shift :math:`x_b` to :math:`x_k` when
            :math:`\|s_k\| \leq \text{interpolation_rounding_error_constant}\|x_k-x_b\|`
        threshold_for_safety_step (float): Threshold for when to call the safety step,
            :math:`\|s_k\| \leq \text{threshold_for_safety_step} \rho_k`.
        intermediate_processing_of_initial_points (bool): If using random directions,
            whether to do intermediate processing between point evaluations.
        min_improvement_for_successful_iteration (float): minimum share of the predicted
            improvement that has to be realized for an iteration to count as successful.
        threshold_for_very_succesful_iteration (float): share of predicted improvement
            that has to be surpassed for an iteration to count as very successful.
        trust_region_reduction_ratio_when_not_successful (float): ratio by which to
            decrease the trust region radius when realized improvement does not match
            the min_improvement_for_successful_iteration. Default is 0.5 for
            deterministic problems or 0.98 if ``criterion_noisy``.
        clip_criterion_if_overflowing (bool): Whether to clip the criterion if it would
            raise an OverflowError otherwise.
        trust_region_increase_factor_after_success (float): Ratio by which to increase
            the trust region radius :math:`\Delta_k` in very successful iterations
            (:math:`\gamma_{inc}`).
        trust_region_increase_factor_after_large_success (float):
            Ratio of the proposed step (:math:`\|s_k\|`) by which to increase the
            trust region radius (:math:`\Delta_k`) in very successful iterations
            (:math:`\overline{\gamma}_{inc}`).
        trust_region_lower_bound_decrease_factor (float):
            Ratio to decrease the lower bound on the trust region radius
            (:math:`\rho_k`) by when it is reduced (:math:`\alpha_1`).
            Default is 0.9 if ``criterion_noisy`` and 0.1 else.
        trust_region_update_from_lower_bound_to_trust_region_factor (float):
            Ratio of the lower bound on the trust region (:math:`\rho_k`) to decrease
            the actual trust region radius (:math:`\Delta_k`) by when the lower bound
            is reduced (:math:`\alpha_2`). Default is 0.95 if ``criterion_noisy`` and
            0.5 else.

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
        # we do not want the optimizer to stop at some absolute value
        "model.abs_tol": -np.inf,
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
