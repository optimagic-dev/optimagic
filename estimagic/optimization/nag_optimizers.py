"""Implement algorithms by the (Numerical Algorithms Group)[https://www.nag.com/]."""
import warnings
from functools import partial

import numpy as np

from estimagic.config import IS_DFOLS_INSTALLED
from estimagic.config import IS_PYBOBYQA_INSTALLED
from estimagic.optimization.algo_options import CLIP_CRITERION_IF_OVERFLOWING
from estimagic.optimization.algo_options import (
    CONVERGENCE_MINIMAL_TRUSTREGION_RADIUS_TOLERANCE,
)
from estimagic.optimization.algo_options import (
    CONVERGENCE_NOISE_CORRECTED_CRITERION_TOLERANCE,
)
from estimagic.optimization.algo_options import CONVERGENCE_SLOW_PROGRESS
from estimagic.optimization.algo_options import INITIAL_DIRECTIONS
from estimagic.optimization.algo_options import INTERPOLATION_ROUNDING_ERROR
from estimagic.optimization.algo_options import RANDOM_DIRECTIONS_ORTHOGONAL
from estimagic.optimization.algo_options import RESET_OPTIONS
from estimagic.optimization.algo_options import STOPPING_MAX_CRITERION_EVALUATIONS
from estimagic.optimization.algo_options import THRESHOLD_FOR_SAFETY_STEP
from estimagic.optimization.algo_options import TRUSTREGION_EXPANSION_FACTOR_SUCCESSFUL
from estimagic.optimization.algo_options import (
    TRUSTREGION_EXPANSION_FACTOR_VERY_SUCCESSFUL,
)
from estimagic.optimization.algo_options import TRUSTREGION_FAST_START_OPTIONS
from estimagic.optimization.algo_options import TRUSTREGION_PRECONDITION_INTERPOLATION
from estimagic.optimization.algo_options import (
    TRUSTREGION_SHRINKING_FACTOR_LOWER_RADIUS,
)
from estimagic.optimization.algo_options import (
    TRUSTREGION_SHRINKING_FACTOR_NOT_SUCCESSFUL,
)
from estimagic.optimization.algo_options import (
    TRUSTREGION_SHRINKING_FACTOR_UPPER_RADIUS,
)
from estimagic.optimization.algo_options import TRUSTREGION_THRESHOLD_SUCCESSFUL
from estimagic.optimization.algo_options import TRUSTREGION_THRESHOLD_VERY_SUCCESSFUL
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
    clip_criterion_if_overflowing=CLIP_CRITERION_IF_OVERFLOWING,
    convergence_minimal_trustregion_radius_tolerance=CONVERGENCE_MINIMAL_TRUSTREGION_RADIUS_TOLERANCE,  # noqa: E501
    convergence_noise_corrected_criterion_tolerance=CONVERGENCE_NOISE_CORRECTED_CRITERION_TOLERANCE,  # noqa: E501
    convergence_scaled_criterion_tolerance=0.0,
    convergence_slow_progress=None,
    initial_directions=INITIAL_DIRECTIONS,
    interpolation_rounding_error=INTERPOLATION_ROUNDING_ERROR,
    noise_additive_level=None,
    noise_multiplicative_level=None,
    noise_n_evals_per_point=None,
    random_directions_orthogonal=RANDOM_DIRECTIONS_ORTHOGONAL,
    stopping_max_criterion_evaluations=STOPPING_MAX_CRITERION_EVALUATIONS,
    threshold_for_safety_step=THRESHOLD_FOR_SAFETY_STEP,
    trustregion_expansion_factor_successful=TRUSTREGION_EXPANSION_FACTOR_SUCCESSFUL,
    trustregion_expansion_factor_very_successful=TRUSTREGION_EXPANSION_FACTOR_VERY_SUCCESSFUL,  # noqa: E501
    trustregion_fast_start_options=None,
    trustregion_initial_radius=None,
    trustregion_method_to_replace_extra_points="geometry_improving",
    trustregion_n_extra_points_to_replace_successful=0,
    trustregion_n_interpolation_points=None,
    trustregion_precondition_interpolation=TRUSTREGION_PRECONDITION_INTERPOLATION,
    trustregion_reset_options=None,
    trustregion_shrinking_factor_not_successful=TRUSTREGION_SHRINKING_FACTOR_NOT_SUCCESSFUL,  # noqa: E501
    trustregion_shrinking_factor_lower_radius=TRUSTREGION_SHRINKING_FACTOR_LOWER_RADIUS,
    trustregion_shrinking_factor_upper_radius=TRUSTREGION_SHRINKING_FACTOR_UPPER_RADIUS,
    trustregion_threshold_successful=TRUSTREGION_THRESHOLD_SUCCESSFUL,
    trustregion_threshold_very_successful=TRUSTREGION_THRESHOLD_VERY_SUCCESSFUL,
):
    r"""Minimize a function with least squares structure using DFO-LS.

    Do not call this function directly but pass its name "nag_dfols" to estimagic's
    maximize or minimize function as `algorithm` argument. Specify your desired
    arguments as a dictionary and pass them as `algo_options` to minimize or
    maximize.

    The DFO-LS algorithm :cite:`Cartis2018b` is designed to solve the nonlinear
    least-squares minimization problem (with optional bound constraints).
    Remember to cite :cite:`Cartis2018b` when using DF-OLS in addition to estimagic.

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

    1. when the lower trust region radius is shrunk below a minimum
       (``convergence_minimal_trustregion_radius_tolerance``).

    2. when the improvements of iterations become very small
       (``convergence_slow_progress``). This is very similar to
       ``relative_criterion_tolerance`` but ``convergence_slow_progress`` is more
       general allowing to specify not only the threshold for convergence but also
       a period over which the improvements must have been very small.

    3. when a sufficient reduction to the criterion value at the start parameters
       has been reached, i.e. when
       :math:`\frac{f(x)}{f(x_0)} \leq
       \text{convergence_scaled_criterion_tolerance}`

    4. when all evaluations on the interpolation points fall within a scaled version of
       the noise level of the criterion function. This is only applicable if the
       criterion function is noisy. You can specify this criterion with
       ``convergence_noise_corrected_criterion_tolerance``.

    DF-OLS supports resetting the optimization and doing a fast start by
    starting with a smaller interpolation set and growing it dynamically.
    For more information see `their detailed documentation
    <https://numericalalgorithmsgroup.github.io/dfols/>`_ and :cite:`Cartis2018b`.

    Args:
        clip_criterion_if_overflowing (bool): see :ref:`algo_options`.
        convergence_minimal_trustregion_radius_tolerance (float): see
            :ref:`algo_options`.
        convergence_noise_corrected_criterion_tolerance (float): Stop when the
            evaluations on the set of interpolation points all fall within this factor
            of the noise level.
            The default is 1, i.e. when all evaluations are within the noise level.
            If you want to not use this criterion but still flag your
            criterion function as noisy, set this tolerance to 0.0.

            .. warning::
                Very small values, as in most other tolerances don't make sense here.

        convergence_scaled_criterion_tolerance (float):
            Terminate if a point is reached where the ratio of the criterion value
            to the criterion value at the start params is below this value, i.e. if
            :math:`f(x_k)/f(x_0) \leq
            \text{convergence_scaled_criterion_tolerance}`. Note this is
            deactivated unless the lowest mathematically possible criterion value (0.0)
            is actually achieved.
        convergence_slow_progress (dict): Arguments for converging when the evaluations
            over several iterations only yield small improvements on average, see
            see :ref:`algo_options` for details.
        initial_directions (str): see :ref:`algo_options`.
        interpolation_rounding_error (float): see :ref:`algo_options`.
        noise_additive_level (float): Used for determining the presence of noise
            and the convergence by all interpolation points being within noise level.
            0 means no additive noise. Only multiplicative or additive is supported.
        noise_multiplicative_level (float): Used for determining the presence of noise
            and the convergence by all interpolation points being within noise level.
            0 means no multiplicative noise. Only multiplicative or additive is
            supported.
        noise_n_evals_per_point (callable): How often to evaluate the criterion
            function at each point.
            This is only applicable for criterion functions with noise,
            when averaging multiple evaluations at the same point produces a more
            accurate value.
            The input parameters are the ``upper_trustregion_radius`` (:math:`\Delta`),
            the ``lower_trustregion_radius`` (:math:`\rho`),
            how many iterations the algorithm has been running for, ``n_iterations``
            and how many resets have been performed, ``n_resets``.
            The function must return an integer.
            Default is no averaging (i.e.
            ``noise_n_evals_per_point(...) = 1``).
        random_directions_orthogonal (bool): see :ref:`algo_options`.
        stopping_max_criterion_evaluations (int): see :ref:`algo_options`.
        threshold_for_safety_step (float): see :ref:`algo_options`.
        trustregion_expansion_factor_successful (float): see :ref:`algo_options`.
        trustregion_expansion_factor_very_successful (float): see :ref:`algo_options`.
        trustregion_fast_start_options (dict): see :ref:`algo_options`.
        trustregion_initial_radius (float): Initial value of the trust region radius.
        trustregion_method_to_replace_extra_points (str): If replacing extra points in
            successful iterations, whether to use geometry improving steps or the
            momentum method. Can be "geometry_improving" or "momentum".
        trustregion_n_extra_points_to_replace_successful (int): The number of extra
            points (other than accepting the trust region step) to replace. Useful when
            ``trustregion_n_interpolation_points > len(x) + 1``.
        trustregion_n_interpolation_points (int): The number of interpolation points to
            use. The default is :code:`len(x) + 1`. If using resets, this is the
            number of points to use in the first run of the solver, before any resets.
        trustregion_precondition_interpolation (bool): see :ref:`algo_options`.
        trustregion_shrinking_factor_not_successful (float): see :ref:`algo_options`.
        trustregion_shrinking_factor_lower_radius (float): see :ref:`algo_options`.
        trustregion_shrinking_factor_upper_radius (float): see :ref:`algo_options`.
        trustregion_threshold_successful (float): Share of the predicted improvement
            that has to be achieved for a trust region iteration to count as successful.
        trustregion_threshold_very_successful (float): Share of the predicted
            improvement that has to be achieved for a trust region iteration to count
            as very successful.

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
    if trustregion_method_to_replace_extra_points == "momentum":
        trustregion_use_momentum = True
    elif trustregion_method_to_replace_extra_points in ["geometry_improving", None]:
        trustregion_use_momentum = False
    else:
        raise ValueError(
            "trustregion_method_to_replace_extra_points must be "
            "'geometry_improving', 'momentum' or None."
        )

    algo_info = {
        "name": "nag_dfols",
        "primary_criterion_entry": "root_contributions",
        "parallelizes": False,
        "needs_scaling": False,
    }

    advanced_options, trustregion_reset_options = _create_nag_advanced_options(
        x=x,
        noise_multiplicative_level=noise_multiplicative_level,
        noise_additive_level=noise_additive_level,
        noise_n_evals_per_point=noise_n_evals_per_point,
        convergence_noise_corrected_criterion_tolerance=convergence_noise_corrected_criterion_tolerance,  # noqa: E501
        trustregion_initial_radius=trustregion_initial_radius,
        trustregion_reset_options=trustregion_reset_options,
        convergence_slow_progress=convergence_slow_progress,
        interpolation_rounding_error=interpolation_rounding_error,
        threshold_for_safety_step=threshold_for_safety_step,
        clip_criterion_if_overflowing=clip_criterion_if_overflowing,
        initial_directions=initial_directions,
        random_directions_orthogonal=random_directions_orthogonal,
        trustregion_precondition_interpolation=trustregion_precondition_interpolation,
        trustregion_threshold_successful=trustregion_threshold_successful,
        trustregion_threshold_very_successful=trustregion_threshold_very_successful,
        trustregion_shrinking_factor_not_successful=trustregion_shrinking_factor_not_successful,  # noqa: E501
        trustregion_expansion_factor_successful=trustregion_expansion_factor_successful,
        trustregion_expansion_factor_very_successful=trustregion_expansion_factor_very_successful,  # noqa:E501
        trustregion_shrinking_factor_lower_radius=trustregion_shrinking_factor_lower_radius,  # noqa: E501
        trustregion_shrinking_factor_upper_radius=trustregion_shrinking_factor_upper_radius,  # noqa: E501
    )

    fast_start = _build_options_dict(
        user_input=trustregion_fast_start_options,
        default_options=TRUSTREGION_FAST_START_OPTIONS,
    )
    if fast_start["floor_of_jacobian_singular_values"] != 1:
        warnings.warn(
            "Setting the `floor_of_jacobian_singular_values` is not supported by "
            "DF-OLS as of version 1.2.1."
        )
    if (
        fast_start["shrink_upper_radius_in_safety_steps"]
        and fast_start["full_geometry_improving_step"]
    ):
        raise ValueError(
            "full_geometry_improving_step of the trustregion_fast_start_options can "
            "only be True if shrink_upper_radius_in_safety_steps is False."
        )

    (
        faststart_jac,
        faststart_step,
    ) = _get_fast_start_method(fast_start["method"])

    if (
        trustregion_reset_options["n_extra_interpolation_points_per_soft_reset"]
        < trustregion_reset_options["n_extra_interpolation_points_per_soft_reset"]
    ):
        raise ValueError(
            "In the trustregion_reset_options "
            "'n_extra_interpolation_points_per_soft_reset' must "
            "be larger or the same as 'n_extra_interpolation_points_per_hard_reset'."
        )

    dfols_options = {
        "growing.full_rank.use_full_rank_interp": faststart_jac,
        "growing.perturb_trust_region_step": faststart_step,
        "restarts.hard.use_old_rk": trustregion_reset_options[
            "reuse_criterion_value_at_hard_reset"
        ],
        "restarts.auto_detect.min_chgJ_slope": trustregion_reset_options[
            "auto_detect_min_jacobian_increase"
        ],  # noqa: E501
        "restarts.max_npt": trustregion_reset_options["max_interpolation_points"],
        "restarts.increase_npt": trustregion_reset_options[
            "n_extra_interpolation_points_per_soft_reset"
        ]
        > 0,
        "restarts.increase_npt_amt": trustregion_reset_options[
            "n_extra_interpolation_points_per_soft_reset"
        ],
        "restarts.hard.increase_ndirs_initial_amt": trustregion_reset_options[
            "n_extra_interpolation_points_per_hard_reset"
        ]
        - trustregion_reset_options["n_extra_interpolation_points_per_soft_reset"],
        "model.rel_tol": convergence_scaled_criterion_tolerance,
        "regression.num_extra_steps": trustregion_n_extra_points_to_replace_successful,
        "regression.momentum_extra_steps": trustregion_use_momentum,
        "regression.increase_num_extra_steps_with_restart": trustregion_reset_options[
            "n_additional_extra_points_to_replace_per_reset"
        ],
        "growing.ndirs_initial": fast_start["min_inital_points"],
        "growing.delta_scale_new_dirns": fast_start[
            "scale_of_trustregion_step_perturbation"
        ],
        "growing.full_rank.scale_factor": fast_start[
            "scale_of_jacobian_components_perturbation"
        ],
        "growing.full_rank.svd_max_jac_cond": fast_start[
            "jacobian_max_condition_number"
        ],
        "growing.do_geom_steps": fast_start["geometry_improving_steps"],
        "growing.safety.do_safety_step": fast_start["safety_steps"],
        "growing.safety.reduce_delta": fast_start[
            "shrink_upper_radius_in_safety_steps"
        ],
        "growing.safety.full_geom_step": fast_start["full_geometry_improving_step"],
        "growing.reset_delta": fast_start["reset_trustregion_radius_after_fast_start"],
        "growing.reset_rho": fast_start[
            "reset_min_trustregion_radius_after_fast_start"
        ],
        "growing.gamma_dec": fast_start["shrinking_factor_not_successful"],
        "growing.num_new_dirns_each_iter": fast_start[
            "n_extra_search_directions_per_iteration"
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
        maxfun=stopping_max_criterion_evaluations,
        rhobeg=trustregion_initial_radius,
        npt=trustregion_n_interpolation_points,
        rhoend=convergence_minimal_trustregion_radius_tolerance,
        nsamples=noise_n_evals_per_point,
        objfun_has_noise=noise_additive_level or noise_multiplicative_level,
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
    clip_criterion_if_overflowing=CLIP_CRITERION_IF_OVERFLOWING,
    convergence_criterion_value=None,
    convergence_minimal_trustregion_radius_tolerance=CONVERGENCE_MINIMAL_TRUSTREGION_RADIUS_TOLERANCE,  # noqa: E501
    convergence_noise_corrected_criterion_tolerance=CONVERGENCE_NOISE_CORRECTED_CRITERION_TOLERANCE,  # noqa: E501
    convergence_slow_progress=None,
    initial_directions=INITIAL_DIRECTIONS,
    interpolation_rounding_error=INTERPOLATION_ROUNDING_ERROR,
    noise_additive_level=None,
    noise_multiplicative_level=None,
    noise_n_evals_per_point=None,
    random_directions_orthogonal=RANDOM_DIRECTIONS_ORTHOGONAL,
    seek_global_optimum=False,
    stopping_max_criterion_evaluations=STOPPING_MAX_CRITERION_EVALUATIONS,
    threshold_for_safety_step=THRESHOLD_FOR_SAFETY_STEP,
    trustregion_expansion_factor_successful=TRUSTREGION_EXPANSION_FACTOR_SUCCESSFUL,
    trustregion_expansion_factor_very_successful=TRUSTREGION_EXPANSION_FACTOR_VERY_SUCCESSFUL,  # noqa: E501
    trustregion_initial_radius=None,
    trustregion_minimum_change_hession_for_underdetermined_interpolation=True,
    trustregion_n_interpolation_points=None,
    trustregion_precondition_interpolation=TRUSTREGION_PRECONDITION_INTERPOLATION,
    trustregion_reset_options=None,
    trustregion_shrinking_factor_not_successful=TRUSTREGION_SHRINKING_FACTOR_NOT_SUCCESSFUL,  # noqa: E501
    trustregion_shrinking_factor_lower_radius=TRUSTREGION_SHRINKING_FACTOR_LOWER_RADIUS,
    trustregion_shrinking_factor_upper_radius=TRUSTREGION_SHRINKING_FACTOR_UPPER_RADIUS,
    trustregion_threshold_successful=TRUSTREGION_THRESHOLD_SUCCESSFUL,
    trustregion_threshold_very_successful=TRUSTREGION_THRESHOLD_VERY_SUCCESSFUL,
):
    r"""Minimize a function using the BOBYQA algorithm.

    Do not call this function directly but pass its name "nag_pybobyqa" to estimagic's
    maximize or minimize function as `algorithm` argument. Specify your desired
    arguments as a dictionary and pass them as `algo_options` to minimize or
    maximize.

    BOBYQA (:cite:`Powell2009`, :cite:`Cartis2018`, :cite:`Cartis2018a`) is a
    derivative-free trust-region method. It is designed to solve nonlinear local
    minimization problems.

    Remember to cite :cite:`Powell2009` and :cite:`Cartis2018` when using pybobyqa in
    addition to estimagic. If you take advantage of the ``seek_global_optimum`` option,
    cite :cite:`Cartis2018a` additionally.

    There are two main situations when using a derivative-free algorithm like BOBYQA
    is preferable to derivative-based algorithms:

    1. The criterion function is not deterministic, i.e. if we evaluate the criterion
       function multiple times at the same parameter vector we get different results.

    2. The criterion function is very expensive to evaluate and only finite differences
       are available to calculate its derivative.

    The detailed documentation of the algorithm can be found `here
    <https://numericalalgorithmsgroup.github.io/pybobyqa/>`_.

    There are four possible convergence criteria:

    1. when the trust region radius is shrunk below a minimum. This is
       approximately equivalent to an absolute parameter tolerance.

    2. when the criterion value falls below an absolute, user-specified value,
       the optimization terminates successfully.

    3. when insufficient improvements have been gained over a certain number of
       iterations. The (absolute) threshold for what constitutes an insufficient
       improvement, how many iterations have to be insufficient and with which
       iteration to compare can all be specified by the user.

    4. when all evaluations on the interpolation points fall within a scaled version of
       the noise level of the criterion function. This is only applicable if the
       criterion function is noisy.

    Args:
        clip_criterion_if_overflowing (bool): see :ref:`algo_options`.
        convergence_criterion_value (float): Terminate successfully if
            the criterion value falls below this threshold. This is deactivated
            (i.e. set to -inf) by default.
        convergence_minimal_trustregion_radius_tolerance (float): Minimum allowed
            value of the trust region radius, which determines when a successful
            termination occurs.
        convergence_noise_corrected_criterion_tolerance (float): Stop when the
            evaluations on the set of interpolation points all fall within this
            factor of the noise level.
            The default is 1, i.e. when all evaluations are within the noise level.
            If you want to not use this criterion but still flag your
            criterion function as noisy, set this tolerance to 0.0.

            .. warning::
                Very small values, as in most other tolerances don't make sense here.

        convergence_slow_progress (dict): Arguments for converging when the evaluations
            over several iterations only yield small improvements on average, see
            see :ref:`algo_options` for details.
        initial_directions (str): see :ref:`algo_options`.
        interpolation_rounding_error (float): see :ref:`algo_options`.
        noise_additive_level (float): Used for determining the presence of noise
            and the convergence by all interpolation points being within noise level.
            0 means no additive noise. Only multiplicative or additive is supported.
        noise_multiplicative_level (float): Used for determining the presence of noise
            and the convergence by all interpolation points being within noise level.
            0 means no multiplicative noise. Only multiplicative or additive is
            supported.
        noise_n_evals_per_point (callable): How often to evaluate the criterion
            function at each point.
            This is only applicable for criterion functions with noise,
            when averaging multiple evaluations at the same point produces a more
            accurate value.
            The input parameters are the ``upper_trustregion_radius`` (``delta``),
            the ``lower_trustregion_radius`` (``rho``),
            how many iterations the algorithm has been running for, ``n_iterations``
            and how many resets have been performed, ``n_resets``.
            The function must return an integer.
            Default is no averaging (i.e. ``noise_n_evals_per_point(...) = 1``).
        random_directions_orthogonal (bool): see :ref:`algo_options`.
        seek_global_optimum (bool): whether to apply the heuristic to escape local
            minima presented in :cite:`Cartis2018a`. Only applies for noisy criterion
            functions.
        stopping_max_criterion_evaluations (int): see :ref:`algo_options`.
        threshold_for_safety_step (float): see :ref:`algo_options`.
        trustregion_expansion_factor_successful (float): see :ref:`algo_options`.
        trustregion_expansion_factor_very_successful (float): see :ref:`algo_options`.
        trustregion_initial_radius (float): Initial value of the trust region radius.
        trustregion_minimum_change_hession_for_underdetermined_interpolation (bool):
            Whether to solve the underdetermined quadratic interpolation problem by
            minimizing the Frobenius norm of the Hessian, or change in Hessian.
        trustregion_n_interpolation_points (int): The number of interpolation points to
            use. With $n=len(x)$ the default is $2n+1$ if the criterion is not noisy.
            Otherwise, it is set to $(n+1)(n+2)/2)$.

            Larger values are particularly useful for noisy problems.
            Py-BOBYQA requires

            .. math::
                n + 1 \leq \text{trustregion_n_interpolation_points} \leq (n+1)(n+2)/2.
        trustregion_precondition_interpolation (bool): see :ref:`algo_options`.
        trustregion_reset_options (dict): Options for resetting the optimization,
            see :ref:`algo_options` for details.
        trustregion_shrinking_factor_not_successful (float): see :ref:`algo_options`.
        trustregion_shrinking_factor_upper_radius (float): see :ref:`algo_options`.
        trustregion_shrinking_factor_lower_radius (float): see :ref:`algo_options`.
        trustregion_threshold_successful (float): see :ref:`algo_options`.
        trustregion_threshold_very_successful (float): see :ref:`algo_options`.

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

    if convergence_criterion_value is None:
        convergence_criterion_value = -np.inf

    algo_info = {
        "name": "nag_pybobyqa",
        "primary_criterion_entry": "value",
        "parallelizes": False,
        "needs_scaling": False,
    }
    criterion = partial(
        criterion_and_derivative, task="criterion", algorithm_info=algo_info
    )
    advanced_options, trustregion_reset_options = _create_nag_advanced_options(
        x=x,
        noise_multiplicative_level=noise_multiplicative_level,
        noise_additive_level=noise_additive_level,
        trustregion_initial_radius=trustregion_initial_radius,
        noise_n_evals_per_point=noise_n_evals_per_point,
        convergence_noise_corrected_criterion_tolerance=convergence_noise_corrected_criterion_tolerance,  # noqa: E501
        trustregion_reset_options=trustregion_reset_options,
        convergence_slow_progress=convergence_slow_progress,
        interpolation_rounding_error=interpolation_rounding_error,
        threshold_for_safety_step=threshold_for_safety_step,
        clip_criterion_if_overflowing=clip_criterion_if_overflowing,
        initial_directions=initial_directions,
        random_directions_orthogonal=random_directions_orthogonal,
        trustregion_precondition_interpolation=trustregion_precondition_interpolation,
        trustregion_threshold_successful=trustregion_threshold_successful,
        trustregion_threshold_very_successful=trustregion_threshold_very_successful,
        trustregion_shrinking_factor_not_successful=trustregion_shrinking_factor_not_successful,  # noqa: E501
        trustregion_expansion_factor_successful=trustregion_expansion_factor_successful,
        trustregion_expansion_factor_very_successful=trustregion_expansion_factor_very_successful,  # noqa:E501
        trustregion_shrinking_factor_lower_radius=trustregion_shrinking_factor_lower_radius,  # noqa: E501
        trustregion_shrinking_factor_upper_radius=trustregion_shrinking_factor_upper_radius,  # noqa: E501
    )

    pybobyqa_options = {
        "model.abs_tol": convergence_criterion_value,
        "interpolation.minimum_change_hessian": trustregion_minimum_change_hession_for_underdetermined_interpolation,  # noqa: E501
        "restarts.max_unsuccessful_restarts_total": trustregion_reset_options[
            "max_unsuccessful_resets"
        ],
        "restarts.rhobeg_scale_after_unsuccessful_restart": trustregion_reset_options[
            "trust_region_scaling_at_unsuccessful_reset"
        ],  # noqa E501
        "restarts.hard.use_old_fk": trustregion_reset_options[
            "reuse_criterion_value_at_hard_reset"
        ],
        "restarts.auto_detect.min_chg_model_slope": trustregion_reset_options[
            "auto_detect_min_jacobian_increase"
        ],  # noqa: E501
    }

    advanced_options.update(pybobyqa_options)

    res = pybobyqa.solve(
        criterion,
        x0=x,
        bounds=(lower_bounds, upper_bounds),
        maxfun=stopping_max_criterion_evaluations,
        rhobeg=trustregion_initial_radius,
        user_params=advanced_options,
        scaling_within_bounds=False,
        do_logging=False,
        print_progress=False,
        objfun_has_noise=noise_additive_level or noise_multiplicative_level,
        nsamples=noise_n_evals_per_point,
        npt=trustregion_n_interpolation_points,
        rhoend=convergence_minimal_trustregion_radius_tolerance,
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
    if processed["message"].startswith("Error (bad input)"):
        raise ValueError(processed["message"])
    return processed


def _create_nag_advanced_options(
    x,
    noise_multiplicative_level,
    noise_additive_level,
    trustregion_initial_radius,
    noise_n_evals_per_point,
    convergence_noise_corrected_criterion_tolerance,
    trustregion_reset_options,
    convergence_slow_progress,
    interpolation_rounding_error,
    threshold_for_safety_step,
    clip_criterion_if_overflowing,
    initial_directions,
    random_directions_orthogonal,
    trustregion_precondition_interpolation,
    trustregion_threshold_successful,
    trustregion_threshold_very_successful,
    trustregion_shrinking_factor_not_successful,
    trustregion_expansion_factor_successful,
    trustregion_expansion_factor_very_successful,
    trustregion_shrinking_factor_lower_radius,
    trustregion_shrinking_factor_upper_radius,
):
    if noise_multiplicative_level is not None and noise_additive_level is not None:
        raise ValueError("You cannot specify both multiplicative and additive noise.")
    if trustregion_initial_radius is None:
        trustregion_initial_radius = calculate_trustregion_initial_radius(x)
    # -np.inf as a default leads to errors when building the documentation with sphinx.
    noise_n_evals_per_point = _change_evals_per_point_interface(noise_n_evals_per_point)
    trustregion_reset_options = _build_options_dict(
        user_input=trustregion_reset_options,
        default_options=RESET_OPTIONS,
    )
    if trustregion_reset_options["reset_type"] not in ["soft", "hard"]:
        raise ValueError(
            "reset_type in the trustregion_reset_options must be soft or hard."
        )
    if initial_directions not in ["coordinate", "random"]:
        raise ValueError("inital_directions must be either 'coordinate' or 'random'.")
    convergence_slow_progress = _build_options_dict(
        user_input=convergence_slow_progress,
        default_options=CONVERGENCE_SLOW_PROGRESS,
    )

    advanced_options = {
        "general.rounding_error_constant": interpolation_rounding_error,
        "general.safety_step_thresh": threshold_for_safety_step,
        "general.check_objfun_for_overflow": clip_criterion_if_overflowing,
        "tr_radius.eta1": trustregion_threshold_successful,
        "tr_radius.eta2": trustregion_threshold_very_successful,
        "tr_radius.gamma_dec": trustregion_shrinking_factor_not_successful,
        "tr_radius.gamma_inc": trustregion_expansion_factor_successful,
        "tr_radius.gamma_inc_overline": trustregion_expansion_factor_very_successful,
        "tr_radius.alpha1": trustregion_shrinking_factor_lower_radius,
        "tr_radius.alpha2": trustregion_shrinking_factor_upper_radius,
        "general.rounding_error_constant": interpolation_rounding_error,
        "general.safety_step_thresh": threshold_for_safety_step,
        "general.check_objfun_for_overflow": clip_criterion_if_overflowing,
        "init.random_initial_directions": initial_directions == "random",
        "init.random_directions_make_orthogonal": random_directions_orthogonal,
        "slow.thresh_for_slow": convergence_slow_progress[
            "threshold_to_characterize_as_slow"
        ],
        "slow.max_slow_iters": convergence_slow_progress[
            "max_insufficient_improvements"
        ],
        "slow.history_for_slow": convergence_slow_progress["comparison_period"],
        "noise.multiplicative_noise_level": noise_multiplicative_level,
        "noise.additive_noise_level": noise_additive_level,
        "noise.quit_on_noise_level": convergence_noise_corrected_criterion_tolerance > 0
        and (noise_multiplicative_level or noise_additive_level),
        "noise.scale_factor_for_quit": convergence_noise_corrected_criterion_tolerance,
        "interpolation.precondition": trustregion_precondition_interpolation,
        "restarts.use_restarts": trustregion_reset_options["use_resets"],
        "restarts.max_unsuccessful_restarts": trustregion_reset_options[
            "max_consecutive_unsuccessful_resets"
        ],
        "restarts.rhoend_scale": trustregion_reset_options[
            "minimal_trustregion_radius_tolerance_scaling_at_reset"
        ],
        "restarts.use_soft_restarts": trustregion_reset_options["reset_type"] == "soft",
        "restarts.soft.move_xk": trustregion_reset_options["move_center_at_soft_reset"],
        "restarts.soft.max_fake_successful_steps": trustregion_reset_options[
            "max_iterations_without_new_best_after_soft_reset"
        ],  # noqa: E501
        "restarts.auto_detect": trustregion_reset_options["auto_detect"],
        "restarts.auto_detect.history": trustregion_reset_options[
            "auto_detect_history"
        ],  # noqa: E501
        "restarts.auto_detect.min_correl": trustregion_reset_options[
            "auto_detect_min_correlations"
        ],
        "restarts.soft.num_geom_steps": trustregion_reset_options[
            "points_to_replace_at_soft_reset"
        ],
    }

    return advanced_options, trustregion_reset_options


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
                upper_trustregion_radius=delta,
                lower_trustregion_radius=rho,
                n_iterations=iter,
                n_resets=nrestarts,
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


def _get_fast_start_method(user_value):
    """Get fast start method arguments from user value."""
    allowed_values = ["auto", "jacobian", "trustregion"]
    if user_value not in allowed_values:
        raise ValueError(
            "`perturb_jacobian_or_trustregion_step` must be one of "
            f"{allowed_values}. You provided {user_value}."
        )
    if user_value == "auto":
        faststart_jac = None
        faststart_step = None
    else:
        faststart_jac = user_value == "jacobian"
        faststart_step = not faststart_jac

    return faststart_jac, faststart_step
