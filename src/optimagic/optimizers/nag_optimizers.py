"""Implement algorithms by the (Numerical Algorithms Group)[https://www.nag.com/].

The following arguments are not supported as ``algo_options``:

- ``scaling_within_bounds``
- ``init.run_in_parallel``
- ``do_logging``, ``print_progress`` and all their advanced options.

"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Literal, cast

import numpy as np
from numpy.typing import NDArray

from optimagic import mark
from optimagic.config import IS_DFOLS_INSTALLED, IS_PYBOBYQA_INSTALLED
from optimagic.exceptions import NotInstalledError
from optimagic.optimization.algo_options import STOPPING_MAXFUN
from optimagic.optimization.algorithm import Algorithm, InternalOptimizeResult
from optimagic.optimization.internal_optimization_problem import (
    InternalOptimizationProblem,
)
from optimagic.typing import (
    AggregationLevel,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
)
from optimagic.utilities import calculate_trustregion_initial_radius

CONVERGENCE_MINIMAL_TRUSTREGION_RADIUS_TOLERANCE = 1e-8
"""float: Stop when the lower trust region radius falls below this value."""

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

THRESHOLD_FOR_SAFETY_STEP = 0.5
r"""float: Threshold for when to call the safety step (:math:`\gamma_s`).

    :math:`\text{proposed step} \leq \text{threshold_for_safety_step} \cdot
    \text{current_lower_trustregion_radius}`.

"""

CONVERGENCE_NOISE_CORRECTED_FTOL = 1.0
"""float: Stop when the evaluations on the set of interpolation points all fall within
    this factor of the noise level. The default is 1, i.e. when all evaluations are
    within the noise level. If you want to not use this criterion but still flag your
    criterion function as noisy, set this tolerance to 0.0.

    .. warning::
        Very small values, as in most other tolerances don't make sense here.

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


@mark.minimizer(
    name="nag_dfols",
    solver_type=AggregationLevel.LEAST_SQUARES,
    is_available=IS_DFOLS_INSTALLED,
    is_global=False,
    needs_jac=False,
    needs_hess=False,
    needs_bounds=False,
    supports_parallelism=False,
    supports_bounds=True,
    supports_infinite_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class NagDFOLS(Algorithm):
    r"""Minimize a function with least-squares structure using the DFO-LS algorithm.

    The DFO-LS algorithm :cite:`Cartis2018b` is a model-based derivative-free
    trust-region algorithm developed by the Numerical Algorithms Group and researchers
    at the University of Oxford. It is designed to solve the nonlinear least-squares
    minimization problem (with optional bound constraints)

    .. math::

        \min_{x\in\mathbb{R}^n}  &\quad  f(x) := \sum_{i=1}^{m}r_{i}(x)^2 \\
        \text{s.t.} &\quad  \text{lower_bounds} \leq x \leq \text{upper_bounds}

    The :math:`r_{i}` are called root contributions in optimagic.

    DFO-LS is a derivative-free algorithm: it does not require the user to provide the
    derivatives of :math:`f(x)` or :math:`r_{i}(x)`, nor does it attempt to estimate
    them internally (by using finite differencing, for instance). Instead, it
    constructs an interpolation-based model of the root contributions that exploits
    the least-squares structure of the problem and minimizes this model within a
    trust region.

    There are two main situations when using a derivative-free algorithm (such as
    DFO-LS) is preferable to a derivative-based algorithm (which is the vast majority
    of least-squares solvers):

    1. If the root contributions are noisy, then calculating or even estimating their
       derivatives may be impossible (or at least very inaccurate). By noisy we mean
       that if we evaluate :math:`r_{i}(x)` multiple times at the same value of x, we
       get different results. This may happen when a Monte Carlo simulation is used,
       for instance.

    2. If the root contributions are expensive to evaluate, then estimating
       derivatives (which requires n evaluations of each :math:`r_{i}(x)` for every
       point of interest x) may be prohibitively expensive. Derivative-free methods
       are designed to solve the problem with the fewest number of evaluations of the
       criterion as possible.

    DFO-LS was developed by the same group as Py-BOBYQA (``nag_pybobyqa``), its
    general-purpose scalar counterpart. If your problem has least-squares structure,
    DFO-LS typically needs far fewer criterion evaluations than Py-BOBYQA and other
    scalar optimizers because it exploits this structure.

    There are four possible convergence criteria:

    1. when the lower trust region radius is shrunk below a minimum
       (``convergence_minimal_trustregion_radius_tolerance``).

    2. when the improvements of iterations become very small
       (``convergence_slow_progress``). This is similar to a relative criterion
       tolerance, but more general, because you can specify not only the threshold
       for convergence but also a period over which the improvements must have been
       very small.

    3. when a sufficient reduction of the criterion value relative to its value at
       the start parameters has been reached, i.e. when
       :math:`f(x_k)/f(x_0) \leq \textsf{convergence_ftol_scaled}`.

    4. when all evaluations on the interpolation points fall within a scaled version
       of the noise level of the criterion function. This is only applicable if the
       criterion function is noisy. You can specify this criterion with
       ``convergence_noise_corrected_criterion_tolerance``.

    DFO-LS supports resetting the optimization and doing a fast start by starting
    with a smaller interpolation set and growing it dynamically. For more information
    see `the detailed DFO-LS documentation
    <https://numericalalgorithmsgroup.github.io/dfols/>`_ and :cite:`Cartis2018b`.

    Remember to cite :cite:`Cartis2018b` when using DFO-LS in addition to optimagic.

    .. note::
        We recommend to install DFO-LS version 1.5.3 or higher. Versions 1.5.0 or
        lower also work, but the versions 1.5.1 and 1.5.2 contain bugs that can lead
        to errors being raised.

    .. note::
        The following arguments of ``dfols.solve`` are not supported by optimagic:
        ``scaling_within_bounds``, ``init.run_in_parallel``, ``do_logging``,
        ``print_progress`` and all their advanced options.

    """

    clip_criterion_if_overflowing: bool = CLIP_CRITERION_IF_OVERFLOWING
    """Whether to clip the criterion if it would raise an ``OverflowError``
    otherwise."""

    convergence_minimal_trustregion_radius_tolerance: NonNegativeFloat = (
        CONVERGENCE_MINIMAL_TRUSTREGION_RADIUS_TOLERANCE  # noqa: E501
    )
    """Stop when the lower trust-region radius falls below this value.

    This is approximately equivalent to an absolute parameter tolerance and
    corresponds to ``rhoend`` in the DFO-LS documentation, from which the default
    value is taken.

    """

    convergence_noise_corrected_criterion_tolerance: NonNegativeFloat = (
        CONVERGENCE_NOISE_CORRECTED_FTOL  # noqa: E501
    )
    """Stop when the evaluations on the set of interpolation points all fall within
    this factor of the noise level.

    The default is 1, i.e. when all evaluations are within the noise level. If you
    want to not use this criterion but still flag your criterion function as noisy,
    set this tolerance to 0.0.

    .. warning::
        Very small values, as in most other tolerances, don't make sense here.

    """

    convergence_ftol_scaled: NonNegativeFloat = 0.0
    r"""Stop if the criterion value falls below this fraction of its value at the
    start parameters, i.e. terminate if
    :math:`f(x_k)/f(x_0) \leq \textsf{convergence_ftol_scaled}` is reached.

    This is ``model.rel_tol`` in the DFO-LS documentation. With the default of 0.0
    this criterion is deactivated unless the lowest mathematically possible criterion
    value (0.0) is actually achieved.

    """

    convergence_slow_progress: dict[str, Any] | None = None
    """Specification of when to terminate (or reset) the optimization because of only
    slow improvements.

    This is similar to a relative criterion tolerance, only that instead of a single
    improvement the average improvement over several iterations must be small.
    Possible entries are:

    - ``threshold_to_characterize_as_slow`` (float): Threshold whether an improvement
      is insufficient. Note that the improvement is divided by the
      ``comparison_period``, so this is the required average improvement per
      iteration over the comparison period. Default is 1e-8.
    - ``max_insufficient_improvements`` (int): Number of consecutive insufficient
      improvements before termination (or reset). Default is ``20 * len(x)``.
    - ``comparison_period`` (int): How many iterations to go back to calculate the
      improvement. For example 5 would mean that each criterion evaluation is
      compared to the criterion value from 5 iterations before. Default is 5.

    """

    initial_directions: Literal[
        "coordinate",
        "random",
    ] = "coordinate"
    """Whether to draw the initial directions used to build the first interpolation
    set as coordinate directions ("coordinate") or random directions ("random")."""

    interpolation_rounding_error: float = INTERPOLATION_ROUNDING_ERROR
    r"""Scaling factor that controls when the interpolation base point is re-centered
    to reduce roundoff errors.

    Internally, all the NAG algorithms store interpolation points with respect to a
    base point :math:`x_b`; that is, they store :math:`\{y_t - x_b\}`, which reduces
    the risk of roundoff errors. The base point :math:`x_b` is shifted to the current
    iterate :math:`x_k` when
    :math:`\text{proposed step} \leq \textsf{interpolation_rounding_error} \cdot
    \|x_k - x_b\|`. This is ``general.rounding_error_constant`` in the DFO-LS
    documentation, from which the default value is taken.

    """

    noise_additive_level: float | None = None
    """Amount of additive noise in the criterion function.

    It is used for determining the presence of noise and for the convergence
    criterion that all evaluations on the interpolation points are within the noise
    level. 0 means no additive noise. Only additive or multiplicative noise can be
    specified, not both.

    """

    noise_multiplicative_level: float | None = None
    """Amount of multiplicative noise in the criterion function.

    It is used for determining the presence of noise and for the convergence
    criterion that all evaluations on the interpolation points are within the noise
    level. 0 means no multiplicative noise. Only additive or multiplicative noise can
    be specified, not both.

    """

    noise_n_evals_per_point: NonNegativeInt | None = None
    r"""How often to evaluate the criterion function at each point.

    This is only applicable for criterion functions with noise, when averaging
    multiple evaluations at the same point produces a more accurate value. It must be
    a function with the keyword arguments ``upper_trustregion_radius``
    (:math:`\Delta`), ``lower_trustregion_radius`` (:math:`\rho`), ``n_iterations``
    and ``n_resets`` that returns the number of evaluations as an integer. The
    default is no averaging, i.e. to evaluate the criterion only once at each point.

    """

    random_directions_orthogonal: bool = RANDOM_DIRECTIONS_ORTHOGONAL
    """Whether to make randomly drawn initial directions orthogonal.

    This is only relevant if ``initial_directions`` is "random".

    """

    stopping_maxfun: PositiveInt = STOPPING_MAXFUN
    """Maximum number of criterion evaluations.

    If reached, the optimization stops, but this is not counted as successful
    convergence.

    """

    threshold_for_safety_step: NonNegativeFloat = THRESHOLD_FOR_SAFETY_STEP
    r"""Threshold for when to call the safety step (:math:`\gamma_s`).

    A safety step is called when
    :math:`\text{proposed step} \leq \textsf{threshold_for_safety_step} \cdot
    \rho_k`, where :math:`\rho_k` is the current lower trust-region radius. The
    default value is taken from DFO-LS.

    """

    trustregion_expansion_factor_successful: NonNegativeFloat = (
        TRUSTREGION_EXPANSION_FACTOR_SUCCESSFUL
    )
    r"""Ratio by which to expand the upper trust-region radius :math:`\Delta_k` in
    very successful iterations (:math:`\gamma_{inc}` in the notation of the paper).

    The default value is taken from DFO-LS.

    """

    trustregion_expansion_factor_very_successful: NonNegativeFloat = (
        TRUSTREGION_EXPANSION_FACTOR_VERY_SUCCESSFUL  # noqa: E501
    )
    r"""Ratio of the proposed step (:math:`\|s_k\|`) by which to expand the upper
    trust-region radius (:math:`\Delta_k`) in very successful iterations
    (:math:`\overline{\gamma}_{inc}` in the notation of the paper).

    The default value is taken from DFO-LS.

    """

    trustregion_fast_start_options: dict[str, Any] | None = None
    r"""Options to start the optimization while building the full trust-region model.

    To activate this, set the number of interpolation points at which to evaluate the
    criterion before doing the first step (``min_inital_points``) to something
    smaller than the number of parameters. Possible entries are:

    - ``min_inital_points`` (int): Number of initial interpolation points in addition
      to the start point. This should only be changed to a value less than
      ``len(x)``, and only if the default setup cost of ``len(x) + 1`` evaluations of
      the criterion is impractical. If this is set to be less than the default, the
      input value of ``trustregion_n_interpolation_points`` should be set to
      ``len(x)``. If the default is used, all the other parameters have no effect.
      Default is ``trustregion_n_interpolation_points - 1``. If the default setup
      costs of the evaluations are very large, DFO-LS can start with less than
      ``len(x)`` interpolation points and add points to the trust-region model with
      every iteration. Note that the option name is indeed ``min_inital_points``,
      i.e. it contains a spelling mistake.
    - ``method`` ("jacobian", "trustregion" or "auto"): When there are less
      interpolation points than ``len(x)``, the model is underdetermined. This can be
      fixed in two ways: If "jacobian", the interpolated Jacobian is perturbed to
      have full rank, allowing the trust-region step to include components in the
      full search space. This is the default if the number of root contributions is
      at least ``len(x)``. If "trustregion", the trust-region step is perturbed by an
      orthogonal direction not yet searched. This is the default if the number of
      root contributions is smaller than ``len(x)``.
    - ``scale_of_trustregion_step_perturbation`` (float): When adding new search
      directions, the length of the step is the trust-region radius multiplied by
      this value. The default is 0.1 if ``method == "trustregion"`` else 1.
    - ``scale_of_jacobian_components_perturbation`` (float): Magnitude of the extra
      components added to the Jacobian. The default is 1e-2.
    - ``floor_of_jacobian_singular_values`` (float): Floor the singular values of the
      Jacobian at this factor of the last non-zero value. As of version 1.2.1 this
      option is not yet supported by DFO-LS!
    - ``jacobian_max_condition_number`` (float): Cap on the condition number of the
      Jacobian after applying floors to singular values (effectively another floor on
      the smallest singular value, since the largest singular value is fixed).
    - ``geometry_improving_steps`` (bool): Whether to do geometry-improving steps in
      the trust-region algorithm, as per the usual algorithm during the fast start.
    - ``safety_steps`` (bool): Whether to perform safety steps.
    - ``shrink_upper_radius_in_safety_steps`` (bool): During the fast start, whether
      to shrink the upper trust-region radius in safety steps.
    - ``full_geometry_improving_step`` (bool): During the fast start, whether to do a
      full geometry-improving step within safety steps (the same as the post fast
      start phase of the algorithm). Since this involves reducing the upper
      trust-region radius, this can only be ``True`` if
      ``shrink_upper_radius_in_safety_steps`` is ``False``.
    - ``reset_trustregion_radius_after_fast_start`` (bool): Whether to reset the
      upper trust-region radius to its initial value at the end of the fast start
      phase.
    - ``reset_min_trustregion_radius_after_fast_start`` (bool): Whether to reset the
      minimum trust-region radius (:math:`\rho_k`) to its initial value at the end of
      the fast start phase.
    - ``shrinking_factor_not_successful`` (float): Ratio by which to shrink the
      trust-region radius when the realized improvement does not match the
      ``trustregion_threshold_successful`` during the fast start phase. By default it
      is the same as ``trustregion_shrinking_factor_not_successful``.
    - ``n_extra_search_directions_per_iteration`` (int): Number of new search
      directions to add with each iteration where we do not have a full set of search
      directions. This approach is not recommended! Default is 0.

    """

    trustregion_initial_radius: NonNegativeFloat | None = None
    r"""Initial value of the trust-region radius.

    This is ``rhobeg`` in the DFO-LS documentation. By default it is set to
    :math:`0.1 \max(\|x_0\|_{\infty}, 1)`, as in DFO-LS.

    """

    trustregion_method_to_replace_extra_points: (
        Literal["geometry_improving", "momentum"] | None
    ) = "geometry_improving"
    """If replacing extra points in successful iterations, whether to use geometry
    improving steps ("geometry_improving") or the momentum method ("momentum").

    This is only relevant if
    ``trustregion_n_extra_points_to_replace_successful > 0``.

    """

    trustregion_n_extra_points_to_replace_successful: NonNegativeInt = 0
    """The number of extra points (other than accepting the trust-region step) to
    replace in successful iterations.

    This is ``regression.num_extra_steps`` in the DFO-LS documentation. It is useful
    when ``trustregion_n_interpolation_points > len(x) + 1``.

    """

    trustregion_n_interpolation_points: NonNegativeInt | None = None
    """The number of interpolation points to use.

    This is ``npt`` in the DFO-LS documentation. The default is ``len(x) + 1``, as in
    DFO-LS. If using resets, this is the number of points to use in the first run of
    the solver, before any resets.

    """

    trustregion_precondition_interpolation: bool = (
        TRUSTREGION_PRECONDITION_INTERPOLATION
    )
    """Whether to scale the interpolation linear system to improve conditioning.

    The default value is taken from DFO-LS.

    """

    trustregion_reset_options: dict[str, Any] | None = None
    r"""Options for resetting the optimization.

    Possible entries are:

    - ``use_resets`` (bool): Whether to do resets when the lower trust-region radius
      (:math:`\rho_k`) reaches the stopping criterion (:math:`\rho_{end}`), or
      (optionally) when all interpolation points are within the noise level. The
      default is ``True`` if the criterion is noisy.
    - ``minimal_trustregion_radius_tolerance_scaling_at_reset`` (float): Factor with
      which the trust-region stopping criterion is multiplied at each reset.
    - ``reset_type`` (str): Whether to use "soft" or "hard" resets. The default is
      "soft".
    - ``move_center_at_soft_reset`` (bool): Whether to move the trust-region center
      (:math:`x_k`) to the best new point evaluated instead of keeping it constant.
    - ``points_to_replace_at_soft_reset`` (int): Number of interpolation points to
      move at each soft reset.
    - ``reuse_criterion_value_at_hard_reset`` (bool): Whether or not to recycle the
      criterion value at the best iterate found when performing a hard reset. This
      saves one criterion evaluation.
    - ``max_iterations_without_new_best_after_soft_reset`` (int): The maximum number
      of successful steps in a given run where the new criterion value is worse than
      the best value found in previous runs before terminating. The default is
      ``stopping_maxfun``.
    - ``auto_detect`` (bool): Whether or not to automatically determine when to
      reset. This is an additional condition and resets can still be triggered by a
      small upper trust-region radius, etc. There are two criteria used: upper
      trust-region radius shrinkage (no increases over the history, more decreases
      than no changes) and changes in the model Jacobian (consistently increasing
      trend as measured by the slope and correlation coefficient of the line of best
      fit).
    - ``auto_detect_history`` (int): How many iterations of model changes and trust
      region radii to store.
    - ``auto_detect_min_jacobian_increase`` (float): Minimum rate of increase of the
      Jacobian over past iterations to cause a reset.
    - ``auto_detect_min_correlations`` (float): Minimum correlation of the Jacobian
      data set required to cause a reset.
    - ``max_consecutive_unsuccessful_resets`` (int): Maximum number of consecutive
      unsuccessful resets allowed (i.e. resets which did not outperform the best
      known value from earlier runs).
    - ``max_interpolation_points`` (int): Maximum allowed value of the number of
      interpolation points. This is useful if the number of interpolation points
      increases with each reset, e.g. when
      ``n_extra_interpolation_points_per_soft_reset > 0``. The default is
      ``trustregion_n_interpolation_points``.
    - ``n_extra_interpolation_points_per_soft_reset`` (int): Number of points to add
      to the interpolation set with each soft reset.
    - ``n_extra_interpolation_points_per_hard_reset`` (int): Number of points to add
      to the interpolation set with each hard reset.
    - ``n_additional_extra_points_to_replace_per_reset`` (int): This parameter
      modifies ``trustregion_n_extra_points_to_replace_successful``. With each reset
      it is increased by this number.

    """

    trustregion_shrinking_factor_not_successful: NonNegativeFloat | None = (
        TRUSTREGION_SHRINKING_FACTOR_NOT_SUCCESSFUL  # noqa: E501
    )
    """Ratio by which to shrink the upper trust-region radius when the realized
    improvement does not match the ``trustregion_threshold_successful``.

    This is ``tr_radius.gamma_dec`` in the DFO-LS documentation. The default is 0.98
    if the criterion is noisy and 0.5 else, as in DFO-LS.

    """

    trustregion_shrinking_factor_lower_radius: NonNegativeFloat | None = (
        TRUSTREGION_SHRINKING_FACTOR_LOWER_RADIUS
    )
    r"""Ratio by which to shrink the lower trust-region radius (:math:`\rho_k`)
    (:math:`\alpha_1` in the notation of the paper).

    The default is 0.9 if the criterion is noisy and 0.1 else, as in DFO-LS.

    """

    trustregion_shrinking_factor_upper_radius: NonNegativeFloat | None = (
        TRUSTREGION_SHRINKING_FACTOR_UPPER_RADIUS
    )
    r"""Ratio of the current lower trust-region radius (:math:`\rho_k`) by which to
    shrink the upper trust-region radius (:math:`\Delta_k`) when the lower one is
    shrunk (:math:`\alpha_2` in the notation of the paper).

    The default is 0.95 if the criterion is noisy and 0.5 else, as in DFO-LS.

    """

    trustregion_threshold_successful: float = TRUSTREGION_THRESHOLD_SUCCESSFUL
    """Share of the predicted improvement that has to be achieved for a trust-region
    iteration to count as successful.

    This is ``tr_radius.eta1`` in the DFO-LS documentation, from which the default
    value is taken.

    """

    trustregion_threshold_very_successful: float = TRUSTREGION_THRESHOLD_VERY_SUCCESSFUL
    """Share of the predicted improvement that has to be achieved for a trust-region
    iteration to count as very successful.

    This is ``tr_radius.eta2`` in the DFO-LS documentation, from which the default
    value is taken.

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        res = nag_dfols_internal(
            criterion=problem.fun,
            x=x0,
            lower_bounds=problem.bounds.lower,
            upper_bounds=problem.bounds.upper,
            clip_criterion_if_overflowing=self.clip_criterion_if_overflowing,
            convergence_minimal_trustregion_radius_tolerance=self.convergence_minimal_trustregion_radius_tolerance,  # noqa: E501
            convergence_noise_corrected_criterion_tolerance=self.convergence_noise_corrected_criterion_tolerance,  # noqa: E501
            convergence_ftol_scaled=self.convergence_ftol_scaled,
            convergence_slow_progress=self.convergence_slow_progress,
            initial_directions=self.initial_directions,
            interpolation_rounding_error=self.interpolation_rounding_error,
            noise_additive_level=self.noise_additive_level,
            noise_multiplicative_level=self.noise_multiplicative_level,
            noise_n_evals_per_point=self.noise_n_evals_per_point,
            random_directions_orthogonal=self.random_directions_orthogonal,
            stopping_maxfun=self.stopping_maxfun,
            threshold_for_safety_step=self.threshold_for_safety_step,
            trustregion_expansion_factor_successful=self.trustregion_expansion_factor_successful,
            trustregion_expansion_factor_very_successful=self.trustregion_expansion_factor_very_successful,  # noqa: E501
            trustregion_fast_start_options=self.trustregion_fast_start_options,
            trustregion_initial_radius=self.trustregion_initial_radius,
            trustregion_method_to_replace_extra_points=self.trustregion_method_to_replace_extra_points,
            trustregion_n_extra_points_to_replace_successful=self.trustregion_n_extra_points_to_replace_successful,
            trustregion_n_interpolation_points=self.trustregion_n_interpolation_points,
            trustregion_precondition_interpolation=self.trustregion_precondition_interpolation,
            trustregion_reset_options=self.trustregion_reset_options,
            trustregion_shrinking_factor_not_successful=self.trustregion_shrinking_factor_not_successful,
            trustregion_shrinking_factor_lower_radius=self.trustregion_shrinking_factor_lower_radius,
            trustregion_shrinking_factor_upper_radius=self.trustregion_shrinking_factor_upper_radius,
            trustregion_threshold_successful=self.trustregion_threshold_successful,
            trustregion_threshold_very_successful=self.trustregion_threshold_very_successful,
        )
        return res


def nag_dfols_internal(
    criterion,
    x,
    lower_bounds,
    upper_bounds,
    clip_criterion_if_overflowing,
    convergence_minimal_trustregion_radius_tolerance,  # noqa: E501
    convergence_noise_corrected_criterion_tolerance,  # noqa: E501
    convergence_ftol_scaled,
    convergence_slow_progress,
    initial_directions,
    interpolation_rounding_error,
    noise_additive_level,
    noise_multiplicative_level,
    noise_n_evals_per_point,
    random_directions_orthogonal,
    stopping_maxfun,
    threshold_for_safety_step,
    trustregion_expansion_factor_successful,
    trustregion_expansion_factor_very_successful,  # noqa: E501
    trustregion_fast_start_options,
    trustregion_initial_radius,
    trustregion_method_to_replace_extra_points,
    trustregion_n_extra_points_to_replace_successful,
    trustregion_n_interpolation_points,
    trustregion_precondition_interpolation,
    trustregion_reset_options,
    trustregion_shrinking_factor_not_successful,  # noqa: E501
    trustregion_shrinking_factor_lower_radius,
    trustregion_shrinking_factor_upper_radius,
    trustregion_threshold_successful,
    trustregion_threshold_very_successful,
):
    r"""Minimize a function with least squares structure using DFO-LS.

    For details see
    :ref: `list_of_nag_algorithms`.

    """
    if not IS_DFOLS_INSTALLED:
        raise NotInstalledError(
            "The 'nag_dfols' algorithm requires the DFO-LS package to be installed."
            "You can install it with 'pip install DFO-LS'. "
            "For additional installation instructions visit: ",
            r"https://numericalalgorithmsgroup.github.io/dfols/build/html/install.html",
        )
    import dfols

    if trustregion_method_to_replace_extra_points == "momentum":
        trustregion_use_momentum = True
    elif trustregion_method_to_replace_extra_points in ["geometry_improving", None]:
        trustregion_use_momentum = False
    else:
        raise ValueError(
            "trustregion_method_to_replace_extra_points must be "
            "'geometry_improving', 'momentum' or None."
        )

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
        trustregion_expansion_factor_very_successful=trustregion_expansion_factor_very_successful,  # noqa: E501
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
        ],
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
        "model.rel_tol": convergence_ftol_scaled,
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
        "logging.save_diagnostic_info": True,
        "logging.save_xk": True,
    }

    advanced_options.update(dfols_options)

    raw_res = dfols.solve(
        criterion,
        x0=x,
        bounds=(lower_bounds, upper_bounds),
        maxfun=stopping_maxfun,
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

    res = _process_nag_result(raw_res, len(x))
    out = InternalOptimizeResult(
        x=res["solution_x"],
        fun=res["solution_criterion"],
        success=res["success"],
        message=res["message"],
        n_iterations=res["n_iterations"],
        n_fun_evals=res["n_fun_evals"],
    )
    return out


@mark.minimizer(
    name="nag_pybobyqa",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_PYBOBYQA_INSTALLED,
    is_global=False,
    needs_jac=False,
    needs_hess=False,
    needs_bounds=False,
    supports_parallelism=False,
    supports_bounds=True,
    supports_infinite_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class NagPyBOBYQA(Algorithm):
    """Minimize a scalar function using the BOBYQA algorithm.

    BOBYQA (:cite:`Powell2009`, :cite:`Cartis2018`, :cite:`Cartis2018a`) is a
    derivative-free trust-region method. It is designed to solve nonlinear local
    minimization problems (with optional bound constraints).

    This wraps Py-BOBYQA, a flexible Python implementation of Powell's BOBYQA (Bound
    Optimization BY Quadratic Approximation, :cite:`Powell2009`) developed by the
    Numerical Algorithms Group and researchers at the University of Oxford
    :cite:`Cartis2018`. In each iteration, the algorithm builds a quadratic model
    that interpolates the criterion function at a set of points and minimizes this
    model within a trust region. Compared to Powell's original implementation,
    Py-BOBYQA adds features for noisy problems (averaging of multiple evaluations
    and noise-aware termination), multiple restarts, and a heuristic for escaping
    local minima (``seek_global_optimum``, :cite:`Cartis2018a`).

    Remember to cite :cite:`Powell2009` and :cite:`Cartis2018` when using Py-BOBYQA
    in addition to optimagic. If you take advantage of the ``seek_global_optimum``
    option, cite :cite:`Cartis2018a` additionally.

    There are two main situations when using a derivative-free algorithm like BOBYQA
    is preferable to derivative-based algorithms:

    1. The criterion function is not deterministic, i.e. if we evaluate the criterion
       function multiple times at the same parameter vector we get different results.

    2. The criterion function is very expensive to evaluate and only finite
       differences are available to calculate its derivative.

    Py-BOBYQA was developed by the same group as DFO-LS (``nag_dfols``). If your
    criterion function has least-squares structure, use ``nag_dfols`` instead, which
    exploits this structure and typically needs far fewer criterion evaluations.

    The detailed documentation of the algorithm can be found `in the Py-BOBYQA
    documentation <https://numericalalgorithmsgroup.github.io/pybobyqa/>`_.

    There are four possible convergence criteria:

    1. when the lower trust-region radius is shrunk below a minimum
       (``convergence_minimal_trustregion_radius_tolerance``). This is approximately
       equivalent to an absolute parameter tolerance.

    2. when the criterion value falls below an absolute, user-specified value
       (``convergence_criterion_value``), the optimization terminates successfully.

    3. when insufficient improvements have been gained over a certain number of
       iterations (``convergence_slow_progress``). The (absolute) threshold for what
       constitutes an insufficient improvement, how many iterations have to be
       insufficient and with which iteration to compare can all be specified by the
       user.

    4. when all evaluations on the interpolation points fall within a scaled version
       of the noise level of the criterion function
       (``convergence_noise_corrected_criterion_tolerance``). This is only applicable
       if the criterion function is noisy.

    .. note::
        The following arguments of ``pybobyqa.solve`` are not supported by optimagic:
        ``scaling_within_bounds``, ``init.run_in_parallel``, ``do_logging``,
        ``print_progress`` and all their advanced options.

    """

    clip_criterion_if_overflowing: bool = CLIP_CRITERION_IF_OVERFLOWING
    """Whether to clip the criterion if it would raise an ``OverflowError``
    otherwise."""

    convergence_minimal_trustregion_radius_tolerance: NonNegativeFloat = (
        CONVERGENCE_MINIMAL_TRUSTREGION_RADIUS_TOLERANCE  # noqa: E501
    )
    """Stop when the lower trust-region radius falls below this value.

    This is approximately equivalent to an absolute parameter tolerance and
    corresponds to ``rhoend`` in the Py-BOBYQA documentation, from which the default
    value is taken.

    """

    convergence_noise_corrected_criterion_tolerance: NonNegativeFloat = (
        CONVERGENCE_NOISE_CORRECTED_FTOL  # noqa: E501
    )
    """Stop when the evaluations on the set of interpolation points all fall within
    this factor of the noise level.

    The default is 1, i.e. when all evaluations are within the noise level. If you
    want to not use this criterion but still flag your criterion function as noisy,
    set this tolerance to 0.0.

    .. warning::
        Very small values, as in most other tolerances, don't make sense here.

    """

    convergence_criterion_value: float | None = None
    """Terminate successfully if the criterion value falls below this threshold.

    This is deactivated (i.e. set to -inf) by default. It is ``model.abs_tol`` in the
    Py-BOBYQA documentation.

    """

    convergence_slow_progress: dict[str, Any] | None = None
    """Specification of when to terminate (or reset) the optimization because of only
    slow improvements.

    This is similar to a relative criterion tolerance, only that instead of a single
    improvement the average improvement over several iterations must be small.
    Possible entries are:

    - ``threshold_to_characterize_as_slow`` (float): Threshold whether an improvement
      is insufficient. Note that the improvement is divided by the
      ``comparison_period``, so this is the required average improvement per
      iteration over the comparison period. Default is 1e-8.
    - ``max_insufficient_improvements`` (int): Number of consecutive insufficient
      improvements before termination (or reset). Default is ``20 * len(x)``.
    - ``comparison_period`` (int): How many iterations to go back to calculate the
      improvement. For example 5 would mean that each criterion evaluation is
      compared to the criterion value from 5 iterations before. Default is 5.

    """

    initial_directions: Literal[
        "coordinate",
        "random",
    ] = "coordinate"
    """Whether to draw the initial directions used to build the first interpolation
    set as coordinate directions ("coordinate") or random directions ("random")."""

    interpolation_rounding_error: float = INTERPOLATION_ROUNDING_ERROR
    r"""Scaling factor that controls when the interpolation base point is re-centered
    to reduce roundoff errors.

    Internally, all the NAG algorithms store interpolation points with respect to a
    base point :math:`x_b`; that is, they store :math:`\{y_t - x_b\}`, which reduces
    the risk of roundoff errors. The base point :math:`x_b` is shifted to the current
    iterate :math:`x_k` when
    :math:`\text{proposed step} \leq \textsf{interpolation_rounding_error} \cdot
    \|x_k - x_b\|`. This is ``general.rounding_error_constant`` in the Py-BOBYQA
    documentation, from which the default value is taken.

    """

    noise_additive_level: float | None = None
    """Amount of additive noise in the criterion function.

    It is used for determining the presence of noise and for the convergence
    criterion that all evaluations on the interpolation points are within the noise
    level. 0 means no additive noise. Only additive or multiplicative noise can be
    specified, not both.

    """

    noise_multiplicative_level: float | None = None
    """Amount of multiplicative noise in the criterion function.

    It is used for determining the presence of noise and for the convergence
    criterion that all evaluations on the interpolation points are within the noise
    level. 0 means no multiplicative noise. Only additive or multiplicative noise can
    be specified, not both.

    """

    noise_n_evals_per_point: NonNegativeInt | None = None
    r"""How often to evaluate the criterion function at each point.

    This is only applicable for criterion functions with noise, when averaging
    multiple evaluations at the same point produces a more accurate value. It must be
    a function with the keyword arguments ``upper_trustregion_radius``
    (:math:`\Delta`), ``lower_trustregion_radius`` (:math:`\rho`), ``n_iterations``
    and ``n_resets`` that returns the number of evaluations as an integer. The
    default is no averaging, i.e. to evaluate the criterion only once at each point.

    """

    random_directions_orthogonal: bool = RANDOM_DIRECTIONS_ORTHOGONAL
    """Whether to make randomly drawn initial directions orthogonal.

    This is only relevant if ``initial_directions`` is "random".

    """

    seek_global_optimum: bool = False
    """Whether to apply the heuristic to escape local minima presented in
    :cite:`Cartis2018a`.

    The heuristic repeatedly restarts the optimization from the best point found so
    far with an enlarged trust-region radius. It is only a heuristic, so there is no
    guarantee that the global optimum is found. To use it, finite lower and upper
    bounds must be provided for all parameters.

    """

    stopping_max_criterion_evaluations: PositiveInt = STOPPING_MAXFUN
    """Maximum number of criterion evaluations.

    If reached, the optimization stops, but this is not counted as successful
    convergence.

    """

    threshold_for_safety_step: NonNegativeFloat = THRESHOLD_FOR_SAFETY_STEP
    r"""Threshold for when to call the safety step (:math:`\gamma_s`).

    A safety step is called when
    :math:`\text{proposed step} \leq \textsf{threshold_for_safety_step} \cdot
    \rho_k`, where :math:`\rho_k` is the current lower trust-region radius. The
    default value is taken from Py-BOBYQA.

    """

    trustregion_expansion_factor_successful: NonNegativeFloat = (
        TRUSTREGION_EXPANSION_FACTOR_SUCCESSFUL
    )
    r"""Ratio by which to expand the upper trust-region radius :math:`\Delta_k` in
    very successful iterations (:math:`\gamma_{inc}` in the notation of the paper).

    The default value is taken from Py-BOBYQA.

    """

    trustregion_expansion_factor_very_successful: NonNegativeFloat = (
        TRUSTREGION_EXPANSION_FACTOR_VERY_SUCCESSFUL  # noqa: E501
    )
    r"""Ratio of the proposed step (:math:`\|s_k\|`) by which to expand the upper
    trust-region radius (:math:`\Delta_k`) in very successful iterations
    (:math:`\overline{\gamma}_{inc}` in the notation of the paper).

    The default value is taken from Py-BOBYQA.

    """

    trustregion_initial_radius: NonNegativeFloat | None = None
    r"""Initial value of the trust-region radius.

    This is ``rhobeg`` in the Py-BOBYQA documentation. By default it is set to
    :math:`0.1 \max(\|x_0\|_{\infty}, 1)`, as in Py-BOBYQA.

    """

    trustregion_minimum_change_hession_for_underdetermined_interpolation: bool = True
    """Whether to solve the underdetermined quadratic interpolation problem by
    minimizing the Frobenius norm of the change in the Hessian.

    If True (the default, as in Py-BOBYQA and Powell's original BOBYQA), the
    quadratic model is chosen such that the Frobenius norm of the change in its
    Hessian relative to the previous iteration is minimal. If False, the Frobenius
    norm of the Hessian itself is minimized. This is
    ``interpolation.minimum_change_hessian`` in the Py-BOBYQA documentation.

    """

    trustregion_n_interpolation_points: NonNegativeInt | None = None
    r"""The number of interpolation points to use.

    This is ``npt`` in the Py-BOBYQA documentation. With :math:`n = len(x)` the
    default is :math:`2n+1` if the criterion is not noisy. Otherwise, it is set to
    :math:`(n+1)(n+2)/2`. Larger values are particularly useful for noisy problems.
    Py-BOBYQA requires

    .. math::
        n + 1 \leq \textsf{trustregion_n_interpolation_points} \leq (n+1)(n+2)/2.

    """

    trustregion_precondition_interpolation: bool = (
        TRUSTREGION_PRECONDITION_INTERPOLATION
    )
    """Whether to scale the interpolation linear system to improve conditioning.

    The default value is taken from Py-BOBYQA.

    """

    trustregion_reset_options: dict[str, Any] | None = None
    r"""Options for resetting the optimization.

    Possible entries are:

    - ``use_resets`` (bool): Whether to do resets when the lower trust-region radius
      (:math:`\rho_k`) reaches the stopping criterion (:math:`\rho_{end}`), or
      (optionally) when all interpolation points are within the noise level. The
      default is ``True`` if the criterion is noisy.
    - ``minimal_trustregion_radius_tolerance_scaling_at_reset`` (float): Factor with
      which the trust-region stopping criterion is multiplied at each reset.
    - ``reset_type`` (str): Whether to use "soft" or "hard" resets. The default is
      "soft".
    - ``move_center_at_soft_reset`` (bool): Whether to move the trust-region center
      (:math:`x_k`) to the best new point evaluated instead of keeping it constant.
    - ``points_to_replace_at_soft_reset`` (int): Number of interpolation points to
      move at each soft reset.
    - ``reuse_criterion_value_at_hard_reset`` (bool): Whether or not to recycle the
      criterion value at the best iterate found when performing a hard reset. This
      saves one criterion evaluation.
    - ``max_iterations_without_new_best_after_soft_reset`` (int): The maximum number
      of successful steps in a given run where the new criterion value is worse than
      the best value found in previous runs before terminating. The default is
      ``stopping_max_criterion_evaluations``.
    - ``auto_detect`` (bool): Whether or not to automatically determine when to
      reset. This is an additional condition and resets can still be triggered by a
      small upper trust-region radius, etc. There are two criteria used: upper
      trust-region radius shrinkage (no increases over the history, more decreases
      than no changes) and changes in the model Jacobian (consistently increasing
      trend as measured by the slope and correlation coefficient of the line of best
      fit).
    - ``auto_detect_history`` (int): How many iterations of model changes and trust
      region radii to store.
    - ``auto_detect_min_jacobian_increase`` (float): Minimum rate of increase of the
      Jacobian over past iterations to cause a reset.
    - ``auto_detect_min_correlations`` (float): Minimum correlation of the Jacobian
      data set required to cause a reset.
    - ``max_consecutive_unsuccessful_resets`` (int): Maximum number of consecutive
      unsuccessful resets allowed (i.e. resets which did not outperform the best
      known value from earlier runs).
    - ``max_unsuccessful_resets`` (int): Number of total unsuccessful resets allowed.
      The default is 20 if ``seek_global_optimum`` and else unrestricted.
    - ``trust_region_scaling_at_unsuccessful_reset`` (float): Factor by which to
      expand the initial lower trust-region radius (:math:`\rho_{beg}`) after
      unsuccessful resets. The default is 1.1 if ``seek_global_optimum`` else 1.

    """

    trustregion_shrinking_factor_not_successful: NonNegativeFloat | None = (
        TRUSTREGION_SHRINKING_FACTOR_NOT_SUCCESSFUL  # noqa: E501
    )
    """Ratio by which to shrink the upper trust-region radius when the realized
    improvement does not match the ``trustregion_threshold_successful``.

    This is ``tr_radius.gamma_dec`` in the Py-BOBYQA documentation. The default is
    0.98 if the criterion is noisy and 0.5 else, as in Py-BOBYQA.

    """

    trustregion_shrinking_factor_lower_radius: NonNegativeFloat | None = (
        TRUSTREGION_SHRINKING_FACTOR_LOWER_RADIUS
    )
    r"""Ratio by which to shrink the lower trust-region radius (:math:`\rho_k`)
    (:math:`\alpha_1` in the notation of the paper).

    The default is 0.9 if the criterion is noisy and 0.1 else, as in Py-BOBYQA.

    """

    trustregion_shrinking_factor_upper_radius: NonNegativeFloat | None = (
        TRUSTREGION_SHRINKING_FACTOR_UPPER_RADIUS
    )
    r"""Ratio of the current lower trust-region radius (:math:`\rho_k`) by which to
    shrink the upper trust-region radius (:math:`\Delta_k`) when the lower one is
    shrunk (:math:`\alpha_2` in the notation of the paper).

    The default is 0.95 if the criterion is noisy and 0.5 else, as in Py-BOBYQA.

    """

    trustregion_threshold_successful: float = TRUSTREGION_THRESHOLD_SUCCESSFUL
    """Share of the predicted improvement that has to be achieved for a trust-region
    iteration to count as successful.

    This is ``tr_radius.eta1`` in the Py-BOBYQA documentation, from which the default
    value is taken.

    """

    trustregion_threshold_very_successful: float = TRUSTREGION_THRESHOLD_VERY_SUCCESSFUL
    """Share of the predicted improvement that has to be achieved for a trust-region
    iteration to count as very successful.

    This is ``tr_radius.eta2`` in the Py-BOBYQA documentation, from which the default
    value is taken.

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        res = nag_pybobyqa_internal(
            criterion=cast(
                Callable[[NDArray[np.float64]], NDArray[np.float64]],
                problem.fun,
            ),
            x=x0,
            lower_bounds=problem.bounds.lower,
            upper_bounds=problem.bounds.upper,
            clip_criterion_if_overflowing=self.clip_criterion_if_overflowing,
            convergence_minimal_trustregion_radius_tolerance=self.convergence_minimal_trustregion_radius_tolerance,  # noqa: E501
            convergence_noise_corrected_criterion_tolerance=self.convergence_noise_corrected_criterion_tolerance,  # noqa: E501
            convergence_slow_progress=self.convergence_slow_progress,
            convergence_criterion_value=self.convergence_criterion_value,
            initial_directions=self.initial_directions,
            interpolation_rounding_error=self.interpolation_rounding_error,
            noise_additive_level=self.noise_additive_level,
            noise_multiplicative_level=self.noise_multiplicative_level,
            noise_n_evals_per_point=self.noise_n_evals_per_point,
            random_directions_orthogonal=self.random_directions_orthogonal,
            seek_global_optimum=self.seek_global_optimum,
            stopping_max_criterion_evaluations=self.stopping_max_criterion_evaluations,
            threshold_for_safety_step=self.threshold_for_safety_step,
            trustregion_expansion_factor_successful=self.trustregion_expansion_factor_successful,
            trustregion_expansion_factor_very_successful=self.trustregion_expansion_factor_very_successful,  # noqa: E501
            trustregion_initial_radius=self.trustregion_initial_radius,
            trustregion_minimum_change_hession_for_underdetermined_interpolation=self.trustregion_minimum_change_hession_for_underdetermined_interpolation,  # noqa: E501
            trustregion_n_interpolation_points=self.trustregion_n_interpolation_points,
            trustregion_precondition_interpolation=self.trustregion_precondition_interpolation,
            trustregion_reset_options=self.trustregion_reset_options,
            trustregion_shrinking_factor_not_successful=self.trustregion_shrinking_factor_not_successful,
            trustregion_shrinking_factor_lower_radius=self.trustregion_shrinking_factor_lower_radius,
            trustregion_shrinking_factor_upper_radius=self.trustregion_shrinking_factor_upper_radius,
            trustregion_threshold_successful=self.trustregion_threshold_successful,
            trustregion_threshold_very_successful=self.trustregion_threshold_very_successful,
        )
        return res


def nag_pybobyqa_internal(
    criterion,
    x,
    lower_bounds,
    upper_bounds,
    clip_criterion_if_overflowing,
    convergence_criterion_value,
    convergence_minimal_trustregion_radius_tolerance,  # noqa: E501
    convergence_noise_corrected_criterion_tolerance,  # noqa: E501
    convergence_slow_progress,
    initial_directions,
    interpolation_rounding_error,
    noise_additive_level,
    noise_multiplicative_level,
    noise_n_evals_per_point,
    random_directions_orthogonal,
    seek_global_optimum,
    stopping_max_criterion_evaluations,
    threshold_for_safety_step,
    trustregion_expansion_factor_successful,
    trustregion_expansion_factor_very_successful,  # noqa: E501
    trustregion_initial_radius,
    trustregion_minimum_change_hession_for_underdetermined_interpolation,
    trustregion_n_interpolation_points,
    trustregion_precondition_interpolation,
    trustregion_reset_options,
    trustregion_shrinking_factor_not_successful,  # noqa: E501
    trustregion_shrinking_factor_lower_radius,
    trustregion_shrinking_factor_upper_radius,
    trustregion_threshold_successful,
    trustregion_threshold_very_successful,
):
    r"""Minimize a function using the BOBYQA algorithm.

    For details see
    :ref: `list_of_nag_algorithms`.

    """
    if not IS_PYBOBYQA_INSTALLED:
        raise NotInstalledError(
            "The 'nag_pybobyqa' algorithm requires the Py-BOBYQA package to be "
            "installed. You can install it with 'pip install Py-BOBYQA'. "
            "For additional installation instructions visit: ",
            r"https://numericalalgorithmsgroup.github.io/pybobyqa/build/html/"
            "install.html",
        )
    import pybobyqa

    if convergence_criterion_value is None:
        convergence_criterion_value = -np.inf

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
        trustregion_expansion_factor_very_successful=trustregion_expansion_factor_very_successful,  # noqa: E501
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
        ],
        "restarts.hard.use_old_fk": trustregion_reset_options[
            "reuse_criterion_value_at_hard_reset"
        ],
        "restarts.auto_detect.min_chg_model_slope": trustregion_reset_options[
            "auto_detect_min_jacobian_increase"
        ],
        "logging.save_diagnostic_info": True,
        "logging.save_xk": True,
    }

    advanced_options.update(pybobyqa_options)

    raw_res = pybobyqa.solve(
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

    res = _process_nag_result(raw_res, len(x))

    out = InternalOptimizeResult(
        x=res["solution_x"],
        fun=res["solution_criterion"],
        success=res["success"],
        message=res["message"],
        n_iterations=res["n_iterations"],
    )

    return out


def _process_nag_result(nag_result_obj, len_x):
    """Convert the NAG result object to our result dictionary.

    Args:
        nag_result_obj: NAG result object
        len_x (int): length of the supplied parameters, i.e. the dimensionality of the
            problem.


    Returns:
        results (dict): See :ref:`internal_optimizer_output` for details.

    """
    if hasattr(nag_result_obj, "f"):
        solution_fun = nag_result_obj.f
    else:
        solution_fun = nag_result_obj.obj

    processed = {
        "solution_criterion": solution_fun,
        "n_fun_evals": nag_result_obj.nx,
        "message": nag_result_obj.msg,
        "success": nag_result_obj.flag == nag_result_obj.EXIT_SUCCESS,
        "reached_convergence_criterion": None,
        "diagnostic_info": nag_result_obj.diagnostic_info,
    }
    try:
        n_iterations = int(nag_result_obj.diagnostic_info["iters_total"].iloc[-1])
        processed["n_iterations"] = n_iterations
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        processed["n_iterations"] = None

    if hasattr(nag_result_obj, "states"):
        processed.update({"states": nag_result_obj.states})
    if hasattr(nag_result_obj, "history_params"):
        processed.update({"history_params": nag_result_obj.history_params})
    if nag_result_obj.x is not None:
        processed["solution_x"] = nag_result_obj.x
    else:
        processed["solution_x"] = np.array([np.nan] * len_x)
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

    is_noisy = bool(noise_additive_level or noise_multiplicative_level)

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
        "noise.quit_on_noise_level": (
            convergence_noise_corrected_criterion_tolerance > 0
        )
        and is_noisy,
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
        ],
        "restarts.auto_detect": trustregion_reset_options["auto_detect"],
        "restarts.auto_detect.history": trustregion_reset_options[
            "auto_detect_history"
        ],
        "restarts.auto_detect.min_correl": trustregion_reset_options[
            "auto_detect_min_correlations"
        ],
        "restarts.soft.num_geom_steps": trustregion_reset_options[
            "points_to_replace_at_soft_reset"
        ],
    }

    return advanced_options, trustregion_reset_options


def _change_evals_per_point_interface(func):
    """Change the interface of the user supplied function to the one expected by NAG.

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
            f"You specified illegal options {', '.join(invalid)}. Allowed are: , ".join(
                full_options.keys()
            )
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
