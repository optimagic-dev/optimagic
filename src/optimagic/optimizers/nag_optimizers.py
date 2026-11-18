"""Implement algorithms by the `Numerical Algorithms Group <https://www.nag.com/>`_.

The module wraps DFO-LS (``nag_dfols``) and Py-BOBYQA (``nag_pybobyqa``). Both solvers
take a few arguments directly and a dictionary of advanced options with keys such as
``"restarts.use_restarts"``. The wrappers keep this structure: arguments and the
miscellaneous ``general.*`` options are options of the algorithm, every other group of
advanced options is one option group, e.g. ``trustregion_options``.

Two groups belong to a feature that the solvers switch on with a separate flag: the
restart options (``restarts.use_restarts``) and the noise options (the argument
``objfun_has_noise``). For these, flag and group are one option that accepts a
boolean or the group: ``restarts`` and ``noisy``.

The dictionaries below map optimagic options to the upstream names. They are the single
source of truth for the wrappers and their tests. An entry of an option group is
written as ``"<option>.<entry>"``. For ``restarts`` and ``noisy`` there is one row for
the option itself, which is the flag, and one row for each entry of the group.

"""

from __future__ import annotations

from dataclasses import dataclass, field, is_dataclass
from typing import Any, Callable, Mapping

import numpy as np
from numpy.typing import NDArray

from optimagic import mark
from optimagic.config import IS_DFOLS_INSTALLED, IS_PYBOBYQA_INSTALLED
from optimagic.exceptions import InvalidAlgoOptionError, NotInstalledError
from optimagic.optimization.algo_option_tools import DeprecatedOption, option_group
from optimagic.optimization.algo_options import STOPPING_MAXFUN
from optimagic.optimization.algorithm import Algorithm, InternalOptimizeResult
from optimagic.optimization.internal_optimization_problem import (
    InternalOptimizationProblem,
)
from optimagic.typing import (
    AggregationLevel,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    ProbabilityFloat,
)

# ======================================================================================
# Mapping from optimagic options to upstream names
# ======================================================================================

_COMMON_SOLVE_ARGUMENTS = {
    "stopping_maxfun": "maxfun",
    "convergence_min_trustregion_radius": "rhoend",
    "trustregion_initial_radius": "rhobeg",
    "n_interpolation_points": "npt",
    "noisy": "objfun_has_noise",
    "n_evals_per_point": "nsamples",
}

_COMMON_ADVANCED_OPTIONS = {
    "convergence_target_value": "model.abs_tol",
    "rounding_error_constant": "general.rounding_error_constant",
    "safety_step_threshold": "general.safety_step_thresh",
    "init_options.random_initial_directions": "init.random_initial_directions",
    "init_options.random_directions_make_orthogonal": (
        "init.random_directions_make_orthogonal"
    ),
    "trustregion_options.shrinking_threshold": "tr_radius.eta1",
    "trustregion_options.expansion_threshold": "tr_radius.eta2",
    "trustregion_options.shrinking_factor": "tr_radius.gamma_dec",
    "trustregion_options.expansion_factor": "tr_radius.gamma_inc",
    "trustregion_options.step_expansion_factor": "tr_radius.gamma_inc_overline",
    "trustregion_options.min_radius_shrinking_factor": "tr_radius.alpha1",
    "trustregion_options.radius_factor_after_min_radius_shrinking": "tr_radius.alpha2",
    "slow_progress_options.history_for_slow": "slow.history_for_slow",
    "slow_progress_options.threshold_for_slow": "slow.thresh_for_slow",
    "slow_progress_options.max_slow_iterations": "slow.max_slow_iters",
    "noisy.quit_on_noise_level": "noise.quit_on_noise_level",
    "noisy.scale_factor_for_quit": "noise.scale_factor_for_quit",
    "noisy.multiplicative_noise_level": "noise.multiplicative_noise_level",
    "noisy.additive_noise_level": "noise.additive_noise_level",
    "interpolation_options.precondition": "interpolation.precondition",
    "restarts": "restarts.use_restarts",
    "restarts.max_unsuccessful_restarts": "restarts.max_unsuccessful_restarts",
    "restarts.convergence_min_radius_scale": "restarts.rhoend_scale",
    "restarts.use_soft_restarts": "restarts.use_soft_restarts",
    "restarts.soft_n_geometry_steps": "restarts.soft.num_geom_steps",
    "restarts.soft_move_iterate": "restarts.soft.move_xk",
    "restarts.soft_max_fake_successful_steps": (
        "restarts.soft.max_fake_successful_steps"
    ),
    "restarts.auto_detect": "restarts.auto_detect",
    "restarts.auto_detect_history": "restarts.auto_detect.history",
    "restarts.auto_detect_min_correlation": "restarts.auto_detect.min_correl",
}

DFOLS_SOLVE_ARGUMENTS: Mapping[str, str] = dict(_COMMON_SOLVE_ARGUMENTS)
"""Options of ``nag_dfols`` and the arguments of ``dfols.solve`` they are passed to."""

DFOLS_ADVANCED_OPTIONS: Mapping[str, str] = {
    **_COMMON_ADVANCED_OPTIONS,
    "convergence_ftol_scaled": "model.rel_tol",
    "check_fun_for_overflow": "general.check_objfun_for_overflow",
    "interpolation_options.throw_error_on_nans": "interpolation.throw_error_on_nans",
    "regression_options.n_extra_steps": "regression.num_extra_steps",
    "regression_options.increase_n_extra_steps_with_restart": (
        "regression.increase_num_extra_steps_with_restart"
    ),
    "regression_options.momentum_extra_steps": "regression.momentum_extra_steps",
    "restarts.hard_use_old_fun_value": "restarts.hard.use_old_rk",
    "restarts.auto_detect_min_model_change_slope": (
        "restarts.auto_detect.min_chgJ_slope"
    ),
    "restarts.increase_n_interpolation_points": "restarts.increase_npt",
    "restarts.increase_n_interpolation_points_amount": ("restarts.increase_npt_amt"),
    "restarts.hard_increase_n_initial_directions_amount": (
        "restarts.hard.increase_ndirs_initial_amt"
    ),
    "restarts.max_n_interpolation_points": "restarts.max_npt",
    "restarts.throw_error_on_nans": "restarts.throw_error_on_nans",
    "growing_options.n_initial_directions": "growing.ndirs_initial",
    "growing_options.n_new_directions_each_iteration": (
        "growing.num_new_dirns_each_iter"
    ),
    "growing_options.radius_scale_new_directions": "growing.delta_scale_new_dirns",
    "growing_options.do_geometry_steps": "growing.do_geom_steps",
    "growing_options.reset_radius": "growing.reset_delta",
    "growing_options.reset_min_radius": "growing.reset_rho",
    "growing_options.shrinking_factor": "growing.gamma_dec",
    "growing_options.do_safety_step": "growing.safety.do_safety_step",
    "growing_options.safety_reduce_radius": "growing.safety.reduce_delta",
    "growing_options.safety_full_geometry_step": "growing.safety.full_geom_step",
    "growing_options.use_full_rank_interpolation": (
        "growing.full_rank.use_full_rank_interp"
    ),
    "growing_options.perturb_trust_region_step": "growing.perturb_trust_region_step",
    "growing_options.full_rank_scale_factor": "growing.full_rank.scale_factor",
    "growing_options.full_rank_svd_scale_factor": "growing.full_rank.svd_scale_factor",
    "growing_options.full_rank_min_singular_value": "growing.full_rank.min_sing_val",
    "growing_options.full_rank_max_jacobian_condition_number": (
        "growing.full_rank.svd_max_jac_cond"
    ),
}
"""Options of ``nag_dfols`` and the keys of ``user_params`` they are passed to."""

DFOLS_UNSUPPORTED_SOLVE_ARGUMENTS: tuple[str, ...] = (
    # handled by optimagic
    "argsf",
    "scaling_within_bounds",
    "do_logging",
    "print_progress",
    # regularizer, not supported yet
    "h",
    "lh",
    "prox_uh",
    "argsh",
    "argsprox",
    # convex constraints given as projections, not supported yet
    "projections",
)
"""Arguments of ``dfols.solve`` that cannot be set through optimagic."""

DFOLS_UNSUPPORTED_ADVANCED_OPTIONS: tuple[str, ...] = (
    # handled by optimagic; the wrapper sets the entries of _DIAGNOSTIC_OPTIONS
    "logging.n_to_print_whole_x_vector",
    "logging.save_diagnostic_info",
    "logging.save_poisedness",
    "logging.save_xk",
    "logging.save_rk",
    # does not evaluate in parallel
    "init.run_in_parallel",
    # regularizer, not supported yet
    "func_tol.criticality_measure",
    "func_tol.tr_step",
    "func_tol.max_iters",
    "sfista.max_iters_scaling",
    # convex constraints given as projections, not supported yet
    "dykstra.d_tol",
    "dykstra.max_iters",
    "matrix_rank.r_tol",
    # evaluation database, not supported yet
    "database.new_direction_tol",
)
"""Keys of the DFO-LS ``user_params`` that cannot be set through optimagic."""

PYBOBYQA_SOLVE_ARGUMENTS: Mapping[str, str] = {
    **_COMMON_SOLVE_ARGUMENTS,
    "seek_global_optimum": "seek_global_minimum",
}
"""Options of ``nag_pybobyqa`` and the arguments of ``pybobyqa.solve`` they are passed
to."""

PYBOBYQA_ADVANCED_OPTIONS: Mapping[str, str] = {
    **_COMMON_ADVANCED_OPTIONS,
    "interpolation_options.minimum_change_hessian": (
        "interpolation.minimum_change_hessian"
    ),
    "restarts.hard_use_old_fun_value": "restarts.hard.use_old_fk",
    "restarts.auto_detect_min_model_change_slope": (
        "restarts.auto_detect.min_chg_model_slope"
    ),
    "restarts.max_unsuccessful_restarts_total": (
        "restarts.max_unsuccessful_restarts_total"
    ),
    "restarts.initial_radius_scale_after_unsuccessful_restart": (
        "restarts.rhobeg_scale_after_unsuccessful_restart"
    ),
}
"""Options of ``nag_pybobyqa`` and the keys of ``user_params`` they are passed to."""

PYBOBYQA_UNSUPPORTED_SOLVE_ARGUMENTS: tuple[str, ...] = (
    # handled by optimagic
    "args",
    "scaling_within_bounds",
    "do_logging",
    "print_progress",
    # convex constraints given as projections, not supported yet
    "projections",
)
"""Arguments of ``pybobyqa.solve`` that cannot be set through optimagic."""

PYBOBYQA_UNSUPPORTED_ADVANCED_OPTIONS: tuple[str, ...] = (
    # handled by optimagic; the wrapper sets the entries of _DIAGNOSTIC_OPTIONS
    "logging.n_to_print_whole_x_vector",
    "logging.save_diagnostic_info",
    "logging.save_poisedness",
    "logging.save_xk",
    # does not evaluate in parallel
    "init.run_in_parallel",
    # only catches errors of the function, which optimagic handles
    "general.check_objfun_for_overflow",
    # convex constraints given as projections, not supported yet
    "projections.dykstra.d_tol",
    "projections.dykstra.max_iters",
    "projections.feasible_tol",
    "projections.pgd_tol",
)
"""Keys of the Py-BOBYQA ``user_params`` that cannot be set through optimagic."""

_DIAGNOSTIC_OPTIONS: dict[str, bool] = {
    # The diagnostic information has one row per iteration. It is the only way to
    # get the number of iterations from the solvers.
    "logging.save_diagnostic_info": True,
    # Computing the poisedness of the interpolation points in every iteration is by
    # far the most expensive part of the diagnostic information and is not needed.
    "logging.save_poisedness": False,
}
"""Advanced options the wrappers set to count the iterations of the solvers."""


# ======================================================================================
# Option groups
# ======================================================================================


@option_group
@dataclass(frozen=True)
class NagInitOptions:
    """Options for the first set of points at which the function is evaluated.

    Before the first step, the solver evaluates the function at the start parameters
    and at one point per search direction around them.

    """

    random_initial_directions: bool = False
    """Whether the first points are placed in random directions.

    With ``False``, each point changes one parameter at a time. Random directions
    can help if the parameters are strongly correlated or if moving along single
    parameters is uninformative; the results then depend on the state of NumPy's
    global random number generator.

    This is ``init.random_initial_directions`` of the solver, from which the default
    is taken.

    """

    random_directions_make_orthogonal: bool = True
    """Whether random directions are made orthogonal to each other.

    Orthogonal directions spread the first points evenly, which makes the first model
    more reliable. Only relevant if ``random_initial_directions`` is ``True``.

    This is ``init.random_directions_make_orthogonal`` of the solver, from which the
    default is taken.

    """


@option_group
@dataclass(frozen=True)
class NagTrustregionOptions:
    """Options that control how the trust-region radius changes.

    The solver works with two radii. The trust-region radius is the size of the
    region in which the next step is searched. It grows after good steps and shrinks
    after bad ones. The minimal radius is a lower bound of the trust-region radius
    that only shrinks. It measures how precise the solution already is: the
    optimization converges when it reaches ``convergence_min_trustregion_radius``.

    The quality of a step is the improvement of the function value divided by the
    improvement the model predicted.

    """

    shrinking_threshold: ProbabilityFloat = 0.1
    """Step quality below which the trust-region radius shrinks.

    Larger values make the solver more cautious: more steps count as bad and the
    radius shrinks more often.

    This is ``tr_radius.eta1`` of the solver, from which the default is taken.

    """

    expansion_threshold: ProbabilityFloat = 0.7
    """Step quality above which the trust-region radius grows.

    Smaller values make the solver more aggressive: more steps count as very good
    and the radius grows more often. Must be larger than ``shrinking_threshold``.

    This is ``tr_radius.eta2`` of the solver, from which the default is taken.

    """

    shrinking_factor: ProbabilityFloat | None = None
    """Factor by which the trust-region radius is multiplied after a bad step.

    Values close to 1 shrink the radius slowly, which is what noisy functions need:
    a step can look bad just because of noise. Smaller values give up on a region
    faster.

    This is ``tr_radius.gamma_dec`` of the solver. With ``None`` the solver uses its
    default, which is 0.5, or 0.98 for noisy functions (see ``noisy``).

    """

    expansion_factor: PositiveFloat = 2.0
    """Factor by which the trust-region radius is multiplied after a very good step.

    Larger values let the solver take long steps sooner, at the risk of leaving the
    region in which the model is reliable.

    This is ``tr_radius.gamma_inc`` of the solver, from which the default is taken.

    """

    step_expansion_factor: PositiveFloat = 4.0
    """Multiple of the step length to which the trust-region radius may grow.

    After a very good step the new radius is the larger of ``expansion_factor``
    times the old radius and this factor times the length of the step. Larger values
    let the radius grow faster after long successful steps.

    This is ``tr_radius.gamma_inc_overline`` of the solver, from which the default is
    taken.

    """

    min_radius_shrinking_factor: ProbabilityFloat | None = None
    """Factor by which the minimal radius is multiplied when it shrinks.

    Values close to 1 approach the final precision in small steps. This costs more
    function evaluations but is more robust for noisy functions.

    This is ``tr_radius.alpha1`` of the solver. With ``None`` the solver uses its
    default, which is 0.1, or 0.9 for noisy functions (see ``noisy``).

    """

    radius_factor_after_min_radius_shrinking: ProbabilityFloat | None = None
    """Factor that sets the trust-region radius when the minimal radius shrinks.

    The trust-region radius is then set to this factor times the old minimal radius,
    but not below the new minimal radius. Larger values keep the search region wider
    after the minimal radius was reduced.

    This is ``tr_radius.alpha2`` of the solver. With ``None`` the solver uses its
    default, which is 0.5, or 0.95 for noisy functions (see ``noisy``).

    """


@option_group
@dataclass(frozen=True)
class NagSlowProgressOptions:
    """Options for stopping (or restarting) because of slow progress.

    An iteration is slow if the function value improved too little over the last few
    iterations. After too many slow iterations in a row the optimization stops
    without success, or restarts if restarts are used.

    This is only a safety net. The main convergence criterion is
    ``convergence_min_trustregion_radius``.

    """

    history_for_slow: PositiveInt = 5
    """Number of iterations over which the improvement is measured.

    Each function value is compared with the one from this many iterations before.
    Larger values smooth over single iterations without improvement.

    This is ``slow.history_for_slow`` of the solver, from which the default is taken.

    """

    threshold_for_slow: NonNegativeFloat = 1e-8
    """Average improvement per iteration below which an iteration is slow.

    Larger values stop the optimization earlier when progress stalls and save
    function evaluations, but can stop before an optimum is reached. DFO-LS measures
    the improvement in the logarithm of the function value, i.e. relative to its
    size. Py-BOBYQA measures it in the function value itself.

    This is ``slow.thresh_for_slow`` of the solver. The default of Py-BOBYQA is also
    1e-8. DFO-LS uses 1e-4; optimagic is stricter such that slow progress only ends
    an optimization that is really stuck.

    """

    max_slow_iterations: PositiveInt | None = None
    """Number of consecutive slow iterations after which the optimization stops.

    Smaller values stop a stalled optimization earlier.

    This is ``slow.max_slow_iters`` of the solver. With ``None`` the solver uses its
    default, which is 20 times the number of parameters.

    """


@option_group
@dataclass(frozen=True)
class NagNoiseOptions:
    """Options that describe the noise in the function.

    Pass an instance as ``noisy`` if you know how large the noise is. This declares
    the function as noisy, and the solver can stop (or restart) as soon as the
    differences between the function values it compares are no larger than the
    noise. If you do not know how large the noise is, ``noisy=True`` is enough.

    """

    quit_on_noise_level: bool = True
    """Whether to stop (or restart) when all compared function values are within the
    noise level.

    This saves function evaluations that cannot improve the result anymore. It only
    has an effect if you set ``additive_noise_level`` or
    ``multiplicative_noise_level``.

    This is ``noise.quit_on_noise_level`` of the solver. The default is the one the
    solver uses for noisy functions.

    """

    scale_factor_for_quit: NonNegativeFloat = 1.0
    """Multiple of the noise level that counts as "within the noise level".

    Larger values stop earlier. Values much smaller than 1 make the criterion
    useless because differences are rarely that far below the noise.

    This is ``noise.scale_factor_for_quit`` of the solver, from which the default is
    taken.

    """

    multiplicative_noise_level: NonNegativeFloat | None = None
    """Size of the noise relative to the function value.

    Use this if the noise grows with the function value, e.g. 0.01 if the function
    value is off by about one percent. You can only set this or
    ``additive_noise_level``, not both.

    This is ``noise.multiplicative_noise_level`` of the solver, from which the
    default is taken.

    """

    additive_noise_level: NonNegativeFloat | None = None
    """Size of the noise in units of the function value.

    Use this if the noise does not depend on the function value, e.g. the standard
    deviation of repeated evaluations at the same parameters. You can only set this
    or ``multiplicative_noise_level``, not both.

    This is ``noise.additive_noise_level`` of the solver, from which the default is
    taken.

    """


@option_group
@dataclass(frozen=True)
class DfolsInterpolationOptions:
    """Options for fitting the model of the residuals to the evaluated points."""

    precondition: bool = True
    """Whether the linear system that determines the model is rescaled.

    Rescaling makes the solution of the system more accurate. There is rarely a
    reason to switch it off.

    This is ``interpolation.precondition`` of DFO-LS, from which the default is taken.

    """

    throw_error_on_nans: bool = False
    """Whether NaN function values raise an error.

    With ``False``, the optimization ends with an unsuccessful result if the model
    cannot be built because the function returned NaN. With ``True``, a
    ``numpy.linalg.LinAlgError`` is raised instead.

    This is ``interpolation.throw_error_on_nans`` of DFO-LS, from which the default
    is taken.

    """


@option_group
@dataclass(frozen=True)
class PybobyqaInterpolationOptions:
    """Options for fitting the quadratic model to the evaluated points."""

    precondition: bool = True
    """Whether the linear system that determines the model is rescaled.

    Rescaling makes the solution of the system more accurate. There is rarely a
    reason to switch it off.

    This is ``interpolation.precondition`` of Py-BOBYQA, from which the default is
    taken.

    """

    minimum_change_hessian: bool = True
    """Whether a new model keeps its curvature as close as possible to the old one.

    Usually there are fewer points than needed to pin down a quadratic model. With
    ``True``, the remaining freedom is used to change the Hessian of the model as
    little as possible, as in Powell's original BOBYQA. With ``False``, the Hessian
    itself is made as small as possible, which forgets curvature learned in earlier
    iterations.

    This is ``interpolation.minimum_change_hessian`` of Py-BOBYQA, from which the
    default is taken.

    """


@option_group
@dataclass(frozen=True)
class DfolsRegressionOptions:
    """Options for using more interpolation points than strictly needed.

    They are relevant if ``n_interpolation_points`` is larger than the number of
    parameters plus one. The model is then fitted by regression, which averages out
    noise. An iteration normally replaces one point. Replacing more points keeps the
    larger set of points up to date.

    """

    n_extra_steps: NonNegativeInt = 0
    """Number of additional points that are replaced after each successful step.

    Larger values renew the set of points faster and cost that many extra function
    evaluations per successful iteration.

    This is ``regression.num_extra_steps`` of DFO-LS, from which the default is
    taken.

    """

    increase_n_extra_steps_with_restart: NonNegativeInt = 0
    """Amount by which ``n_extra_steps`` grows with each restart.

    Only relevant if restarts are used (see ``restarts``). Useful together with
    ``increase_n_interpolation_points`` of :class:`DfolsRestartOptions`, such that a
    growing set of points is still renewed fast enough.

    This is ``regression.increase_num_extra_steps_with_restart`` of DFO-LS, from
    which the default is taken.

    """

    momentum_extra_steps: bool = False
    """Whether the additional points are placed in the direction of the last step.

    With ``False``, they are placed where they improve the spread of the points most.
    With ``True``, they are placed in random directions that point the same way as
    the last step.

    This is ``regression.momentum_extra_steps`` of DFO-LS, from which the default is
    taken.

    """


@option_group
@dataclass(frozen=True)
class DfolsRestartOptions:
    """Options for restarting DFO-LS instead of stopping.

    Pass an instance as ``restarts`` to use restarts with your own settings. For
    restarts with the default settings, ``restarts=True`` is enough.

    Restarts are the main tool of DFO-LS for noisy functions. A restart is triggered
    when the minimal radius reaches ``convergence_min_trustregion_radius``, when all
    compared function values are within the noise level, when progress is slow, or
    when the solver detects that it is stuck.

    """

    max_unsuccessful_restarts: NonNegativeInt = 10
    """Number of restarts in a row without improvement after which the optimization
    stops.

    Smaller values save function evaluations, larger values try harder to improve
    the result.

    This is ``restarts.max_unsuccessful_restarts`` of DFO-LS, from which the
    default is taken.

    """

    convergence_min_radius_scale: NonNegativeFloat = 1.0
    """Factor by which ``convergence_min_trustregion_radius`` is multiplied at each
    restart.

    Values below 1 let every restart run to a higher precision than the previous
    one.

    This is ``restarts.rhoend_scale`` of DFO-LS, from which the default is taken.

    """

    use_soft_restarts: bool = True
    """Whether restarts are soft or hard.

    A soft restart keeps most of the evaluated points and only moves a few of them,
    which is cheap. A hard restart builds a completely new set of points around the
    best point found so far.

    This is ``restarts.use_soft_restarts`` of DFO-LS, from which the default is
    taken.

    """

    soft_n_geometry_steps: NonNegativeInt = 3
    """Number of points that are moved at a soft restart.

    More points make a soft restart more similar to a hard one and cost that many
    function evaluations.

    This is ``restarts.soft.num_geom_steps`` of DFO-LS, from which the default is
    taken.

    """

    soft_move_iterate: bool = True
    """Whether a soft restart continues from the best of the newly evaluated points.

    With ``False``, it continues from the point at which the previous run ended.

    This is ``restarts.soft.move_xk`` of DFO-LS, from which the default is taken.

    """

    soft_max_fake_successful_steps: PositiveInt | None = None
    """Number of steps after a soft restart that may improve the current run without
    beating the best value of earlier runs.

    When the number is reached, the optimization stops without success. Smaller
    values end runs that do not catch up with earlier runs sooner.

    This is ``restarts.soft.max_fake_successful_steps`` of DFO-LS. With ``None``
    DFO-LS uses its default, which is ``stopping_maxfun``.

    """

    hard_use_old_fun_value: bool = True
    """Whether a hard restart reuses the function value at the best point.

    This saves one function evaluation per hard restart. Set it to ``False`` for
    noisy functions if you want a fresh evaluation at the best point.

    This is ``restarts.hard.use_old_rk`` of DFO-LS, from which the default is taken.

    """

    auto_detect: bool = True
    """Whether the solver also restarts when it detects that it is stuck.

    The solver then restarts before the minimal radius is reached, if the
    trust-region radius only shrank recently and the model keeps changing more and
    more from iteration to iteration. This is typical for noise.

    This is ``restarts.auto_detect`` of DFO-LS, from which the default is taken.

    """

    auto_detect_history: PositiveInt = 30
    """Number of recent iterations that are used to detect that the solver is stuck.

    Larger values make the detection slower but less likely to trigger by accident.

    This is ``restarts.auto_detect.history`` of DFO-LS, from which the default is
    taken.

    """

    auto_detect_min_model_change_slope: NonNegativeFloat = 0.015
    """Rate at which the model changes must grow to trigger a restart.

    The rate is the slope of a line fitted to the logarithm of the model changes over
    the recent iterations. Larger values trigger fewer restarts.

    This is ``restarts.auto_detect.min_chgJ_slope`` of DFO-LS, from which the default
    is taken.

    """

    auto_detect_min_correlation: ProbabilityFloat = 0.1
    """Correlation between iteration number and model changes needed to trigger a
    restart.

    Larger values require a clearer upward trend of the model changes and trigger
    fewer restarts.

    This is ``restarts.auto_detect.min_correl`` of DFO-LS, from which the default
    is taken.

    """

    increase_n_interpolation_points: bool = False
    """Whether the number of interpolation points grows with each restart.

    More points average out noise better. Each restart then becomes more robust and
    more expensive than the previous one.

    This is ``restarts.increase_npt`` of DFO-LS, from which the default is taken.

    """

    increase_n_interpolation_points_amount: NonNegativeInt = 1
    """Number of interpolation points that are added at each restart.

    Only relevant if ``increase_n_interpolation_points`` is ``True``.

    This is ``restarts.increase_npt_amt`` of DFO-LS, from which the default is taken.

    """

    hard_increase_n_initial_directions_amount: NonNegativeInt = 1
    """Amount by which ``n_initial_directions`` of the ``growing_options`` grows with
    each hard restart.

    Only relevant if you start with fewer points than usual (see
    ``growing_options``). Set it to the same value as
    ``increase_n_interpolation_points_amount`` to avoid a growing phase after
    restarts.

    This is ``restarts.hard.increase_ndirs_initial_amt`` of DFO-LS, from which the
    default is taken.

    """

    max_n_interpolation_points: PositiveInt | None = None
    """Largest number of interpolation points that restarts may reach.

    Only relevant if ``increase_n_interpolation_points`` is ``True``.

    This is ``restarts.max_npt`` of DFO-LS. With ``None`` DFO-LS uses its default,
    which is ``n_interpolation_points``. You therefore have to set this option if the
    number of points should grow.

    """

    throw_error_on_nans: bool = False
    """Whether NaN function values raise an error when the solver checks if it is
    stuck.

    With ``False``, the optimization ends with an unsuccessful result. With ``True``,
    a ``numpy.linalg.LinAlgError`` is raised instead.

    This is ``restarts.throw_error_on_nans`` of DFO-LS, from which the default is
    taken.

    """


@option_group
@dataclass(frozen=True)
class PybobyqaRestartOptions:
    """Options for restarting Py-BOBYQA instead of stopping.

    Pass an instance as ``restarts`` to use restarts with your own settings. For
    restarts with the default settings, ``restarts=True`` is enough.

    Restarts are the main tool of Py-BOBYQA for noisy functions and for escaping
    local optima. A restart is triggered when the minimal radius reaches
    ``convergence_min_trustregion_radius``, when all compared function values are
    within the noise level, when progress is slow, or when the solver detects that it
    is stuck.

    """

    max_unsuccessful_restarts: NonNegativeInt = 10
    """Number of restarts in a row without improvement after which the optimization
    stops.

    Smaller values save function evaluations, larger values try harder to improve
    the result.

    This is ``restarts.max_unsuccessful_restarts`` of Py-BOBYQA, from which the
    default is taken.

    """

    convergence_min_radius_scale: NonNegativeFloat = 1.0
    """Factor by which ``convergence_min_trustregion_radius`` is multiplied at each
    restart.

    Values below 1 let every restart run to a higher precision than the previous
    one.

    This is ``restarts.rhoend_scale`` of Py-BOBYQA, from which the default is taken.

    """

    use_soft_restarts: bool = True
    """Whether restarts are soft or hard.

    A soft restart keeps most of the evaluated points and only moves a few of them,
    which is cheap. A hard restart builds a completely new set of points around the
    best point found so far.

    This is ``restarts.use_soft_restarts`` of Py-BOBYQA, from which the default is
    taken.

    """

    soft_n_geometry_steps: NonNegativeInt = 3
    """Number of points that are moved at a soft restart.

    More points make a soft restart more similar to a hard one and cost that many
    function evaluations.

    This is ``restarts.soft.num_geom_steps`` of Py-BOBYQA, from which the default is
    taken.

    """

    soft_move_iterate: bool = True
    """Whether a soft restart continues from the best of the newly evaluated points.

    With ``False``, it continues from the point at which the previous run ended.

    This is ``restarts.soft.move_xk`` of Py-BOBYQA, from which the default is taken.

    """

    soft_max_fake_successful_steps: PositiveInt | None = None
    """Number of steps after a soft restart that may improve the current run without
    beating the best value of earlier runs.

    When the number is reached, the optimization stops without success. Smaller
    values end runs that do not catch up with earlier runs sooner.

    This is ``restarts.soft.max_fake_successful_steps`` of Py-BOBYQA. With ``None``
    Py-BOBYQA uses its default, which is ``stopping_maxfun``.

    """

    hard_use_old_fun_value: bool = True
    """Whether a hard restart reuses the function value at the best point.

    This saves one function evaluation per hard restart. Set it to ``False`` for
    noisy functions if you want a fresh evaluation at the best point.

    This is ``restarts.hard.use_old_fk`` of Py-BOBYQA, from which the default is
    taken.

    """

    auto_detect: bool = True
    """Whether the solver also restarts when it detects that it is stuck.

    The solver then restarts before the minimal radius is reached, if the
    trust-region radius only shrank recently and the model keeps changing more and
    more from iteration to iteration. This is typical for noise.

    This is ``restarts.auto_detect`` of Py-BOBYQA, from which the default is taken.

    """

    auto_detect_history: PositiveInt = 30
    """Number of recent iterations that are used to detect that the solver is stuck.

    Larger values make the detection slower but less likely to trigger by accident.

    This is ``restarts.auto_detect.history`` of Py-BOBYQA, from which the default is
    taken.

    """

    auto_detect_min_model_change_slope: NonNegativeFloat = 0.015
    """Rate at which the model changes must grow to trigger a restart.

    The rate is the slope of a line fitted to the logarithm of the model changes over
    the recent iterations. Larger values trigger fewer restarts.

    This is ``restarts.auto_detect.min_chg_model_slope`` of Py-BOBYQA, from which the
    default is taken.

    """

    auto_detect_min_correlation: ProbabilityFloat = 0.1
    """Correlation between iteration number and model changes needed to trigger a
    restart.

    Larger values require a clearer upward trend of the model changes and trigger
    fewer restarts.

    This is ``restarts.auto_detect.min_correl`` of Py-BOBYQA, from which the default
    is taken.

    """

    max_unsuccessful_restarts_total: NonNegativeInt | None = None
    """Total number of restarts without improvement after which the optimization
    stops.

    In contrast to ``max_unsuccessful_restarts``, the restarts do not have to be in a
    row. Smaller values limit the cost of a search for a global optimum.

    This is ``restarts.max_unsuccessful_restarts_total`` of Py-BOBYQA. With ``None``
    Py-BOBYQA uses its default, which is 20 if ``seek_global_optimum`` is ``True`` and
    ``stopping_maxfun`` otherwise.

    """

    initial_radius_scale_after_unsuccessful_restart: PositiveFloat | None = None
    """Factor by which the initial trust-region radius grows after a restart without
    improvement.

    Values above 1 make every further restart search a wider region, which helps to
    leave a local optimum.

    This is ``restarts.rhobeg_scale_after_unsuccessful_restart`` of Py-BOBYQA. With
    ``None`` Py-BOBYQA uses its default, which is 1.1 if ``seek_global_optimum`` is
    ``True`` and 1 otherwise.

    """


@option_group
@dataclass(frozen=True)
class DfolsGrowingOptions:
    """Options for starting with fewer function evaluations than usual.

    DFO-LS normally evaluates the function once per parameter before it takes its
    first step. If this is too expensive, set ``n_initial_directions`` to a smaller
    number. DFO-LS then starts to take steps earlier and adds the missing points
    while it runs. This is called the growing phase. If ``n_initial_directions`` is
    not set, the other options of this group have no effect.

    """

    n_initial_directions: PositiveInt | None = None
    """Number of points that are evaluated before the first step, not counting the
    start parameters.

    Smaller values let DFO-LS make progress after very few function evaluations, at
    the price of less reliable first steps. Only use values below the number of
    parameters, and keep ``n_interpolation_points`` at its default.

    This is ``growing.ndirs_initial`` of DFO-LS. With ``None`` DFO-LS uses its
    default, which is ``n_interpolation_points - 1``, i.e. no growing phase.

    """

    n_new_directions_each_iteration: NonNegativeInt = 0
    """Number of extra points that are added in each iteration of the growing phase.

    Larger values end the growing phase sooner but spend function evaluations on
    points that are not steps of the optimization. DFO-LS does not recommend this.

    This is ``growing.num_new_dirns_each_iter`` of DFO-LS, from which the default is
    taken.

    """

    radius_scale_new_directions: PositiveFloat | None = None
    """Length of the steps that explore new directions, as a multiple of the
    trust-region radius.

    This is ``growing.delta_scale_new_dirns`` of DFO-LS. With ``None`` DFO-LS uses its
    default, which is 1, or 0.1 if the trust-region step is perturbed (see
    ``perturb_trust_region_step``).

    """

    do_geometry_steps: bool = False
    """Whether points are moved to improve the spread of the points during the growing
    phase.

    Such steps cost function evaluations that do not reduce the function value
    directly, which works against the purpose of a cheap start.

    This is ``growing.do_geom_steps`` of DFO-LS, from which the default is taken.

    """

    reset_radius: bool = False
    """Whether the trust-region radius is set back to its initial value when the
    growing phase ends.

    Use this if the radius shrinks too much while the model is still incomplete.

    This is ``growing.reset_delta`` of DFO-LS, from which the default is taken.

    """

    reset_min_radius: bool = False
    """Whether the minimal radius is set back to its initial value when the growing
    phase ends.

    Requires ``reset_radius`` to be ``True``.

    This is ``growing.reset_rho`` of DFO-LS, from which the default is taken.

    """

    shrinking_factor: ProbabilityFloat | None = None
    """Factor by which the trust-region radius is multiplied after a bad step during
    the growing phase.

    Values close to 1 keep the radius large while the model is still incomplete.

    This is ``growing.gamma_dec`` of DFO-LS. With ``None`` DFO-LS uses its default,
    which is 0.5, or 0.98 for noisy functions (see ``noisy``). It does not follow a
    value you set for ``shrinking_factor`` of the ``trustregion_options``.

    """

    do_safety_step: bool = True
    """Whether very short steps are replaced by safety steps during the growing phase.

    A safety step does not evaluate the function at a point that is very close to
    the current one. It shrinks the radius or improves the set of points instead.

    This is ``growing.safety.do_safety_step`` of DFO-LS, from which the default is
    taken.

    """

    safety_reduce_radius: bool = False
    """Whether safety steps shrink the trust-region radius during the growing phase.

    Cannot be combined with ``safety_full_geometry_step``.

    This is ``growing.safety.reduce_delta`` of DFO-LS, from which the default is
    taken.

    """

    safety_full_geometry_step: bool = False
    """Whether safety steps also improve the spread of the points during the growing
    phase, as they do after the growing phase.

    Cannot be combined with ``safety_reduce_radius``.

    This is ``growing.safety.full_geom_step`` of DFO-LS, from which the default is
    taken.

    """

    use_full_rank_interpolation: bool | None = None
    """Whether the incomplete model is filled up such that steps can go in all
    directions.

    During the growing phase the model only knows the directions explored so far.
    With ``True``, DFO-LS adds small artificial components for the other directions.
    The alternative is ``perturb_trust_region_step``. Exactly one of the two must be
    ``True`` if you set them.

    This is ``growing.full_rank.use_full_rank_interp`` of DFO-LS. With ``None``
    DFO-LS chooses: ``True`` if there are at least as many residuals as parameters
    and ``False`` otherwise.

    """

    perturb_trust_region_step: bool | None = None
    """Whether each step is perturbed in a direction that has not been explored yet.

    This is the alternative to ``use_full_rank_interpolation``. It is the method that
    works if there are fewer residuals than parameters.

    This is ``growing.perturb_trust_region_step`` of DFO-LS. With ``None`` DFO-LS
    chooses: ``True`` if there are fewer residuals than parameters and ``False``
    otherwise.

    """

    full_rank_scale_factor: NonNegativeFloat = 1e-2
    """Size of the artificial components that fill up the incomplete model.

    Larger values make the steps move more into unexplored directions. Only relevant
    if the model is filled up (see ``use_full_rank_interpolation``).

    This is ``growing.full_rank.scale_factor`` of DFO-LS, from which the default is
    taken.

    """

    full_rank_svd_scale_factor: ProbabilityFloat = 1.0
    """Lower bound for the singular values of the filled-up model Jacobian, as a
    multiple of its smallest nonzero singular value.

    Only relevant if the model is filled up (see ``use_full_rank_interpolation``).

    This is ``growing.full_rank.svd_scale_factor`` of DFO-LS, from which the default
    is taken.

    """

    full_rank_min_singular_value: ProbabilityFloat = 1e-6
    """Absolute lower bound for the singular values of the filled-up model Jacobian.

    Only relevant if the model is filled up (see ``use_full_rank_interpolation``).

    This is ``growing.full_rank.min_sing_val`` of DFO-LS, from which the default is
    taken.

    """

    full_rank_max_jacobian_condition_number: PositiveFloat = 1e8
    """Largest condition number the filled-up model Jacobian may have.

    Smaller values make the first steps more stable but less informed by the points
    evaluated so far. Only relevant if the model is filled up (see
    ``use_full_rank_interpolation``).

    This is ``growing.full_rank.svd_max_jac_cond`` of DFO-LS, from which the default
    is taken.

    """


# ======================================================================================
# Deprecated options
# ======================================================================================

_REMOVED_IN = "0.7.0"

_OLD_SLOW_PROGRESS_KEYS = {
    "threshold_to_characterize_as_slow": "slow_progress_options.threshold_for_slow",
    "max_insufficient_improvements": "slow_progress_options.max_slow_iterations",
    "comparison_period": "slow_progress_options.history_for_slow",
}

_OLD_RESET_KEYS = {
    "minimal_trustregion_radius_tolerance_scaling_at_reset": (
        "restarts.convergence_min_radius_scale"
    ),
    "move_center_at_soft_reset": "restarts.soft_move_iterate",
    "reuse_criterion_value_at_hard_reset": "restarts.hard_use_old_fun_value",
    "max_iterations_without_new_best_after_soft_reset": (
        "restarts.soft_max_fake_successful_steps"
    ),
    "auto_detect": "restarts.auto_detect",
    "auto_detect_history": "restarts.auto_detect_history",
    "auto_detect_min_jacobian_increase": (
        "restarts.auto_detect_min_model_change_slope"
    ),
    "auto_detect_min_correlations": "restarts.auto_detect_min_correlation",
    "points_to_replace_at_soft_reset": "restarts.soft_n_geometry_steps",
    "max_consecutive_unsuccessful_resets": "restarts.max_unsuccessful_restarts",
}

_OLD_PYBOBYQA_RESET_KEYS = {
    **_OLD_RESET_KEYS,
    "max_unsuccessful_resets": "restarts.max_unsuccessful_restarts_total",
    "trust_region_scaling_at_unsuccessful_reset": (
        "restarts.initial_radius_scale_after_unsuccessful_restart"
    ),
}

_OLD_DFOLS_RESET_KEYS = {
    **_OLD_RESET_KEYS,
    "max_interpolation_points": "restarts.max_n_interpolation_points",
    "n_additional_extra_points_to_replace_per_reset": (
        "regression_options.increase_n_extra_steps_with_restart"
    ),
}

_OLD_FAST_START_KEYS = {
    "min_inital_points": "growing_options.n_initial_directions",
    "scale_of_trustregion_step_perturbation": (
        "growing_options.radius_scale_new_directions"
    ),
    "scale_of_jacobian_components_perturbation": (
        "growing_options.full_rank_scale_factor"
    ),
    "floor_of_jacobian_singular_values": "growing_options.full_rank_svd_scale_factor",
    "jacobian_max_condition_number": (
        "growing_options.full_rank_max_jacobian_condition_number"
    ),
    "geometry_improving_steps": "growing_options.do_geometry_steps",
    "safety_steps": "growing_options.do_safety_step",
    "shrink_upper_radius_in_safety_steps": "growing_options.safety_reduce_radius",
    "full_geometry_improving_step": "growing_options.safety_full_geometry_step",
    "reset_trustregion_radius_after_fast_start": "growing_options.reset_radius",
    "reset_min_trustregion_radius_after_fast_start": "growing_options.reset_min_radius",
    "shrinking_factor_not_successful": "growing_options.shrinking_factor",
    "n_extra_search_directions_per_iteration": (
        "growing_options.n_new_directions_each_iteration"
    ),
}


def _rename_old_keys(
    value: Mapping[str, Any] | None,
    key_map: Mapping[str, str],
    option: str,
    special_keys: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Translate the entries of a deprecated dictionary that were simply renamed.

    Only the entries the user passed are translated. Entries in ``special_keys`` are
    skipped because they are handled by the caller.

    """
    if value is None:
        return {}
    unknown = sorted(set(value) - set(key_map) - set(special_keys))
    if unknown:
        valid = sorted([*key_map, *special_keys])
        msg = (
            f"The deprecated option `{option}` has no entries {unknown}. Valid "
            f"entries were {valid}."
        )
        raise InvalidAlgoOptionError(msg)
    return {key_map[key]: val for key, val in value.items() if key in key_map}


def _translate_slow_progress(
    value: Mapping[str, Any] | None, options: Mapping[str, Any]
) -> dict[str, Any]:
    return _rename_old_keys(
        value, _OLD_SLOW_PROGRESS_KEYS, option="convergence_slow_progress"
    )


def _translate_reset_type(value: Mapping[str, Any]) -> dict[str, Any]:
    if "reset_type" not in value:
        return {}
    if value["reset_type"] not in ("soft", "hard"):
        msg = (
            "The entry `reset_type` of the deprecated option "
            "`trustregion_reset_options` must be 'soft' or 'hard'."
        )
        raise InvalidAlgoOptionError(msg)
    return {"restarts.use_soft_restarts": value["reset_type"] == "soft"}


def _declares_noise(options: Mapping[str, Any]) -> bool:
    """Whether the options of one call declare the function as noisy.

    In the old wrappers, setting a noise level was the only way to do so.

    """
    return bool(
        options.get("noise_additive_level")
        or options.get("noise_multiplicative_level")
        or options.get("noisy")
    )


def _apply_use_resets(
    value: Mapping[str, Any] | None,
    entries: dict[str, Any],
    restarts_by_default: bool,
) -> dict[str, Any]:
    """Translate the old entry ``use_resets`` and decide which entries apply.

    The old default of ``use_resets`` was None, which left the decision to the solver.
    The other entries of the old dictionary only had an effect if restarts were used.
    The new ``restarts`` option cannot hold entries without switching restarts on, so
    the entries are only translated if restarts were used.

    Args:
        value: The deprecated ``trustregion_reset_options``.
        entries: The translation of all entries except ``use_resets``.
        restarts_by_default: Whether the solver restarts if ``use_resets`` is None.
            This can only be read from the options that are passed together with
            the deprecated dictionary, not from an algorithm they are applied to.

    """
    use_resets = (value or {}).get("use_resets")
    if use_resets is None:
        return entries if restarts_by_default else {}
    if not use_resets:
        return {"restarts": False}
    if not any(target.startswith("restarts.") for target in entries):
        entries["restarts"] = True
    return entries


def _translate_pybobyqa_reset_options(
    value: Mapping[str, Any] | None, options: Mapping[str, Any]
) -> dict[str, Any]:
    out = _rename_old_keys(
        value,
        _OLD_PYBOBYQA_RESET_KEYS,
        option="trustregion_reset_options",
        special_keys=("use_resets", "reset_type"),
    )
    out |= _translate_reset_type(value or {})
    restarts_by_default = _declares_noise(options) or bool(
        options.get("seek_global_optimum")
    )
    return _apply_use_resets(value, out, restarts_by_default)


def _translate_dfols_reset_options(
    value: Mapping[str, Any] | None, options: Mapping[str, Any]
) -> dict[str, Any]:
    soft_key = "n_extra_interpolation_points_per_soft_reset"
    hard_key = "n_extra_interpolation_points_per_hard_reset"
    out = _rename_old_keys(
        value,
        _OLD_DFOLS_RESET_KEYS,
        option="trustregion_reset_options",
        special_keys=("use_resets", "reset_type", soft_key, hard_key),
    )
    original = value
    value = value or {}
    out |= _translate_reset_type(value)

    n_soft = value.get(soft_key, 0)
    if soft_key in value:
        out["restarts.increase_n_interpolation_points"] = n_soft > 0
        if n_soft > 0:
            out["restarts.increase_n_interpolation_points_amount"] = n_soft
    if hard_key in value:
        # The old entry was the number of points added per hard restart. DFO-LS wants
        # the additional growth of the initial directions on top of the points that
        # are added at every restart.
        if value[hard_key] < n_soft:
            msg = (
                f"The entry `{hard_key}` of the deprecated option "
                f"`trustregion_reset_options` must not be smaller than `{soft_key}`."
            )
            raise InvalidAlgoOptionError(msg)
        out["restarts.hard_increase_n_initial_directions_amount"] = (
            value[hard_key] - n_soft
        )
    return _apply_use_resets(original, out, _declares_noise(options))


def _translate_fast_start_options(
    value: Mapping[str, Any] | None, options: Mapping[str, Any]
) -> dict[str, Any]:
    out = _rename_old_keys(
        value,
        _OLD_FAST_START_KEYS,
        option="trustregion_fast_start_options",
        special_keys=("method",),
    )
    method = (value or {}).get("method", "auto")
    if method not in ("auto", "jacobian", "trustregion"):
        msg = (
            "The entry `method` of the deprecated option "
            "`trustregion_fast_start_options` must be 'auto', 'jacobian' or "
            "'trustregion'."
        )
        raise InvalidAlgoOptionError(msg)
    if method != "auto":
        out["growing_options.use_full_rank_interpolation"] = method == "jacobian"
        out["growing_options.perturb_trust_region_step"] = method == "trustregion"
    return out


def _translate_initial_directions(
    value: str, options: Mapping[str, Any]
) -> dict[str, Any]:
    if value not in ("coordinate", "random"):
        msg = (
            "The deprecated option `initial_directions` must be 'coordinate' or "
            "'random'."
        )
        raise InvalidAlgoOptionError(msg)
    return {"init_options.random_initial_directions": value == "random"}


def _translate_method_to_replace_extra_points(
    value: str | None, options: Mapping[str, Any]
) -> dict[str, Any]:
    if value not in ("geometry_improving", "momentum", None):
        msg = (
            "The deprecated option `trustregion_method_to_replace_extra_points` must "
            "be 'geometry_improving', 'momentum' or None."
        )
        raise InvalidAlgoOptionError(msg)
    return {"regression_options.momentum_extra_steps": value == "momentum"}


def _translate_noise_tolerance(
    value: float, options: Mapping[str, Any]
) -> dict[str, Any]:
    # The tolerance only had an effect together with a noise level. If `noisy` is
    # passed as well, the entry is returned such that mixing old and new raises.
    has_noise_level = bool(
        options.get("noise_additive_level") or options.get("noise_multiplicative_level")
    )
    if not has_noise_level and "noisy" not in options:
        return {}
    # A tolerance of zero used to switch the criterion off.
    if value == 0:
        return {"noisy.quit_on_noise_level": False}
    return {"noisy.scale_factor_for_quit": value}


def _translate_additive_noise_level(
    value: float | None, options: Mapping[str, Any]
) -> dict[str, Any]:
    # Setting a noise level used to be the only way to declare the function as noisy.
    # A level of None or zero left the function without noise and had no effect.
    return {"noisy.additive_noise_level": value} if value else {}


def _translate_multiplicative_noise_level(
    value: float | None, options: Mapping[str, Any]
) -> dict[str, Any]:
    return {"noisy.multiplicative_noise_level": value} if value else {}


def _deprecated(
    name: str,
    replacement: str | Callable[[Any, Mapping[str, Any]], dict[str, Any]],
) -> DeprecatedOption:
    return DeprecatedOption(name, replacement, removed_in=_REMOVED_IN)


_COMMON_DEPRECATED_OPTIONS = (
    _deprecated(
        "convergence_minimal_trustregion_radius_tolerance",
        "convergence_min_trustregion_radius",
    ),
    _deprecated("interpolation_rounding_error", "rounding_error_constant"),
    _deprecated("threshold_for_safety_step", "safety_step_threshold"),
    _deprecated("trustregion_n_interpolation_points", "n_interpolation_points"),
    _deprecated("noise_n_evals_per_point", "n_evals_per_point"),
    _deprecated("initial_directions", _translate_initial_directions),
    _deprecated(
        "random_directions_orthogonal", "init_options.random_directions_make_orthogonal"
    ),
    _deprecated(
        "trustregion_threshold_successful", "trustregion_options.shrinking_threshold"
    ),
    _deprecated(
        "trustregion_threshold_very_successful",
        "trustregion_options.expansion_threshold",
    ),
    _deprecated(
        "trustregion_shrinking_factor_not_successful",
        "trustregion_options.shrinking_factor",
    ),
    _deprecated(
        "trustregion_expansion_factor_successful",
        "trustregion_options.expansion_factor",
    ),
    _deprecated(
        "trustregion_expansion_factor_very_successful",
        "trustregion_options.step_expansion_factor",
    ),
    _deprecated(
        "trustregion_shrinking_factor_lower_radius",
        "trustregion_options.min_radius_shrinking_factor",
    ),
    _deprecated(
        "trustregion_shrinking_factor_upper_radius",
        "trustregion_options.radius_factor_after_min_radius_shrinking",
    ),
    _deprecated(
        "trustregion_precondition_interpolation", "interpolation_options.precondition"
    ),
    _deprecated("convergence_slow_progress", _translate_slow_progress),
    _deprecated(
        "convergence_noise_corrected_criterion_tolerance", _translate_noise_tolerance
    ),
    _deprecated("noise_additive_level", _translate_additive_noise_level),
    _deprecated("noise_multiplicative_level", _translate_multiplicative_noise_level),
)

DFOLS_DEPRECATED_OPTIONS: tuple[DeprecatedOption, ...] = (
    *_COMMON_DEPRECATED_OPTIONS,
    _deprecated("clip_criterion_if_overflowing", "check_fun_for_overflow"),
    _deprecated(
        "trustregion_n_extra_points_to_replace_successful",
        "regression_options.n_extra_steps",
    ),
    _deprecated(
        "trustregion_method_to_replace_extra_points",
        _translate_method_to_replace_extra_points,
    ),
    _deprecated("trustregion_reset_options", _translate_dfols_reset_options),
    _deprecated("trustregion_fast_start_options", _translate_fast_start_options),
)
"""Old option names of ``nag_dfols`` that still work but warn."""

PYBOBYQA_DEPRECATED_OPTIONS: tuple[DeprecatedOption, ...] = (
    *_COMMON_DEPRECATED_OPTIONS,
    # Py-BOBYQA only catches an OverflowError raised by the function. optimagic
    # handles errors of the function itself, so the option never had an effect.
    _deprecated("clip_criterion_if_overflowing", lambda value, options: {}),
    _deprecated("stopping_max_criterion_evaluations", "stopping_maxfun"),
    _deprecated("convergence_criterion_value", "convergence_target_value"),
    _deprecated(
        "trustregion_minimum_change_hession_for_underdetermined_interpolation",
        "interpolation_options.minimum_change_hessian",
    ),
    _deprecated("trustregion_reset_options", _translate_pybobyqa_reset_options),
)
"""Old option names of ``nag_pybobyqa`` that still work but warn."""


# ======================================================================================
# Algorithms
# ======================================================================================


@mark.deprecated_options(DFOLS_DEPRECATED_OPTIONS)
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
    r"""Minimize a least-squares function without derivatives using DFO-LS.

    DFO-LS :cite:`Cartis2018b` is a derivative-free trust-region algorithm developed
    by the Numerical Algorithms Group and researchers at the University of Oxford. It
    solves nonlinear least-squares problems with optional bounds:

    .. math::

        \min_{x\in\mathbb{R}^n}  &\quad  f(x) := \sum_{i=1}^{m}r_{i}(x)^2 \\
        \text{s.t.} &\quad  \text{lower\_bounds} \leq x \leq \text{upper\_bounds}

    To use it, decorate your function with ``om.mark.least_squares`` and return the
    residuals :math:`r_i` instead of their sum of squares.

    DFO-LS does not use derivatives and does not approximate them with finite
    differences. It evaluates the residuals at a set of points around the current
    best parameters, fits a linear model of each residual to these points and
    minimizes the resulting model of :math:`f` within a trust region.

    **When to use it**

    - The function is noisy, i.e. repeated evaluations at the same parameters give
      different values, for example because it contains a simulation. Derivatives
      are then unreliable. Set ``noisy=True``; see also ``n_evals_per_point``
      and ``restarts``.
    - The function is expensive and derivatives are only available via finite
      differences. DFO-LS typically needs few function evaluations. If even one
      evaluation per parameter is too much before the first step, see
      ``growing_options``.

    If your function has no least-squares structure, use ``nag_pybobyqa``, the
    scalar counterpart from the same authors. If it has, DFO-LS typically needs far
    fewer function evaluations because it uses the residuals. ``tranquilo_ls`` and
    ``pounders`` are alternatives of the same type; DFO-LS stands out by its options
    for noisy functions.

    **When the optimization stops**

    The main convergence criterion is ``convergence_min_trustregion_radius``: the
    optimization converged when the solver cannot improve the result by steps that
    are longer than this value. The optimization also ends successfully when the sum
    of squares is practically zero (``convergence_target_value``,
    ``convergence_ftol_scaled``), or, for noisy functions, when all compared function
    values are within the noise level (``noisy``). It ends without success if
    progress is very slow for many iterations (``slow_progress_options``) or if
    ``stopping_maxfun`` is reached. If restarts are used (``restarts``, the default
    for noisy functions), the solver restarts instead of stopping.

    **What the result contains**

    In addition to the usual entries, the result contains ``jac``, the Jacobian of
    the residuals at the solution. It is an estimate from the model that DFO-LS
    fitted to the evaluated points, not an exact derivative, and it refers to the
    internal parameters, i.e. after scaling and constraints were applied.
    ``algorithm_output`` contains the number of runs (``n_runs``, more than one if
    there were restarts) and the number of points at which the function was
    evaluated (``n_evaluated_points``).

    **Options of DFO-LS that are not available**

    - Logging and printing options and ``argsf``: optimagic handles logging and
      additional arguments of the function.
    - ``scaling_within_bounds``: use the ``scaling`` argument of ``minimize``.
    - ``init.run_in_parallel``: it does not evaluate the function in parallel.
    - Regularizers (``h``, ``lh``, ``prox_uh`` and the related options): not supported
      yet. Ridge penalties can be added as additional residuals.
    - Convex constraints given as projections (``projections`` and the related
      options): not supported yet. Many convex sets can be expressed with
      optimagic's constraints.
    - Starting from a database of earlier evaluations: not supported yet.

    Cite :cite:`Cartis2018b` in addition to optimagic when using DFO-LS. The options
    are described in more technical detail in the `DFO-LS documentation
    <https://numericalalgorithmsgroup.github.io/dfols/>`_.

    """

    stopping_maxfun: PositiveInt = STOPPING_MAXFUN
    """Maximum number of function evaluations.

    If the number is reached, the optimization stops without success. Repeated
    evaluations at the same parameters (see ``n_evals_per_point``) count separately.

    This is ``maxfun`` of DFO-LS. optimagic uses its common default instead of the
    much smaller default of DFO-LS (100 evaluations per parameter, at most 1000).

    """

    convergence_min_trustregion_radius: NonNegativeFloat = 1e-8
    """Stop when the minimal radius of the trust region reaches this value.

    This is the main convergence criterion. The value is roughly the precision of
    the solution in units of the parameters: the solver cannot improve the result
    anymore with steps that are longer than this. Larger values stop earlier and
    save function evaluations. For noisy functions, a precision far below the noise
    is not reachable and larger values such as 1e-4 are reasonable. If restarts are
    used, reaching the value triggers a restart instead.

    This is ``rhoend`` of DFO-LS, from which the default is taken.

    """

    convergence_target_value: NonNegativeFloat = 1e-12
    """Stop when the sum of squared residuals is at or below this value.

    A sum of squares of practically zero cannot be improved. Increase the value if a
    fit that is good enough is all you need. With 0 the optimization only stops if
    all residuals are exactly zero.

    This is ``model.abs_tol`` of DFO-LS, from which the default is taken.

    """

    convergence_ftol_scaled: ProbabilityFloat = 0.0
    r"""Stop when the sum of squared residuals fell to this fraction of its value at
    the start parameters.

    The optimization stops successfully as soon as
    :math:`f(x_k) / f(x_0) \leq \textsf{convergence\_ftol\_scaled}`. With the default
    of 0 the criterion is switched off.

    This is ``model.rel_tol`` of DFO-LS, which uses a default of 1e-20 that
    practically never applies.

    """

    trustregion_initial_radius: PositiveFloat | None = None
    r"""Initial radius of the trust region, i.e. the length of the first steps.

    It is in units of the parameters. Choose it such that a step of this length
    changes the function noticeably but stays in a region where the function can be
    evaluated. Too small values waste the first iterations, too large values make
    the first model unreliable. With bounds, the distance between lower and upper
    bound must be at least twice the radius.

    This is ``rhobeg`` of DFO-LS. With ``None`` DFO-LS uses its default, which is
    :math:`0.1 \max(\|x_0\|_{\infty}, 1)`.

    """

    n_interpolation_points: PositiveInt | None = None
    """Number of points to which the model of the residuals is fitted.

    More points than the minimum of ``len(x) + 1`` make the model less sensitive to
    noise, because it is then fitted by regression. Each additional point costs one
    function evaluation at the start. See also ``regression_options``.

    This is ``npt`` of DFO-LS. With ``None`` DFO-LS uses its default, which is
    ``len(x) + 1``.

    """

    noisy: bool | NagNoiseOptions = False
    """Whether the function is noisy and, optionally, how large the noise is.

    A function is noisy if repeated evaluations at the same parameters give
    different values. With ``True``, DFO-LS becomes more careful: it restarts instead
    of stopping and shrinks the trust region more slowly. These are the defaults of
    the options that say so in their description; you can still set each of them.

    If you know how large the noise is, pass an instance of
    :class:`NagNoiseOptions` instead of ``True``. This also declares the function as
    noisy. DFO-LS can then stop (or restart) as soon as all function values it
    compares are within the noise level.

    ``True`` and an instance both set ``objfun_has_noise`` of DFO-LS, from which the
    default is taken. The entries of the instance are the ``noise.*`` options.

    """

    n_evals_per_point: PositiveInt | Callable[[float, float, int, int], int] | None = (
        None
    )
    """How often the function is evaluated at each point.

    For noisy functions, the solver can evaluate the function several times at the
    same parameters and average the residuals. More evaluations reduce the noise and
    multiply the cost. You can pass a number or a function that returns the number
    and is called as ``f(radius, min_radius, n_iterations, n_restarts)`` with the
    current trust-region radius, the current minimal radius, the number of
    iterations of the current run and the number of restarts so far. A function
    allows to average more when the trust region is small, where noise hurts most.

    This is ``nsamples`` of DFO-LS. With ``None`` DFO-LS uses its default, which is
    one evaluation per point.

    """

    rounding_error_constant: NonNegativeFloat = 0.1
    """Controls how often the solver re-centers its internal coordinates.

    The solver stores points relative to a base point to limit rounding errors. The
    base point is moved to the current best point when the step is shorter than this
    value times the distance between the two. Larger values re-center more often.
    There is rarely a reason to change it.

    This is ``general.rounding_error_constant`` of DFO-LS, from which the default is
    taken.

    """

    safety_step_threshold: NonNegativeFloat = 0.5
    """Steps shorter than this multiple of the minimal radius are not evaluated.

    Evaluating the function at a point very close to the current one gives little
    information. The solver then performs a safety step instead: it improves the
    spread of its points or shrinks the trust region. Larger values treat more steps
    as too short.

    This is ``general.safety_step_thresh`` of DFO-LS, from which the default is
    taken.

    """

    check_fun_for_overflow: bool = True
    """Whether very large residuals are capped.

    This avoids an ``OverflowError`` when the residuals are squared and summed. There
    is rarely a reason to switch it off.

    This is ``general.check_objfun_for_overflow`` of DFO-LS, from which the default
    is taken.

    """

    init_options: NagInitOptions = field(default_factory=NagInitOptions)
    """Options for the first set of points at which the function is evaluated.

    An instance of :class:`NagInitOptions`.

    """

    trustregion_options: NagTrustregionOptions = field(
        default_factory=NagTrustregionOptions
    )
    """Options that control how the trust-region radius changes.

    An instance of :class:`NagTrustregionOptions`.

    """

    slow_progress_options: NagSlowProgressOptions = field(
        default_factory=NagSlowProgressOptions
    )
    """Options for stopping (or restarting) because of slow progress.

    An instance of :class:`NagSlowProgressOptions`.

    """

    interpolation_options: DfolsInterpolationOptions = field(
        default_factory=DfolsInterpolationOptions
    )
    """Options for fitting the model of the residuals to the evaluated points.

    An instance of :class:`DfolsInterpolationOptions`.

    """

    regression_options: DfolsRegressionOptions = field(
        default_factory=DfolsRegressionOptions
    )
    """Options for using more interpolation points than strictly needed.

    An instance of :class:`DfolsRegressionOptions`.

    """

    restarts: bool | DfolsRestartOptions | None = None
    """Whether the optimization restarts instead of stopping.

    A restart enlarges the trust region again and continues from the best point
    found so far. This helps for noisy functions, where the solver can get stuck
    because of the noise. With restarts, the optimization runs until
    ``stopping_maxfun`` is reached or until several restarts in a row did not improve
    the result.

    - ``None``: DFO-LS decides. It restarts if the function is noisy (see ``noisy``)
      and does not restart otherwise.
    - ``False``: No restarts.
    - ``True``: Restarts with the default settings.
    - An instance of :class:`DfolsRestartOptions`: Restarts with your settings.

    This is ``restarts.use_restarts`` of DFO-LS. The entries of the instance are the
    other ``restarts.*`` options.

    """

    growing_options: DfolsGrowingOptions = field(default_factory=DfolsGrowingOptions)
    """Options for starting with fewer function evaluations than usual.

    An instance of :class:`DfolsGrowingOptions`.

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if not IS_DFOLS_INSTALLED:
            raise NotInstalledError(
                "The 'nag_dfols' algorithm requires the DFO-LS package to be "
                'installed. You can install it with `pip install "DFO-LS>=1.6.5"`.'
            )
        import dfols

        raw_res = dfols.solve(
            problem.fun,
            x0=x0,
            bounds=(problem.bounds.lower, problem.bounds.upper),
            user_params={
                **_collect_upstream_options(self, DFOLS_ADVANCED_OPTIONS),
                **_DIAGNOSTIC_OPTIONS,
            },
            scaling_within_bounds=False,
            do_logging=False,
            print_progress=False,
            **_collect_solve_arguments(self, DFOLS_SOLVE_ARGUMENTS),
        )
        return _process_nag_result(raw_res, x0=x0, solver="DFO-LS")


@mark.deprecated_options(PYBOBYQA_DEPRECATED_OPTIONS)
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
    """Minimize a scalar function without derivatives using Py-BOBYQA.

    Py-BOBYQA :cite:`Cartis2018` is a Python implementation of Powell's BOBYQA
    algorithm (Bound Optimization BY Quadratic Approximation, :cite:`Powell2009`),
    developed by the Numerical Algorithms Group and researchers at the University of
    Oxford. It is a derivative-free trust-region algorithm for scalar functions with
    optional bounds.

    Py-BOBYQA does not use derivatives and does not approximate them with finite
    differences. It evaluates the function at a set of points around the current
    best parameters, fits a quadratic model to these points and minimizes the model
    within a trust region. Compared to Powell's original implementation it adds
    features for noisy functions, restarts and a heuristic for escaping local optima
    (:cite:`Cartis2018a`).

    **When to use it**

    - The function is noisy, i.e. repeated evaluations at the same parameters give
      different values, for example because it contains a simulation. Derivatives
      are then unreliable. Set ``noisy=True``; see also ``n_evals_per_point``
      and ``restarts``.
    - The function is expensive and derivatives are only available via finite
      differences.
    - The function has several local optima and you want to try to escape them
      (``seek_global_optimum``).

    If your function has least-squares structure, use ``nag_dfols`` instead, which
    typically needs far fewer function evaluations. Compared to other derivative-free
    scalar optimizers such as ``scipy_neldermead`` or ``nlopt_bobyqa``, Py-BOBYQA
    stands out by its options for noisy functions and restarts. It is implemented in
    pure Python, so the time the solver itself needs grows quickly with the number
    of parameters.

    **When the optimization stops**

    The main convergence criterion is ``convergence_min_trustregion_radius``: the
    optimization converged when the solver cannot improve the result by steps that
    are longer than this value. The optimization also ends successfully when the
    function value reaches ``convergence_target_value``, or, for noisy functions,
    when all compared function values are within the noise level (``noisy``). It
    ends without success if progress is very slow for many iterations
    (``slow_progress_options``) or if ``stopping_maxfun`` is reached. If restarts are
    used (``restarts``, the default for noisy functions and with
    ``seek_global_optimum``), the solver restarts instead of stopping.

    **What the result contains**

    In addition to the usual entries, the result contains ``jac``, the gradient at
    the solution. It is an estimate from the quadratic model that Py-BOBYQA fitted
    to the evaluated points, not an exact derivative. It refers to the function that
    is minimized (in ``maximize`` the negative of your function) and to the internal
    parameters, i.e. after scaling and constraints were applied. The Hessian of the
    model is not reported, because it is not a reliable estimate at the end of an
    optimization. ``algorithm_output`` contains the number of runs (``n_runs``, more
    than one if there were restarts) and the number of points at which the function
    was evaluated (``n_evaluated_points``).

    **Options of Py-BOBYQA that are not available**

    - Logging and printing options and ``args``: optimagic handles logging and
      additional arguments of the function.
    - ``scaling_within_bounds``: use the ``scaling`` argument of ``minimize``.
    - ``init.run_in_parallel``: it does not evaluate the function in parallel.
    - ``general.check_objfun_for_overflow``: it only catches an ``OverflowError``
      raised by your function. Use the ``error_handling`` argument of ``minimize``.
    - Convex constraints given as projections (``projections`` and the related
      options): not supported yet. Many convex sets can be expressed with
      optimagic's constraints.

    Cite :cite:`Powell2009` and :cite:`Cartis2018` in addition to optimagic when using
    Py-BOBYQA, and :cite:`Cartis2018a` if you use ``seek_global_optimum``. The options
    are described in more technical detail in the `Py-BOBYQA documentation
    <https://numericalalgorithmsgroup.github.io/pybobyqa/>`_.

    """

    stopping_maxfun: PositiveInt = STOPPING_MAXFUN
    """Maximum number of function evaluations.

    If the number is reached, the optimization stops without success. Repeated
    evaluations at the same parameters (see ``n_evals_per_point``) count separately.

    This is ``maxfun`` of Py-BOBYQA. optimagic uses its common default instead of the
    much smaller default of Py-BOBYQA (100 evaluations per parameter, at most 1000).

    """

    convergence_min_trustregion_radius: NonNegativeFloat = 1e-8
    """Stop when the minimal radius of the trust region reaches this value.

    This is the main convergence criterion. The value is roughly the precision of
    the solution in units of the parameters: the solver cannot improve the result
    anymore with steps that are longer than this. Larger values stop earlier and
    save function evaluations. For noisy functions, a precision far below the noise
    is not reachable and larger values such as 1e-4 are reasonable. If restarts are
    used, reaching the value triggers a restart instead.

    This is ``rhoend`` of Py-BOBYQA, from which the default is taken.

    """

    convergence_target_value: float | None = None
    """Stop when the function value is at or below this value.

    Use it if you know which function value is good enough. The value refers to the
    function that is minimized; in ``maximize`` this is the negative of your
    function.

    This is ``model.abs_tol`` of Py-BOBYQA. With ``None`` Py-BOBYQA uses its default
    of -1e20, which switches the criterion off.

    """

    trustregion_initial_radius: PositiveFloat | None = None
    r"""Initial radius of the trust region, i.e. the length of the first steps.

    It is in units of the parameters. Choose it such that a step of this length
    changes the function noticeably but stays in a region where the function can be
    evaluated. Too small values waste the first iterations, too large values make
    the first model unreliable. With bounds, the distance between lower and upper
    bound must be at least twice the radius.

    This is ``rhobeg`` of Py-BOBYQA. With ``None`` Py-BOBYQA uses its default, which
    is :math:`0.1 \max(\|x_0\|_{\infty}, 1)`.

    """

    n_interpolation_points: PositiveInt | None = None
    """Number of points to which the quadratic model is fitted.

    More points make the model more accurate and less sensitive to noise. Each
    additional point costs one function evaluation at the start and makes every
    iteration slower. The number must be between ``len(x) + 1`` and
    ``(len(x) + 1) * (len(x) + 2) / 2``.

    This is ``npt`` of Py-BOBYQA. With ``None`` Py-BOBYQA uses its default, which is
    ``2 * len(x) + 1``, or the maximum for noisy functions (see ``noisy``).

    """

    noisy: bool | NagNoiseOptions = False
    """Whether the function is noisy and, optionally, how large the noise is.

    A function is noisy if repeated evaluations at the same parameters give
    different values. With ``True``, Py-BOBYQA becomes more careful: it uses more
    interpolation points, restarts instead of stopping and shrinks the trust region
    more slowly. These are the defaults of the options that say so in their
    description; you can still set each of them.

    If you know how large the noise is, pass an instance of
    :class:`NagNoiseOptions` instead of ``True``. This also declares the function as
    noisy. Py-BOBYQA can then stop (or restart) as soon as all function values it
    compares are within the noise level.

    ``True`` and an instance both set ``objfun_has_noise`` of Py-BOBYQA, from which
    the default is taken. The entries of the instance are the ``noise.*`` options.

    """

    n_evals_per_point: PositiveInt | Callable[[float, float, int, int], int] | None = (
        None
    )
    """How often the function is evaluated at each point.

    For noisy functions, the solver can evaluate the function several times at the
    same parameters and average the values. More evaluations reduce the noise and
    multiply the cost. You can pass a number or a function that returns the number
    and is called as ``f(radius, min_radius, n_iterations, n_restarts)`` with the
    current trust-region radius, the current minimal radius, the number of
    iterations of the current run and the number of restarts so far. A function
    allows to average more when the trust region is small, where noise hurts most.

    This is ``nsamples`` of Py-BOBYQA. With ``None`` Py-BOBYQA uses its default, which
    is one evaluation per point.

    """

    seek_global_optimum: bool = False
    """Whether to try to escape local optima.

    With ``True``, Py-BOBYQA restarts from the best point found so far each time a
    run converged, with a larger trust region after every restart that did not
    improve the result. It stops when too many restarts did not improve the result
    (see ``restarts``). This is a heuristic without a guarantee to find the
    global optimum, and it costs many more function evaluations. It requires lower
    and upper bounds.

    This is ``seek_global_minimum`` of Py-BOBYQA, from which the default is taken.

    """

    rounding_error_constant: NonNegativeFloat = 0.1
    """Controls how often the solver re-centers its internal coordinates.

    The solver stores points relative to a base point to limit rounding errors. The
    base point is moved to the current best point when the step is shorter than this
    value times the distance between the two. Larger values re-center more often.
    There is rarely a reason to change it.

    This is ``general.rounding_error_constant`` of Py-BOBYQA, from which the default
    is taken.

    """

    safety_step_threshold: NonNegativeFloat = 0.5
    """Steps shorter than this multiple of the minimal radius are not evaluated.

    Evaluating the function at a point very close to the current one gives little
    information. The solver then performs a safety step instead: it improves the
    spread of its points or shrinks the trust region. Larger values treat more steps
    as too short.

    This is ``general.safety_step_thresh`` of Py-BOBYQA, from which the default is
    taken.

    """

    init_options: NagInitOptions = field(default_factory=NagInitOptions)
    """Options for the first set of points at which the function is evaluated.

    An instance of :class:`NagInitOptions`.

    """

    trustregion_options: NagTrustregionOptions = field(
        default_factory=NagTrustregionOptions
    )
    """Options that control how the trust-region radius changes.

    An instance of :class:`NagTrustregionOptions`.

    """

    slow_progress_options: NagSlowProgressOptions = field(
        default_factory=NagSlowProgressOptions
    )
    """Options for stopping (or restarting) because of slow progress.

    An instance of :class:`NagSlowProgressOptions`.

    """

    interpolation_options: PybobyqaInterpolationOptions = field(
        default_factory=PybobyqaInterpolationOptions
    )
    """Options for fitting the quadratic model to the evaluated points.

    An instance of :class:`PybobyqaInterpolationOptions`.

    """

    restarts: bool | PybobyqaRestartOptions | None = None
    """Whether the optimization restarts instead of stopping.

    A restart enlarges the trust region again and continues from the best point
    found so far. This helps for noisy functions, where the solver can get stuck
    because of the noise, and for escaping local optima. With restarts, the
    optimization runs until ``stopping_maxfun`` is reached or until several restarts
    in a row did not improve the result.

    - ``None``: Py-BOBYQA decides. It restarts if the function is noisy (see
      ``noisy``) or if ``seek_global_optimum`` is ``True`` and does not restart
      otherwise.
    - ``False``: No restarts.
    - ``True``: Restarts with the default settings.
    - An instance of :class:`PybobyqaRestartOptions`: Restarts with your settings.

    This is ``restarts.use_restarts`` of Py-BOBYQA. The entries of the instance are
    the other ``restarts.*`` options.

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if not IS_PYBOBYQA_INSTALLED:
            raise NotInstalledError(
                "The 'nag_pybobyqa' algorithm requires the Py-BOBYQA package to be "
                'installed. You can install it with `pip install "Py-BOBYQA>=1.5.0"`.'
            )
        import pybobyqa

        raw_res = pybobyqa.solve(
            problem.fun,
            x0=x0,
            bounds=(problem.bounds.lower, problem.bounds.upper),
            user_params={
                **_collect_upstream_options(self, PYBOBYQA_ADVANCED_OPTIONS),
                **_DIAGNOSTIC_OPTIONS,
            },
            scaling_within_bounds=False,
            do_logging=False,
            print_progress=False,
            **_collect_solve_arguments(self, PYBOBYQA_SOLVE_ARGUMENTS),
        )
        return _process_nag_result(raw_res, x0=x0, solver="Py-BOBYQA")


# ======================================================================================
# Helper functions
# ======================================================================================


def _collect_upstream_options(
    algo: Algorithm, option_map: Mapping[str, str]
) -> dict[str, Any]:
    """Collect the options in option_map under their upstream names.

    Options that are None are left out, such that the solver uses its own default.
    This matters beyond the value: DFO-LS only chooses some defaults itself if the
    option is not passed at all.

    ``restarts`` and ``noisy`` are a boolean or an option group. The option itself
    is passed as flag, which is True for a group. The entries are only passed for a
    group.

    """
    out = {}
    for path, upstream_name in option_map.items():
        option, _, entry = path.partition(".")
        value: Any = getattr(algo, option)
        if entry:
            if value is None or isinstance(value, bool):
                continue
            value = getattr(value, entry)
        elif is_dataclass(value):
            value = True
        if value is not None:
            out[upstream_name] = value
    return out


def _collect_solve_arguments(
    algo: Algorithm, option_map: Mapping[str, str]
) -> dict[str, Any]:
    """Collect the options that are arguments of the upstream solve function."""
    out = _collect_upstream_options(algo, option_map)
    n_evals = out.get("nsamples")
    if n_evals is not None and not callable(n_evals):
        # The solvers expect a function of the trust-region radius, the minimal
        # radius, the number of iterations and the number of restarts.
        out["nsamples"] = lambda delta, rho, iter, nrestarts: n_evals  # noqa: A006
    return out


def _process_nag_result(
    raw_res: Any, x0: NDArray[np.float64], solver: str
) -> InternalOptimizeResult:
    """Convert the result object of DFO-LS or Py-BOBYQA to optimagic's result."""
    if raw_res.flag == raw_res.EXIT_INPUT_ERROR:
        msg = (
            f"{solver} rejected the algorithm options or their combination with the "
            f"bounds and start parameters: {raw_res.msg}"
        )
        raise InvalidAlgoOptionError(msg)

    x = raw_res.x if raw_res.x is not None else np.full(len(x0), np.nan)
    # DFO-LS reports the sum of squares as `obj`, Py-BOBYQA the function value as `f`.
    fun = raw_res.f if hasattr(raw_res, "f") else raw_res.obj

    return InternalOptimizeResult(
        x=x,
        fun=fun,
        success=raw_res.flag == raw_res.EXIT_SUCCESS,
        message=raw_res.msg,
        status=int(raw_res.flag),
        n_fun_evals=int(raw_res.nf),
        n_iterations=_count_iterations(raw_res),
        # An estimate from the last model of the solver: the Jacobian of the residuals
        # for DFO-LS and the gradient for Py-BOBYQA. It is None if the solver stopped
        # before it built a model. The Hessian of the Py-BOBYQA model is not reported:
        # at the end the points are too close to each other to estimate curvature.
        jac=raw_res.jacobian if hasattr(raw_res, "jacobian") else raw_res.gradient,
        info={"n_runs": int(raw_res.nruns), "n_evaluated_points": int(raw_res.nx)},
    )


def _count_iterations(raw_res: Any) -> int | None:
    """Count the iterations of all runs of DFO-LS or Py-BOBYQA.

    The solvers do not report the number of iterations. It is the number of rows of
    their diagnostic information, which has one row per iteration over all runs. The
    diagnostic information is missing if the solver stops before the first
    iteration, e.g. because it rejects the options.

    """
    if raw_res.diagnostic_info is None:
        return None
    return len(raw_res.diagnostic_info)
