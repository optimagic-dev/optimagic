"""Implement the fides optimizer."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, Callable, Literal, TypeAlias, cast

import numpy as np
from annotated_types import Ge, Gt, Lt
from numpy.typing import NDArray

from optimagic import mark
from optimagic.config import IS_FIDES_INSTALLED
from optimagic.exceptions import NotInstalledError
from optimagic.optimization.algo_option_tools import (
    DeprecatedOption,
    resolve_changed_default,
)
from optimagic.optimization.algo_options import (
    CONVERGENCE_FTOL_ABS,
    CONVERGENCE_FTOL_REL,
    CONVERGENCE_GTOL_ABS,
    CONVERGENCE_GTOL_REL,
    CONVERGENCE_XTOL_ABS,
    STOPPING_MAXITER,
)
from optimagic.optimization.algorithm import Algorithm, InternalOptimizeResult
from optimagic.optimization.internal_optimization_problem import (
    InternalOptimizationProblem,
)
from optimagic.typing import (
    AggregationLevel,
    GtOneFloat,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    UnitIntervalFloat,
)

if TYPE_CHECKING:
    from fides.hessian_approximation import HessianApproximation
else:
    # HessianApproximation is used in a field annotation, which pydantic resolves
    # at runtime, so it needs a fallback that works without fides and avoids
    # importing it at optimagic import time.
    HessianApproximation = Any

HessianUpdate: TypeAlias = (
    Literal["bfgs", "sr1", "dfp", "bg", "bb"] | str | HessianApproximation
)
"""Hessian approximations of fides: a case-insensitive name or a HessianApproximation
instance from fides."""

SubspaceSolver: TypeAlias = Literal["2D", "full", "scg"]
"""Solvers for the trust-region subproblem of fides."""

StepbackStrategy: TypeAlias = Literal[
    "reflect", "reflect_single", "truncate", "mixed", "refine"
]
"""Strategies of fides for steps that would cross a bound."""

FIDES_OPTION_NAMES = {
    "maxiter": "stopping_maxiter",
    "maxtime": "stopping_maxtime",
    "fatol": "convergence_ftol_abs",
    "frtol": "convergence_ftol_rel",
    "xtol": "convergence_xtol_abs",
    "gatol": "convergence_gtol_abs",
    "grtol": "convergence_gtol_rel",
    "subspace_solver": "subspace_solver",
    "stepback_strategy": "stepback_strategy",
    "theta_max": "max_stepback_fraction",
    "delta_init": "trustregion_initial_radius",
    "mu": "trustregion_shrinking_threshold",
    "eta": "trustregion_expansion_threshold",
    "gamma1": "trustregion_shrinking_factor",
    "gamma2": "trustregion_expansion_factor",
}
"""The option names of ``fides.Optimizer`` and the optimagic options they are set by."""

FIDES_OPTIONS_NOT_EXPOSED = {
    "history_file": "optimagic collects the history and can log it to a database.",
}
"""Options of ``fides.Optimizer`` that cannot be set, with the reason."""

FIDES_DEVIATING_DEFAULTS = {
    "stopping_maxiter": "harmonized across optimagic",
    "convergence_ftol_abs": "harmonized across optimagic",
    "convergence_ftol_rel": "harmonized across optimagic",
    "convergence_gtol_abs": "harmonized across optimagic",
    "convergence_gtol_rel": "harmonized across optimagic",
    "subspace_solver": "'full' has less overhead than '2D' at similar step quality",
    "stepback_strategy": "changes to the fides default in optimagic 0.7.0",
}
"""Options whose default differs from the default of fides, with the reason."""

FIDES_DEPRECATED_OPTIONS = (
    DeprecatedOption("hessian_update_strategy", "hessian_update", "0.7.0"),
    DeprecatedOption("stopping_max_seconds", "stopping_maxtime", "0.7.0"),
    DeprecatedOption("trustregion_subspace_dimension", "subspace_solver", "0.7.0"),
    DeprecatedOption("trustregion_stepback_strategy", "stepback_strategy", "0.7.0"),
    DeprecatedOption(
        "trustregion_max_stepback_fraction", "max_stepback_fraction", "0.7.0"
    ),
    DeprecatedOption(
        "trustregion_decrease_threshold", "trustregion_shrinking_threshold", "0.7.0"
    ),
    DeprecatedOption(
        "trustregion_increase_threshold", "trustregion_expansion_threshold", "0.7.0"
    ),
    DeprecatedOption(
        "trustregion_decrease_factor", "trustregion_shrinking_factor", "0.7.0"
    ),
    DeprecatedOption(
        "trustregion_increase_factor", "trustregion_expansion_factor", "0.7.0"
    ),
)
"""Option names of the fides wrapper that are deprecated."""


@mark.deprecated_options(FIDES_DEPRECATED_OPTIONS)
@mark.minimizer(
    name="fides",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_FIDES_INSTALLED,
    is_global=False,
    needs_jac=True,
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
class Fides(Algorithm):
    """Minimize a scalar function using the Fides trust-region optimizer.

    Fides is a gradient-based local optimizer for smooth scalar functions with or
    without bounds. It is a good choice if evaluating the function and its gradient
    is expensive, because it spends more effort than line-search methods such as
    L-BFGS-B on choosing each step and typically needs fewer evaluations in return.
    The price is a matrix factorization per iteration, so it is less suited for
    problems with thousands of parameters.

    In each iteration, fides builds a quadratic model of the function from the
    gradient and an approximation of the Hessian and minimizes it within a trust
    region, i.e. the region around the current parameters in which the model is
    trusted. The radius of the region grows when the model predicts the function
    well and shrinks when it does not. Bounds are handled with the interior
    trust-region reflective approach of :cite:`Coleman1994` and :cite:`Coleman1996`:
    The parameters stay strictly inside the bounds and steps that would cross a
    bound are reflected or truncated. Fides is named after the Roman goddess of
    trust and reliability.

    Fides is highly configurable. You can choose how the Hessian is approximated
    (``hessian_update``), how the model is minimized (``subspace_solver``), what
    happens at bounds (``stepback_strategy``) and how the trust-region radius is
    managed. The :ref:`fides tutorial <fides_tutorial>` shows when and how to use
    these options.

    The optimizer is taken from the ``fides`` package (:cite:`Froehlich2022`), which
    must be installed separately (``pip install "fides>=0.8.2"``).

    The following features of fides are not supported yet:

    - Exact Hessians and the hybrid Hessian approximations that need them
      (``HybridFixed``, ``HybridFraction``), because optimagic does not pass
      user-provided Hessians to optimizers.
    - Least-squares problems (``resfun=True``) and the Hessian approximations that
      need residuals (``FX``, ``SSM``, ``TSSM``, ``GNSBFGS``).
    - A custom initial Hessian approximation (``hess0``). The approximation always
      starts with the identity matrix.
    - The history file of fides (``history_file``). Use optimagic's history and
      logging instead.

    """

    hessian_update: HessianUpdate = "bfgs"
    r"""How the Hessian of the function is approximated from gradients.

    The approximation determines the quality of the quadratic model and therefore
    of the steps. You can pass a (case-insensitive) name:

    - ``"bfgs"``: The standard choice. The approximation is always positive
      definite, which is reliable for most problems but cannot represent negative
      curvature.
    - ``"sr1"``: Can represent negative curvature. Try it if the function is far
      from convex, e.g. if ``"bfgs"`` makes slow progress around saddle points.
    - ``"dfp"``, ``"bg"`` and ``"bb"``: Older update formulas that mostly serve as
      comparison. They typically converge much more slowly than ``"bfgs"`` and
      ``"sr1"``.

    For more control, pass an instance of a class from
    ``fides.hessian_approximation``, for example ``BFGS(enforce_curv_cond=False)``
    to also update the approximation after steps that violate the curvature
    condition, or ``Broyden(phi=0.5)`` for a mix of BFGS and DFP. The classes that
    need exact Hessians or residuals are not supported.

    This is the ``hessian_update`` argument of ``fides.Optimizer``. Without it, fides
    expects exact Hessians, which optimagic cannot provide, so optimagic defaults to
    ``"bfgs"``.

    """

    convergence_ftol_abs: NonNegativeFloat = CONVERGENCE_FTOL_ABS
    r"""Stop when the function value changes by less than this between iterations.

    This criterion measures progress, not closeness to an optimum. It is a safety
    net for problems where the gradient criteria cannot be met, e.g. because the
    gradient is noisy. Increase it to stop earlier on such problems. Together with
    ``convergence_ftol_rel``, fides stops when

    .. math::

        |f_k - f_{k-1}| < \text{ftol\_abs} + \text{ftol\_rel} \cdot |f_{k-1}|

    after a successful step that was not limited by the trust region.

    This is the ``fatol`` option of fides. The default of 0 is harmonized across
    optimagic and means that only the relative tolerance is used; fides defaults to
    1e-8.

    """

    convergence_ftol_rel: NonNegativeFloat = CONVERGENCE_FTOL_REL
    r"""Stop when the relative change of the function value between iterations is
    smaller than this.

    This criterion measures progress, not closeness to an optimum; see
    ``convergence_ftol_abs`` for the formula. Increase it to stop earlier if function
    evaluations are expensive and a rough solution is enough. Set it to 0 to rely on
    the gradient criteria alone.

    This is the ``frtol`` option of fides. The default of 2e-9 is harmonized across
    optimagic and stricter than the fides default of 1e-8, so that most runs are
    ended by a gradient criterion.

    """

    convergence_xtol_abs: NonNegativeFloat = CONVERGENCE_XTOL_ABS
    r"""Stop when a successful step is shorter than this.

    The length is the Euclidean norm of the step, :math:`\|x_k - x_{k-1}\|`, measured
    in coordinates that are rescaled close to bounds. Like the function tolerances,
    this criterion measures progress and not closeness to an optimum.

    This is the ``xtol`` option of fides. The default of 0 switches the criterion
    off, as in fides.

    """

    convergence_gtol_abs: NonNegativeFloat = CONVERGENCE_GTOL_ABS
    r"""Stop when the norm of the gradient is at most this value.

    A small gradient means that the parameters are close to a local optimum, so this
    is the main convergence criterion. Decrease it for a more precise solution;
    increase it if the gradient is too noisy to become this small. For parameters at
    a bound, only the part of the gradient that points into the feasible region
    counts.

    This is the ``gatol`` option of fides. The default of 1e-5 is harmonized across
    optimagic; fides defaults to 1e-6.

    """

    convergence_gtol_rel: NonNegativeFloat = CONVERGENCE_GTOL_REL
    r"""Stop when the norm of the gradient is at most this value times the absolute
    function value.

    This makes the gradient criterion independent of the scale of the function,
    i.e. fides stops when :math:`\|g_k\| \leq \text{gtol\_rel} \cdot |f_k|`. It has
    no effect if the function value at the optimum is zero.

    This is the ``grtol`` option of fides. The default of 1e-8 is harmonized across
    optimagic; fides defaults to 0, which switches the criterion off.

    """

    stopping_maxiter: PositiveInt = STOPPING_MAXITER
    """Maximum number of iterations.

    Each iteration evaluates the function and its gradient once. If the limit is
    reached, the optimization stops without success.

    This is the ``maxiter`` option of fides. The default of 1,000,000 is harmonized
    across optimagic; fides defaults to 1,000.

    """

    stopping_maxtime: PositiveFloat = np.inf
    """Maximum runtime in seconds.

    Fides stops before an iteration if it expects that the iteration would exceed
    the limit, based on the average duration of the previous iterations.

    This is the ``maxtime`` option of fides. As in fides, there is no limit by
    default.

    """

    trustregion_initial_radius: PositiveFloat = 1.0
    """Radius of the trust region in the first iteration.

    The radius limits the length of the first step, measured in the units of the
    parameters. Decrease it if the first steps lead into regions where the function
    is not defined or behaves very differently than at the start parameters.
    Increase it if the start parameters are far from the optimum and the first
    steps are too cautious. After the first iterations the radius adapts itself.

    This is the ``delta_init`` option of fides and has the same default.

    """

    stepback_strategy: StepbackStrategy | None = None
    """What happens with a step that would cross a bound.

    The option only matters for problems with bounds, and mostly if several
    parameters are at a bound at the optimum.

    - ``"reflect"``: The step is reflected at each bound it reaches, like a ball
      bouncing off a wall. This is the default of fides.
    - ``"reflect_single"``: Like ``"reflect"``, but with at most one reflection.
    - ``"truncate"``: The step is cut off at the bound. Then the parameters at the
      bound are fixed and the step is recomputed for the remaining ones.
    - ``"mixed"``: Fides computes reflected and truncated steps and takes the one
      that promises the larger improvement.
    - ``"refine"``: The reflected step is improved by an additional optimization of
      the model inside the bounds. This can save iterations if many bounds are
      active.

    This is the ``stepback_strategy`` option of fides. The default None currently
    means ``"truncate"`` and emits a warning if the problem has bounds; from
    optimagic 0.7.0 on the default is ``"reflect"``, the default of fides.

    """

    subspace_solver: SubspaceSolver = "full"
    """How the quadratic model is minimized inside the trust region.

    - ``"full"``: Minimize the model exactly over all parameters. This gives the
      best step the model can offer.
    - ``"2D"``: Only search the plane spanned by the gradient and the Newton
      direction. The steps are usually almost as good as the exact ones. This is
      the default of fides.
    - ``"scg"``: Use a conjugate gradient method that stops early. With hundreds of
      parameters, an iteration is much cheaper but more iterations are needed. Use
      it if the function is cheap to evaluate and most of the runtime is spent
      inside the optimizer.

    This is the ``subspace_solver`` option of fides. optimagic defaults to
    ``"full"`` instead of ``"2D"`` because in benchmarks with fides 0.8.2 it needed
    about the same number of function evaluations and less computation time per
    iteration.

    """

    max_stepback_fraction: UnitIntervalFloat = 0.95
    """How close a single step may get to a bound.

    If a step would hit a bound, fides takes at most this fraction of the way to
    the bound, so the parameters always stay strictly inside the bounds. Values
    closer to 1 let the parameters approach a bound in fewer iterations. Smaller
    values keep more distance to the bounds in every step. The fraction increases
    automatically towards 1 as the optimizer converges.

    This is the ``theta_max`` option of fides and has the same default.

    """

    trustregion_shrinking_threshold: Annotated[float, Ge(0), Lt(1)] = 0.25
    """The trust region shrinks after steps whose quality is at or below this value.

    The quality of a step is the actual improvement of the function divided by the
    improvement the model predicted. Larger values shrink the region more often,
    which is more cautious but can slow down progress. The threshold does not decide
    whether a step is accepted: Every step that improves the function is accepted.

    This is the ``mu`` option of fides and has the same default.

    """

    trustregion_expansion_threshold: Annotated[float, Ge(0), Lt(1)] = 0.75
    """The trust region can grow after steps whose quality is at or above this value.

    The quality of a step is the actual improvement of the function divided by the
    improvement the model predicted. Smaller values let the region grow more easily,
    which leads to longer steps and is useful if the function is well approximated
    by a quadratic model. The region only grows after steps that used at least 90
    percent of the current radius.

    This is the ``eta`` option of fides and has the same default.

    """

    trustregion_shrinking_factor: Annotated[float, Gt(0), Lt(1)] = 0.25
    """Factor by which the radius of the trust region is multiplied when it shrinks.

    Smaller values shrink the region faster. This helps if the optimizer needs
    several unsuccessful steps in a row before it makes progress again, e.g. on
    functions that are not defined everywhere. The new radius is never larger than
    a quarter of the length of the unsuccessful step, so values above 0.25 rarely
    make a difference.

    This is the ``gamma1`` option of fides and has the same default.

    """

    trustregion_expansion_factor: GtOneFloat = 2.0
    """Factor by which the radius of the trust region is multiplied when it grows.

    Larger values lead to longer steps more quickly, which saves iterations if the
    start parameters are far from the optimum, but makes unsuccessful steps more
    likely.

    This is the ``gamma2`` option of fides and has the same default.

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if not IS_FIDES_INSTALLED:
            raise NotInstalledError(
                "The 'fides' algorithm requires the fides package to be installed. "
                'You can install it with `pip install "fides>=0.8.2"`.'
            )

        from fides import Optimizer

        options = {
            fides_name: getattr(self, optimagic_name)
            for fides_name, optimagic_name in FIDES_OPTION_NAMES.items()
        }
        n_params = len(x0)
        lower = problem.bounds.lower
        upper = problem.bounds.upper
        lower = np.full(n_params, -np.inf) if lower is None else lower
        upper = np.full(n_params, np.inf) if upper is None else upper

        if np.isfinite(lower).any() or np.isfinite(upper).any():
            options["stepback_strategy"] = resolve_changed_default(
                self.stepback_strategy,
                option="stepback_strategy",
                old_default="truncate",
                new_default="reflect",
                algo_name="fides",
                changed_in="0.7.0",
            )
        elif self.stepback_strategy is None:
            # Without bounds the strategy is never used, so there is nothing to
            # warn about.
            options["stepback_strategy"] = "reflect"

        opt = Optimizer(
            fun=cast(Callable[..., Any], problem.fun_and_jac),
            lb=lower,
            ub=upper,
            verbose=logging.ERROR,
            options=options,
            funargs=None,
            hessian_update=_create_hessian_update(self.hessian_update),
            resfun=False,
        )
        fval, x, grad, hess = opt.minimize(x0)

        return InternalOptimizeResult(
            x=x,
            fun=fval,
            jac=grad,
            hess=hess,
            success=opt.converged,
            message=FIDES_EXIT_MESSAGES.get(opt.exitflag.name),
            n_iterations=opt.iteration,
        )


FIDES_EXIT_MESSAGES = {
    "DID_NOT_RUN": "The optimizer did not run.",
    "MAXITER": "Reached the maximum number of iterations.",
    "MAXTIME": "The next iteration was expected to exceed the maximum runtime.",
    "NOT_FINITE": "Encountered a non-finite function value or gradient.",
    "EXCEEDED_BOUNDARY": "The parameters left the bounds.",
    "DELTA_TOO_SMALL": "The trust-region radius became too small to proceed.",
    "FTOL": "Converged: the change of the function value is below the tolerance.",
    "XTOL": "Converged: the step length is below the tolerance.",
    "GTOL": "Converged: the gradient norm is below the tolerance.",
}
"""The exit flags of fides and the messages optimagic reports for them."""

_HESSIAN_UPDATE_NAMES = ("bfgs", "sr1", "dfp", "bg", "bb")


def _create_hessian_update(hessian_update: HessianUpdate) -> HessianApproximation:
    """Create the Hessian approximation of fides from the user input."""
    from fides import hessian_approximation

    if isinstance(hessian_update, str):
        name = hessian_update.lower()
        if name == "broyden":
            raise ValueError(
                "The Broyden update cannot be selected by name because its parameter "
                "phi has no default. Pass an instance instead, e.g. "
                "`fides.hessian_approximation.Broyden(phi=0.5)`."
            )
        if name in ("fx", "ssm", "tssm", "gnsbfgs", "hybridfixed", "hybridfraction"):
            raise NotImplementedError(_unsupported_hessian_update_message(name))
        if name not in _HESSIAN_UPDATE_NAMES:
            raise ValueError(
                f"Invalid hessian_update: {hessian_update!r}. Choose one of "
                f"{_HESSIAN_UPDATE_NAMES} or pass an instance of a class from "
                "fides.hessian_approximation."
            )
        return getattr(hessian_approximation, name.upper())()

    if not isinstance(hessian_update, hessian_approximation.HessianApproximation):
        raise TypeError(
            "hessian_update must be a string or an instance of a class from "
            f"fides.hessian_approximation, not {type(hessian_update)}."
        )

    if hessian_update.requires_resfun or hessian_update.requires_hess:
        name = type(hessian_update).__name__
        raise NotImplementedError(_unsupported_hessian_update_message(name))

    return hessian_update


def _unsupported_hessian_update_message(name: str) -> str:
    return (
        f"The Hessian approximation {name} is not supported yet because it needs "
        "exact Hessians or least-squares residuals, which optimagic does not pass "
        f"to fides. Choose one of {_HESSIAN_UPDATE_NAMES} or pass an instance of "
        "BFGS, SR1, DFP, BG, BB or Broyden from fides.hessian_approximation."
    )
