"""Implement the fides optimizer."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Literal, cast

import numpy as np
from numpy.typing import NDArray

from optimagic import mark
from optimagic.config import IS_FIDES_INSTALLED
from optimagic.exceptions import NotInstalledError
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
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
)


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

    Fides implements an Interior Trust-Region Reflective method for bound-constrained
    optimization, following the approach of :cite:`Coleman1994` and
    :cite:`Coleman1996`. Accordingly, Fides is named after the Roman goddess of trust
    and reliability. The optimizer is taken from the ``fides`` package
    (:cite:`Froehlich2022`), which must be installed separately
    (``pip install fides>=0.7.4``).

    Fides is a gradient-based local optimizer for smooth, differentiable scalar
    objective functions. It supports lower and upper bounds (including infinite
    bounds) but no linear or nonlinear constraints. It requires first derivatives; the
    Hessian is approximated internally using one of several quasi-Newton update
    strategies, so no user-provided Hessian is needed. It is well suited for problems
    with up to several hundred parameters.

    In contrast to many other trust-region methods, Fides solves the full trust-region
    subproblem exactly rather than approximately. This can yield higher-quality
    proposal steps at a higher per-iteration cost, which makes Fides especially
    attractive when the objective function is expensive to evaluate and the cost of
    solving the trust-region subproblem is negligible in comparison.

    .. note::
        General linear and nonlinear constraints are not supported by optimagic.

    """

    hessian_update_strategy: Literal[
        "bfgs",
        "bb",
        "bg",
        "dfp",
        "sr1",
    ] = "bfgs"
    """Quasi-Newton strategy used to approximate the Hessian of the objective.

    The available strategies are:

    - ``"bfgs"``: the Broyden-Fletcher-Goldfarb-Shanno update, a rank-2 update that
      preserves symmetry and positive definiteness. This is the default.
    - ``"dfp"``: the Davidon-Fletcher-Powell update.
    - ``"sr1"``: the Symmetric Rank 1 update, described in :cite:`Nocedal1999`,
      Chapter 6.2.
    - ``"bb"``: Broyden's "bad" method, introduced in :cite:`Broyden1965`.
    - ``"bg"``: Broyden's "good" method, introduced in :cite:`Broyden1965`.

    The general Broyden class update, a convex combination of the BFGS and DFP updates
    controlled by a parameter :math:`\\phi` (:cite:`Nocedal1999`, Chapter 6.3), and the
    residual-based approximations ``FX``, ``SSM``, ``TSSM`` and ``GNSBFGS`` provided by
    the ``fides`` package are not available through this option, because they require
    access to least-squares residuals or an interpolation parameter that optimagic does
    not pass through.

    """

    convergence_ftol_abs: NonNegativeFloat = CONVERGENCE_FTOL_ABS
    r"""Stop when the absolute change in the objective becomes small.

    Denoting the absolute criterion tolerance by :math:`\alpha` (this parameter) and
    the relative criterion tolerance by :math:`\beta`
    (``convergence_ftol_rel``), Fides stops successfully when

    .. math::

        |f_k - f_{k-1}| < \alpha + \beta \, |f_{k-1}|.

    This parameter therefore governs convergence on its own only when
    ``convergence_ftol_rel`` is set to 0.

    """

    convergence_ftol_rel: NonNegativeFloat = CONVERGENCE_FTOL_REL
    r"""Stop when the relative change in the objective becomes small.

    This is the relative criterion tolerance :math:`\beta` in the joint convergence
    condition documented under ``convergence_ftol_abs``. It governs convergence on its
    own only when ``convergence_ftol_abs`` is set to 0 (as is the default).

    """

    convergence_xtol_abs: NonNegativeFloat = CONVERGENCE_XTOL_ABS
    r"""Stop when the step size becomes small.

    The optimization terminates successfully when the norm of the step
    :math:`\|x_{k} - x_{k-1}\|` falls below this tolerance.

    """

    convergence_gtol_abs: NonNegativeFloat = CONVERGENCE_GTOL_ABS
    """Stop when the gradient norm is less than or equal to this tolerance."""

    convergence_gtol_rel: NonNegativeFloat = CONVERGENCE_GTOL_REL
    """Stop when the gradient norm divided by the absolute function value is less than
    or equal to this tolerance."""

    stopping_maxiter: PositiveInt = STOPPING_MAXITER
    """Maximum number of iterations."""

    stopping_max_seconds: float = np.inf
    """Maximum number of wall-clock seconds. Deactivated (infinite) by default."""

    trustregion_initial_radius: PositiveFloat = 1.0
    """Initial trust-region radius."""

    trustregion_stepback_strategy: Literal[
        "truncate",
        "reflect",
        "reflect_single",
        "mixed",
    ] = "truncate"
    """Refinement strategy applied when a proposed step reaches a parameter bound.

    The available options are:

    - ``"truncate"``: truncate the step at the boundary and re-solve the restricted
      subproblem. This is optimagic's default.
    - ``"reflect"``: recursive reflections at the boundary.
    - ``"reflect_single"``: a single reflection at the boundary.
    - ``"mixed"``: mix reflections and truncations.

    Note that optimagic defaults to ``"truncate"``, whereas the ``fides`` package
    itself defaults to ``"reflect"``.

    """

    trustregion_subspace_dimension: Literal[
        "full",
        "2D",
        "scg",
    ] = "full"
    """Dimension of the subspace in which the trust-region subproblem is solved.

    The available options are:

    - ``"full"``: use the full parameter dimensionality. This is optimagic's default.
    - ``"2D"``: a two-dimensional Newton/gradient subspace.
    - ``"scg"``: a conjugate-gradient subspace via Steihaug's method.

    Note that optimagic defaults to ``"full"``, whereas the ``fides`` package itself
    defaults to ``"2D"``.

    """

    trustregion_max_stepback_fraction: float = 0.95
    """Controls how close steps are allowed to get to the boundary.

    It is the maximal fraction of a step to take if the full step would reach a
    breakpoint (the bound).

    """

    trustregion_decrease_threshold: float = 0.25
    r"""Acceptance threshold for the trust-region ratio.

    The trust-region radius is decreased when the trust-region ratio falls below this
    value. It is denoted by :math:`\mu` in algorithm 4.1 of :cite:`Nocedal2006`.

    """

    trustregion_increase_threshold: float = 0.75
    r"""Threshold for the trust-region ratio above which the radius may be increased.

    It is denoted by :math:`\eta` in algorithm 4.1 of :cite:`Nocedal2006`.

    """

    trustregion_decrease_factor: float = 0.25
    r"""Factor by which the trust-region radius is multiplied when it is decreased.

    It is denoted by :math:`\gamma_1` in algorithm 4.1 of :cite:`Nocedal2006`.

    """

    trustregion_increase_factor: float = 2.0
    r"""Factor by which the trust-region radius is multiplied when it is increased.

    It is denoted by :math:`\gamma_2` in algorithm 4.1 of :cite:`Nocedal2006`.

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        res = fides_internal(
            fun_and_jac=cast(
                Callable[[NDArray[np.float64]], NDArray[np.float64]],
                problem.fun_and_jac,
            ),
            x=x0,
            lower_bounds=problem.bounds.lower,
            upper_bounds=problem.bounds.upper,
            hessian_update_strategy=self.hessian_update_strategy,
            convergence_ftol_abs=self.convergence_ftol_abs,
            convergence_ftol_rel=self.convergence_ftol_rel,
            convergence_xtol_abs=self.convergence_xtol_abs,
            convergence_gtol_abs=self.convergence_gtol_abs,
            convergence_gtol_rel=self.convergence_gtol_rel,
            stopping_maxiter=self.stopping_maxiter,
            stopping_max_seconds=self.stopping_max_seconds,
            trustregion_initial_radius=self.trustregion_initial_radius,
            trustregion_stepback_strategy=self.trustregion_stepback_strategy,
            trustregion_subspace_dimension=self.trustregion_subspace_dimension,
            trustregion_max_stepback_fraction=self.trustregion_max_stepback_fraction,
            trustregion_decrease_threshold=self.trustregion_decrease_threshold,
            trustregion_increase_threshold=self.trustregion_increase_threshold,
            trustregion_decrease_factor=self.trustregion_decrease_factor,
            trustregion_increase_factor=self.trustregion_increase_factor,
        )

        return res


def fides_internal(
    fun_and_jac: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    x: NDArray[np.float64],
    lower_bounds: NDArray[np.float64] | None,
    upper_bounds: NDArray[np.float64] | None,
    hessian_update_strategy: Literal[
        "bfgs",
        "bb",
        "bg",
        "dfp",
        "sr1",
    ],
    convergence_ftol_abs: NonNegativeFloat,
    convergence_ftol_rel: NonNegativeFloat,
    convergence_xtol_abs: NonNegativeFloat,
    convergence_gtol_abs: NonNegativeFloat,
    convergence_gtol_rel: NonNegativeFloat,
    stopping_maxiter: PositiveInt,
    stopping_max_seconds: float,
    trustregion_initial_radius: PositiveFloat,
    trustregion_stepback_strategy: Literal[
        "truncate",
        "reflect",
        "reflect_single",
        "mixed",
    ],
    trustregion_subspace_dimension: Literal[
        "full",
        "2D",
        "scg",
    ],
    trustregion_max_stepback_fraction: float,
    trustregion_decrease_threshold: float,
    trustregion_increase_threshold: float,
    trustregion_decrease_factor: float,
    trustregion_increase_factor: float,
) -> InternalOptimizeResult:
    """Minimize a scalar function using the Fides Optimizer.

    For details see
    :ref: `fides_algorithm`.

    """
    if not IS_FIDES_INSTALLED:
        raise NotInstalledError(
            "The 'fides' algorithm requires the fides package to be installed. "
            "You can install it with `pip install fides>=0.7.4`."
        )

    from fides import Optimizer

    fides_options = {
        "delta_init": trustregion_initial_radius,
        "eta": trustregion_increase_threshold,
        "fatol": convergence_ftol_abs,
        "frtol": convergence_ftol_rel,
        "gamma1": trustregion_decrease_factor,
        "gamma2": trustregion_increase_factor,
        "gatol": convergence_gtol_abs,
        "grtol": convergence_gtol_rel,
        "maxiter": stopping_maxiter,
        "maxtime": stopping_max_seconds,
        "mu": trustregion_decrease_threshold,
        "stepback_strategy": trustregion_stepback_strategy,
        "subspace_solver": trustregion_subspace_dimension,
        "theta_max": trustregion_max_stepback_fraction,
        "xtol": convergence_xtol_abs,
    }

    hessian_instance = _create_hessian_updater_from_user_input(hessian_update_strategy)

    lower_bounds = np.full(len(x), -np.inf) if lower_bounds is None else lower_bounds
    upper_bounds = np.full(len(x), np.inf) if upper_bounds is None else upper_bounds

    opt = Optimizer(
        fun=fun_and_jac,
        lb=lower_bounds,
        ub=upper_bounds,
        verbose=logging.ERROR,
        options=fides_options,
        funargs=None,
        hessian_update=hessian_instance,
        resfun=False,
    )
    raw_res = opt.minimize(x)
    res = _process_fides_res(raw_res, opt)
    out = InternalOptimizeResult(
        x=res["solution_x"],
        fun=res["solution_criterion"],
        jac=res["solution_derivative"],
        hess=res["solution_hessian"],
        success=res["success"],
        message=res["message"],
        n_iterations=res["n_iterations"],
    )

    return out


def _process_fides_res(raw_res, opt):
    """Create an optimagic results dictionary from the Fides output.

    Args:
        raw_res (tuple): Tuple containing the Fides result
        opt (fides.Optimizer): Fides Optimizer after minimize has been called on it.

    """
    fval, x, grad, hess = raw_res
    res = {
        "solution_criterion": fval,
        "solution_x": x,
        "solution_derivative": grad,
        "solution_hessian": hess,
        "success": opt.converged,
        "n_iterations": opt.iteration,
        "message": _process_exitflag(opt.exitflag),
    }
    return res


def _process_exitflag(exitflag):
    messages = {
        "DID_NOT_RUN": "The optimizer did not run",
        "MAXITER": "Reached maximum number of allowed iterations",
        "MAXTIME": "Expected to reach maximum allowed time in next iteration",
        "NOT_FINITE": "Encountered non-finite fval/grad/hess",
        "EXCEEDED_BOUNDARY": "Exceeded specified boundaries",
        "DELTA_TOO_SMALL": "Trust Region Radius too small to proceed",
        "FTOL": "Converged according to fval difference",
        "XTOL": "Converged according to x difference",
        "GTOL": "Converged according to gradient norm",
    }

    out = messages.get(exitflag.name)

    return out


def _create_hessian_updater_from_user_input(hessian_update_strategy):
    from fides import hessian_approximation

    hessians_needing_residuals = (
        hessian_approximation.FX,
        hessian_approximation.SSM,
        hessian_approximation.TSSM,
        hessian_approximation.GNSBFGS,
    )
    unsupported_hess_msg = (
        f"{hessian_update_strategy} not supported because it requires "
        "residuals. Choose one of 'BB', 'BFGS', 'BG', 'DFP' or 'SR1' or pass "
        "an instance of the fides.hessian_approximation.HessianApproximation "
        "class."
    )

    if hessian_update_strategy in ("broyden", "Broyden", "BROYDEN"):
        raise ValueError(
            "You cannot use the Broyden update strategy without specifying the "
            "interpolation parameter phi. Import the Broyden class from "
            "`fides.hessian_approximation`, create an instance of it with your "
            "desired value of phi and pass this instance instead."
        )
    elif isinstance(hessian_update_strategy, str):
        if hessian_update_strategy.lower() in ["fx", "ssm", "tssm", "gnsbfgs"]:
            raise NotImplementedError(unsupported_hess_msg)
        else:
            hessian_name = hessian_update_strategy.upper()
            hessian_class = getattr(hessian_approximation, hessian_name)
            hessian_instance = hessian_class()
    elif isinstance(
        hessian_update_strategy, hessian_approximation.HessianApproximation
    ):
        hessian_instance = hessian_update_strategy
        if isinstance(hessian_instance, hessians_needing_residuals):
            raise NotImplementedError(unsupported_hess_msg)
    else:
        raise TypeError(
            "You must provide a hessian_update_strategy that is either a string or an "
            "instance of the fides.hessian_approximation.HessianApproximation class."
        )
    return hessian_instance
