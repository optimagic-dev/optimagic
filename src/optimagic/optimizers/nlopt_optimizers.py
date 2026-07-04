"""Implement `nlopt` algorithms.

The documentation is heavily based on the documentation of the `NLopt library
<https://nlopt.readthedocs.io>`_.

"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from optimagic import mark
from optimagic.config import IS_NLOPT_INSTALLED
from optimagic.optimization.algo_options import (
    CONVERGENCE_FTOL_ABS,
    CONVERGENCE_FTOL_REL,
    CONVERGENCE_XTOL_ABS,
    CONVERGENCE_XTOL_REL,
    STOPPING_MAXFUN,
    STOPPING_MAXFUN_GLOBAL,
)
from optimagic.optimization.algorithm import Algorithm, InternalOptimizeResult
from optimagic.optimization.internal_optimization_problem import (
    InternalOptimizationProblem,
)
from optimagic.parameters.nonlinear_constraints import (
    equality_as_inequality_constraints,
)
from optimagic.typing import (
    AggregationLevel,
    NonNegativeFloat,
    PositiveInt,
)

if IS_NLOPT_INSTALLED:
    import nlopt


@mark.minimizer(
    name="nlopt_bobyqa",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_NLOPT_INSTALLED,
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
class NloptBOBYQA(Algorithm):
    """Minimize a scalar function using the BOBYQA algorithm.

    BOBYQA (Bound Optimization BY Quadratic Approximation) is a derivative-free local
    optimizer for bound-constrained problems by M. J. D. Powell :cite:`Powell2009`.
    It iteratively constructs a quadratic approximation of the objective function
    from function values only and minimizes this model in a trust region.

    The algorithm is a good choice for smooth scalar problems with a moderate number
    of parameters when derivatives are not available. Because it relies on a
    quadratic model, it may perform poorly for objective functions that are not
    twice-differentiable.

    BOBYQA is the successor of Powell's NEWUOA algorithm (``nlopt_newuoa``) and
    largely supersedes it; in contrast to NEWUOA, it supports bound constraints
    natively. For noisy or very non-smooth problems, simplex-based methods such as
    ``nlopt_neldermead`` or ``nlopt_sbplx`` can be more robust.

    The implementation in NLopt :cite:`Johnson2007` is a C translation of Powell's
    original Fortran subroutine. In addition, it supports all NLopt stopping
    criteria and unequal initial step sizes in the different parameters, achieved
    by internally rescaling the parameters proportional to the initial steps.

    """

    convergence_xtol_rel: NonNegativeFloat = CONVERGENCE_XTOL_REL
    """Stop when an optimization step (or an estimate of the optimum) changes every
    parameter by less than this value multiplied by the absolute value of the
    parameter.

    This corresponds to NLopt's ``xtol_rel`` stopping criterion. A value of 0 means
    that this criterion is disabled.

    """

    convergence_xtol_abs: NonNegativeFloat = CONVERGENCE_XTOL_ABS
    """Stop when an optimization step (or an estimate of the optimum) changes every
    parameter by less than this absolute value.

    This corresponds to NLopt's ``xtol_abs`` stopping criterion. A value of 0 means
    that this criterion is disabled, which is the optimagic default.

    """

    convergence_ftol_rel: NonNegativeFloat = CONVERGENCE_FTOL_REL
    """Stop when an optimization step (or an estimate of the optimum) changes the
    objective function value by less than this value multiplied by the absolute
    value of the function value.

    This corresponds to NLopt's ``ftol_rel`` stopping criterion. A value of 0 means
    that this criterion is disabled.

    """

    convergence_ftol_abs: NonNegativeFloat = CONVERGENCE_FTOL_ABS
    """Stop when an optimization step (or an estimate of the optimum) changes the
    objective function value by less than this absolute value.

    This corresponds to NLopt's ``ftol_abs`` stopping criterion. A value of 0 means
    that this criterion is disabled, which is the optimagic default.

    """

    stopping_maxfun: PositiveInt = STOPPING_MAXFUN
    """Maximum number of objective function evaluations.

    If reached, the optimization stops, but this is not counted as successful
    convergence. This corresponds to NLopt's ``maxeval`` stopping criterion.

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        res = _minimize_nlopt(
            problem=problem,
            x0=x0,
            is_global=self.algo_info.is_global,
            convergence_xtol_rel=self.convergence_xtol_rel,
            convergence_xtol_abs=self.convergence_xtol_abs,
            convergence_ftol_rel=self.convergence_ftol_rel,
            convergence_ftol_abs=self.convergence_ftol_abs,
            stopping_max_eval=self.stopping_maxfun,
            algorithm=nlopt.LN_BOBYQA,
        )

        return res


@mark.minimizer(
    name="nlopt_neldermead",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_NLOPT_INSTALLED,
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
class NloptNelderMead(Algorithm):
    """Minimize a scalar function using the Nelder-Mead simplex algorithm.

    The Nelder-Mead algorithm :cite:`Nelder1965` is a derivative-free local
    optimizer that maintains a simplex of n+1 points in n dimensions. In each
    iteration, the simplex is updated by reflecting, expanding or contracting it
    away from the worst point.

    The algorithm is a slow but robust workhorse that only requires function
    values. It can make progress on noisy or non-smooth problems where model-based
    methods such as ``nlopt_bobyqa`` fail, but it typically needs many function
    evaluations. The related ``nlopt_sbplx`` algorithm applies Nelder-Mead on a
    sequence of subspaces and is claimed to be more efficient and robust.

    The implementation in NLopt :cite:`Johnson2007` differs from the original
    algorithm in that it explicitly supports bound constraints: whenever a new
    point would lie outside the bounds, it is moved exactly onto the boundary.

    """

    convergence_xtol_rel: NonNegativeFloat = CONVERGENCE_XTOL_REL
    """Stop when an optimization step (or an estimate of the optimum) changes every
    parameter by less than this value multiplied by the absolute value of the
    parameter.

    This corresponds to NLopt's ``xtol_rel`` stopping criterion. A value of 0 means
    that this criterion is disabled.

    """

    convergence_xtol_abs: NonNegativeFloat = CONVERGENCE_XTOL_ABS
    """Stop when an optimization step (or an estimate of the optimum) changes every
    parameter by less than this absolute value.

    This corresponds to NLopt's ``xtol_abs`` stopping criterion. A value of 0 means
    that this criterion is disabled, which is the optimagic default.

    """

    convergence_ftol_abs: NonNegativeFloat = CONVERGENCE_FTOL_ABS
    """Stop when an optimization step (or an estimate of the optimum) changes the
    objective function value by less than this absolute value.

    This corresponds to NLopt's ``ftol_abs`` stopping criterion. A value of 0 means
    that this criterion is disabled, which is the optimagic default.

    """

    stopping_maxfun: PositiveInt = STOPPING_MAXFUN
    """Maximum number of objective function evaluations.

    If reached, the optimization stops, but this is not counted as successful
    convergence. This corresponds to NLopt's ``maxeval`` stopping criterion.

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        res = _minimize_nlopt(
            problem=problem,
            x0=x0,
            is_global=self.algo_info.is_global,
            convergence_xtol_rel=self.convergence_xtol_rel,
            convergence_xtol_abs=self.convergence_xtol_abs,
            convergence_ftol_abs=self.convergence_ftol_abs,
            stopping_max_eval=self.stopping_maxfun,
            algorithm=nlopt.LN_NELDERMEAD,
        )

        return res


@mark.minimizer(
    name="nlopt_praxis",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_NLOPT_INSTALLED,
    is_global=False,
    needs_jac=False,
    needs_hess=False,
    needs_bounds=False,
    supports_parallelism=False,
    supports_bounds=False,
    supports_infinite_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class NloptPRAXIS(Algorithm):
    """Minimize a scalar function using the principal-axis method.

    PRAXIS is a derivative-free local optimizer by Richard Brent :cite:`Brent1972`.
    It assumes an approximately quadratic form of the objective function and
    repeatedly updates a set of conjugate search directions (the principal axes)
    along which one-dimensional minimizations are performed.

    The algorithm is not invariant to scaling of the objective function and may
    fail under certain rank-preserving transformations of the objective (e.g.
    transformations that lead to a non-quadratic shape of the objective function).

    The algorithm is not deterministic (it uses random perturbations to avoid
    stagnation) and determinism cannot be achieved via seed setting.

    The implementation in NLopt :cite:`Johnson2007` is based on a C translation of
    Brent's original Fortran code. The original algorithm was designed for
    unconstrained optimization; NLopt only emulates bound constraints, at the cost
    of a significantly reduced speed of convergence. For bound-constrained
    problems, this method is dominated by algorithms with native bound support
    such as ``nlopt_bobyqa`` and ``nlopt_cobyla``.

    .. warning::
        The NLopt implementation failed on a simple benchmark function with finite
        parameter bounds. Passing bounds has therefore been disabled for this
        algorithm in optimagic.

    """

    convergence_xtol_rel: NonNegativeFloat = CONVERGENCE_XTOL_REL
    """Stop when an optimization step (or an estimate of the optimum) changes every
    parameter by less than this value multiplied by the absolute value of the
    parameter.

    This corresponds to NLopt's ``xtol_rel`` stopping criterion. A value of 0 means
    that this criterion is disabled.

    """

    convergence_xtol_abs: NonNegativeFloat = CONVERGENCE_XTOL_ABS
    """Stop when an optimization step (or an estimate of the optimum) changes every
    parameter by less than this absolute value.

    This corresponds to NLopt's ``xtol_abs`` stopping criterion. A value of 0 means
    that this criterion is disabled, which is the optimagic default.

    """

    convergence_ftol_rel: NonNegativeFloat = CONVERGENCE_FTOL_REL
    """Stop when an optimization step (or an estimate of the optimum) changes the
    objective function value by less than this value multiplied by the absolute
    value of the function value.

    This corresponds to NLopt's ``ftol_rel`` stopping criterion. A value of 0 means
    that this criterion is disabled.

    """

    convergence_ftol_abs: NonNegativeFloat = CONVERGENCE_FTOL_ABS
    """Stop when an optimization step (or an estimate of the optimum) changes the
    objective function value by less than this absolute value.

    This corresponds to NLopt's ``ftol_abs`` stopping criterion. A value of 0 means
    that this criterion is disabled, which is the optimagic default.

    """

    stopping_maxfun: PositiveInt = STOPPING_MAXFUN
    """Maximum number of objective function evaluations.

    If reached, the optimization stops, but this is not counted as successful
    convergence. This corresponds to NLopt's ``maxeval`` stopping criterion.

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        res = _minimize_nlopt(
            problem=problem,
            x0=x0,
            is_global=self.algo_info.is_global,
            convergence_xtol_rel=self.convergence_xtol_rel,
            convergence_xtol_abs=self.convergence_xtol_abs,
            convergence_ftol_rel=self.convergence_ftol_rel,
            convergence_ftol_abs=self.convergence_ftol_abs,
            stopping_max_eval=self.stopping_maxfun,
            algorithm=nlopt.LN_PRAXIS,
        )

        return res


@mark.minimizer(
    name="nlopt_cobyla",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_NLOPT_INSTALLED,
    is_global=False,
    needs_jac=False,
    needs_hess=False,
    needs_bounds=False,
    supports_parallelism=False,
    supports_bounds=True,
    supports_infinite_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=True,
    disable_history=False,
)
@dataclass(frozen=True)
class NloptCOBYLA(Algorithm):
    """Minimize a scalar function using the COBYLA algorithm.

    COBYLA (Constrained Optimization BY Linear Approximations) is a derivative-free
    local optimizer by M. J. D. Powell that supports nonlinear inequality and
    equality constraints :cite:`Powell1994`.

    It constructs successive linear approximations of the objective function and
    constraints via a simplex of n+1 points (in n dimensions) and optimizes these
    approximations in a trust region at each step.

    Use it for scalar problems without derivatives, in particular when nonlinear
    constraints are present: it is the only derivative-free NLopt optimizer in
    optimagic that supports them. For unconstrained or bound-constrained smooth
    problems, ``nlopt_bobyqa`` usually converges faster because it uses quadratic
    instead of linear approximations.

    The implementation in NLopt :cite:`Johnson2007` is based on a C translation of
    Powell's original Fortran code and differs from the original implementation in
    a few ways:

    - It incorporates all of the NLopt termination criteria.
    - It adds explicit support for bound constraints.
    - It allows the trust-region radius to increase if the predicted improvement
      was approximately right and the simplex is satisfactory, which can improve
      convergence speed.
    - It pseudo-randomizes the simplex steps, improving robustness by avoiding
      accidentally taking steps that do not improve conditioning, while preserving
      the deterministic nature of the algorithm (a deterministic seed is used).
    - It supports unequal initial step sizes in the different parameters.

    Since the underlying code only supports inequality constraints, equality
    constraints are automatically transformed into pairs of inequality constraints
    by NLopt.

    """

    convergence_xtol_rel: NonNegativeFloat = CONVERGENCE_XTOL_REL
    """Stop when an optimization step (or an estimate of the optimum) changes every
    parameter by less than this value multiplied by the absolute value of the
    parameter.

    This corresponds to NLopt's ``xtol_rel`` stopping criterion. A value of 0 means
    that this criterion is disabled.

    """

    convergence_xtol_abs: NonNegativeFloat = CONVERGENCE_XTOL_ABS
    """Stop when an optimization step (or an estimate of the optimum) changes every
    parameter by less than this absolute value.

    This corresponds to NLopt's ``xtol_abs`` stopping criterion. A value of 0 means
    that this criterion is disabled, which is the optimagic default.

    """

    convergence_ftol_rel: NonNegativeFloat = CONVERGENCE_FTOL_REL
    """Stop when an optimization step (or an estimate of the optimum) changes the
    objective function value by less than this value multiplied by the absolute
    value of the function value.

    This corresponds to NLopt's ``ftol_rel`` stopping criterion. A value of 0 means
    that this criterion is disabled.

    """

    convergence_ftol_abs: NonNegativeFloat = CONVERGENCE_FTOL_ABS
    """Stop when an optimization step (or an estimate of the optimum) changes the
    objective function value by less than this absolute value.

    This corresponds to NLopt's ``ftol_abs`` stopping criterion. A value of 0 means
    that this criterion is disabled, which is the optimagic default.

    """

    stopping_maxfun: PositiveInt = STOPPING_MAXFUN
    """Maximum number of objective function evaluations.

    If reached, the optimization stops, but this is not counted as successful
    convergence. This corresponds to NLopt's ``maxeval`` stopping criterion.

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        res = _minimize_nlopt(
            problem=problem,
            x0=x0,
            is_global=self.algo_info.is_global,
            convergence_xtol_rel=self.convergence_xtol_rel,
            convergence_xtol_abs=self.convergence_xtol_abs,
            convergence_ftol_rel=self.convergence_ftol_rel,
            convergence_ftol_abs=self.convergence_ftol_abs,
            stopping_max_eval=self.stopping_maxfun,
            algorithm=nlopt.LN_COBYLA,
        )

        return res


@mark.minimizer(
    name="nlopt_sbplx",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_NLOPT_INSTALLED,
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
class NloptSbplx(Algorithm):
    """Minimize a scalar function using the Subplex algorithm.

    Sbplx is a re-implementation of Tom Rowan's Subplex method :cite:`Rowan1990`.
    Subplex is a variant of the Nelder-Mead algorithm (``nlopt_neldermead``) that
    applies Nelder-Mead on a sequence of subspaces. It is claimed to be more
    efficient and robust than the original Nelder-Mead algorithm, while retaining
    its ability to make progress on noisy and non-smooth objective functions.

    The implementation in NLopt :cite:`Johnson2007` was rewritten from scratch
    because Rowan's original code could not be relicensed. It differs from the
    original algorithm in that it explicitly supports bound constraints, which
    gives a big improvement in the case where the optimum lies against one of the
    bounds.

    """

    convergence_xtol_rel: NonNegativeFloat = CONVERGENCE_XTOL_REL
    """Stop when an optimization step (or an estimate of the optimum) changes every
    parameter by less than this value multiplied by the absolute value of the
    parameter.

    This corresponds to NLopt's ``xtol_rel`` stopping criterion. A value of 0 means
    that this criterion is disabled.

    """

    convergence_xtol_abs: NonNegativeFloat = CONVERGENCE_XTOL_ABS
    """Stop when an optimization step (or an estimate of the optimum) changes every
    parameter by less than this absolute value.

    This corresponds to NLopt's ``xtol_abs`` stopping criterion. A value of 0 means
    that this criterion is disabled, which is the optimagic default.

    """

    convergence_ftol_rel: NonNegativeFloat = CONVERGENCE_FTOL_REL
    """Stop when an optimization step (or an estimate of the optimum) changes the
    objective function value by less than this value multiplied by the absolute
    value of the function value.

    This corresponds to NLopt's ``ftol_rel`` stopping criterion. A value of 0 means
    that this criterion is disabled.

    """

    convergence_ftol_abs: NonNegativeFloat = CONVERGENCE_FTOL_ABS
    """Stop when an optimization step (or an estimate of the optimum) changes the
    objective function value by less than this absolute value.

    This corresponds to NLopt's ``ftol_abs`` stopping criterion. A value of 0 means
    that this criterion is disabled, which is the optimagic default.

    """

    stopping_maxfun: PositiveInt = STOPPING_MAXFUN
    """Maximum number of objective function evaluations.

    If reached, the optimization stops, but this is not counted as successful
    convergence. This corresponds to NLopt's ``maxeval`` stopping criterion.

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        res = _minimize_nlopt(
            problem=problem,
            x0=x0,
            is_global=self.algo_info.is_global,
            convergence_xtol_rel=self.convergence_xtol_rel,
            convergence_xtol_abs=self.convergence_xtol_abs,
            convergence_ftol_rel=self.convergence_ftol_rel,
            convergence_ftol_abs=self.convergence_ftol_abs,
            stopping_max_eval=self.stopping_maxfun,
            algorithm=nlopt.LN_SBPLX,
        )

        return res


@mark.minimizer(
    name="nlopt_newuoa",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_NLOPT_INSTALLED,
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
class NloptNEWUOA(Algorithm):
    """Minimize a scalar function using the NEWUOA algorithm.

    NEWUOA is a derivative-free local optimizer for unconstrained optimization by
    M. J. D. Powell :cite:`Powell2004`. Like BOBYQA, it iteratively constructs a
    quadratic approximation of the objective function from function values only
    and minimizes this model in a trust region.

    Use it for smooth scalar problems when derivatives are not available. NEWUOA
    is largely superseded by its successor BOBYQA (``nlopt_bobyqa``), which
    supports bound constraints natively and is usually preferable, in particular
    for bound-constrained problems.

    The version in NLopt :cite:`Johnson2007` has been extended to support bound
    constraints: if no finite bounds are specified, optimagic calls the original
    ``LN_NEWUOA`` algorithm for unconstrained optimization; otherwise, the
    ``LN_NEWUOA_BOUND`` variant for bound-constrained problems is used.

    .. note::
        NEWUOA requires the dimension n of the parameter space to be at least 2,
        i.e. the implementation does not handle one-dimensional optimization
        problems.

    """

    convergence_xtol_rel: NonNegativeFloat = CONVERGENCE_XTOL_REL
    """Stop when an optimization step (or an estimate of the optimum) changes every
    parameter by less than this value multiplied by the absolute value of the
    parameter.

    This corresponds to NLopt's ``xtol_rel`` stopping criterion. A value of 0 means
    that this criterion is disabled.

    """

    convergence_xtol_abs: NonNegativeFloat = CONVERGENCE_XTOL_ABS
    """Stop when an optimization step (or an estimate of the optimum) changes every
    parameter by less than this absolute value.

    This corresponds to NLopt's ``xtol_abs`` stopping criterion. A value of 0 means
    that this criterion is disabled, which is the optimagic default.

    """

    convergence_ftol_rel: NonNegativeFloat = CONVERGENCE_FTOL_REL
    """Stop when an optimization step (or an estimate of the optimum) changes the
    objective function value by less than this value multiplied by the absolute
    value of the function value.

    This corresponds to NLopt's ``ftol_rel`` stopping criterion. A value of 0 means
    that this criterion is disabled.

    """

    convergence_ftol_abs: NonNegativeFloat = CONVERGENCE_FTOL_ABS
    """Stop when an optimization step (or an estimate of the optimum) changes the
    objective function value by less than this absolute value.

    This corresponds to NLopt's ``ftol_abs`` stopping criterion. A value of 0 means
    that this criterion is disabled, which is the optimagic default.

    """

    stopping_maxfun: PositiveInt = STOPPING_MAXFUN
    """Maximum number of objective function evaluations.

    If reached, the optimization stops, but this is not counted as successful
    convergence. This corresponds to NLopt's ``maxeval`` stopping criterion.

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if problem.bounds.lower is None or problem.bounds.upper is None:
            algo = nlopt.LN_NEWUOA
        elif np.any(np.isfinite(problem.bounds.lower)) or np.any(
            np.isfinite(problem.bounds.upper)
        ):
            algo = nlopt.LN_NEWUOA_BOUND
        else:
            algo = nlopt.LN_NEWUOA
        res = _minimize_nlopt(
            problem=problem,
            x0=x0,
            is_global=self.algo_info.is_global,
            convergence_xtol_rel=self.convergence_xtol_rel,
            convergence_xtol_abs=self.convergence_xtol_abs,
            convergence_ftol_rel=self.convergence_ftol_rel,
            convergence_ftol_abs=self.convergence_ftol_abs,
            stopping_max_eval=self.stopping_maxfun,
            algorithm=algo,
        )

        return res


@mark.minimizer(
    name="nlopt_tnewton",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_NLOPT_INSTALLED,
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
class NloptTNewton(Algorithm):
    """Minimize a scalar function using a truncated Newton method.

    Truncated Newton methods are designed for large-scale smooth optimization
    problems. Instead of solving the Newton equations exactly, which would require
    the Hessian matrix, they solve them approximately (hence "truncated" or
    "inexact") with a conjugate-gradient iteration that only requires gradient
    information. A detailed description is given in :cite:`Dembo1983`.

    Use it for differentiable scalar problems, in particular with many parameters.
    The gradient of the objective function is required.

    The implementation in NLopt :cite:`Johnson2007` is based on a Fortran
    implementation written by Prof. Ladislav Luksan. NLopt also provides variants
    of the method that add preconditioning via a low-storage BFGS approximation
    and steepest-descent restarting; optimagic wraps the plain variant
    (``LD_TNEWTON``) without preconditioning or restarting.

    """

    convergence_xtol_rel: NonNegativeFloat = CONVERGENCE_XTOL_REL
    """Stop when an optimization step (or an estimate of the optimum) changes every
    parameter by less than this value multiplied by the absolute value of the
    parameter.

    This corresponds to NLopt's ``xtol_rel`` stopping criterion. A value of 0 means
    that this criterion is disabled.

    """

    convergence_xtol_abs: NonNegativeFloat = CONVERGENCE_XTOL_ABS
    """Stop when an optimization step (or an estimate of the optimum) changes every
    parameter by less than this absolute value.

    This corresponds to NLopt's ``xtol_abs`` stopping criterion. A value of 0 means
    that this criterion is disabled, which is the optimagic default.

    """

    convergence_ftol_rel: NonNegativeFloat = CONVERGENCE_FTOL_REL
    """Stop when an optimization step (or an estimate of the optimum) changes the
    objective function value by less than this value multiplied by the absolute
    value of the function value.

    This corresponds to NLopt's ``ftol_rel`` stopping criterion. A value of 0 means
    that this criterion is disabled.

    """

    convergence_ftol_abs: NonNegativeFloat = CONVERGENCE_FTOL_ABS
    """Stop when an optimization step (or an estimate of the optimum) changes the
    objective function value by less than this absolute value.

    This corresponds to NLopt's ``ftol_abs`` stopping criterion. A value of 0 means
    that this criterion is disabled, which is the optimagic default.

    """

    stopping_maxfun: PositiveInt = STOPPING_MAXFUN
    """Maximum number of objective function evaluations.

    If reached, the optimization stops, but this is not counted as successful
    convergence. This corresponds to NLopt's ``maxeval`` stopping criterion.

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        res = _minimize_nlopt(
            problem=problem,
            x0=x0,
            is_global=self.algo_info.is_global,
            convergence_xtol_rel=self.convergence_xtol_rel,
            convergence_xtol_abs=self.convergence_xtol_abs,
            convergence_ftol_rel=self.convergence_ftol_rel,
            convergence_ftol_abs=self.convergence_ftol_abs,
            stopping_max_eval=self.stopping_maxfun,
            algorithm=nlopt.LD_TNEWTON,
        )

        return res


@mark.minimizer(
    name="nlopt_lbfgsb",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_NLOPT_INSTALLED,
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
class NloptLBFGSB(Algorithm):
    """Minimize a scalar function using the low-storage BFGS algorithm.

    L-BFGS is a limited-memory quasi-Newton method. It approximates the inverse
    Hessian of the objective function from a limited number of stored past gradient
    and parameter updates and uses this approximation to compute search directions
    for a line search. Detailed descriptions of the algorithm are given in
    :cite:`Nocedal1980` and :cite:`Nocedal1989`.

    Use it for smooth, differentiable scalar problems, in particular with many
    parameters: due to the limited-memory Hessian approximation, memory use grows
    only linearly in the number of parameters. The gradient of the objective
    function is required.

    The implementation in NLopt :cite:`Johnson2007` is based on a Fortran
    implementation of the low-storage BFGS algorithm written by Prof. Ladislav
    Luksan. It is a different implementation than the one wrapped by
    ``scipy_lbfgsb``, which is based on the original Fortran code by Nocedal and
    coworkers.

    """

    convergence_xtol_rel: NonNegativeFloat = CONVERGENCE_XTOL_REL
    """Stop when an optimization step (or an estimate of the optimum) changes every
    parameter by less than this value multiplied by the absolute value of the
    parameter.

    This corresponds to NLopt's ``xtol_rel`` stopping criterion. A value of 0 means
    that this criterion is disabled.

    """

    convergence_xtol_abs: NonNegativeFloat = CONVERGENCE_XTOL_ABS
    """Stop when an optimization step (or an estimate of the optimum) changes every
    parameter by less than this absolute value.

    This corresponds to NLopt's ``xtol_abs`` stopping criterion. A value of 0 means
    that this criterion is disabled, which is the optimagic default.

    """

    convergence_ftol_rel: NonNegativeFloat = CONVERGENCE_FTOL_REL
    """Stop when an optimization step (or an estimate of the optimum) changes the
    objective function value by less than this value multiplied by the absolute
    value of the function value.

    This corresponds to NLopt's ``ftol_rel`` stopping criterion. A value of 0 means
    that this criterion is disabled.

    """

    convergence_ftol_abs: NonNegativeFloat = CONVERGENCE_FTOL_ABS
    """Stop when an optimization step (or an estimate of the optimum) changes the
    objective function value by less than this absolute value.

    This corresponds to NLopt's ``ftol_abs`` stopping criterion. A value of 0 means
    that this criterion is disabled, which is the optimagic default.

    """

    stopping_maxfun: PositiveInt = STOPPING_MAXFUN
    """Maximum number of objective function evaluations.

    If reached, the optimization stops, but this is not counted as successful
    convergence. This corresponds to NLopt's ``maxeval`` stopping criterion.

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        res = _minimize_nlopt(
            problem=problem,
            x0=x0,
            is_global=self.algo_info.is_global,
            convergence_xtol_rel=self.convergence_xtol_rel,
            convergence_xtol_abs=self.convergence_xtol_abs,
            convergence_ftol_rel=self.convergence_ftol_rel,
            convergence_ftol_abs=self.convergence_ftol_abs,
            stopping_max_eval=self.stopping_maxfun,
            algorithm=nlopt.LD_LBFGS,
        )

        return res


@mark.minimizer(
    name="nlopt_ccsaq",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_NLOPT_INSTALLED,
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
class NloptCCSAQ(Algorithm):
    """Minimize a scalar function using the CCSAQ algorithm.

    CCSAQ uses the quadratic variant of the conservative convex separable
    approximation (CCSA) framework described in :cite:`Svanberg2002`. It is a
    gradient-based local optimizer: at each candidate point, a separable quadratic
    approximation of the objective function is constructed from the gradient, plus
    a penalty term that renders the approximation convex and conservative. The
    approximation is then minimized to obtain the next candidate point.

    The algorithm is "globally convergent" in the sense that it is guaranteed to
    converge to a local optimum from any feasible starting point; it does not find
    the global optimum.

    CCSAQ is closely related to ``nlopt_mma``, which belongs to the same CCSA
    family but approximates the objective with moving asymptotes instead of a
    quadratic model.

    .. note::
        The underlying NLopt implementation :cite:`Johnson2007` supports nonlinear
        inequality constraints, but nonlinear constraints are not supported for
        this algorithm in optimagic. Use ``nlopt_mma`` or ``nlopt_slsqp`` for
        nonlinearly constrained problems.

    """

    convergence_xtol_rel: NonNegativeFloat = CONVERGENCE_XTOL_REL
    """Stop when an optimization step (or an estimate of the optimum) changes every
    parameter by less than this value multiplied by the absolute value of the
    parameter.

    This corresponds to NLopt's ``xtol_rel`` stopping criterion. A value of 0 means
    that this criterion is disabled.

    """

    convergence_xtol_abs: NonNegativeFloat = CONVERGENCE_XTOL_ABS
    """Stop when an optimization step (or an estimate of the optimum) changes every
    parameter by less than this absolute value.

    This corresponds to NLopt's ``xtol_abs`` stopping criterion. A value of 0 means
    that this criterion is disabled, which is the optimagic default.

    """

    convergence_ftol_rel: NonNegativeFloat = CONVERGENCE_FTOL_REL
    """Stop when an optimization step (or an estimate of the optimum) changes the
    objective function value by less than this value multiplied by the absolute
    value of the function value.

    This corresponds to NLopt's ``ftol_rel`` stopping criterion. A value of 0 means
    that this criterion is disabled.

    """

    convergence_ftol_abs: NonNegativeFloat = CONVERGENCE_FTOL_ABS
    """Stop when an optimization step (or an estimate of the optimum) changes the
    objective function value by less than this absolute value.

    This corresponds to NLopt's ``ftol_abs`` stopping criterion. A value of 0 means
    that this criterion is disabled, which is the optimagic default.

    """

    stopping_maxfun: PositiveInt = STOPPING_MAXFUN
    """Maximum number of objective function evaluations.

    If reached, the optimization stops, but this is not counted as successful
    convergence. This corresponds to NLopt's ``maxeval`` stopping criterion.

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        res = _minimize_nlopt(
            problem=problem,
            x0=x0,
            is_global=self.algo_info.is_global,
            convergence_xtol_rel=self.convergence_xtol_rel,
            convergence_xtol_abs=self.convergence_xtol_abs,
            convergence_ftol_rel=self.convergence_ftol_rel,
            convergence_ftol_abs=self.convergence_ftol_abs,
            stopping_max_eval=self.stopping_maxfun,
            algorithm=nlopt.LD_CCSAQ,
        )

        return res


@mark.minimizer(
    name="nlopt_mma",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_NLOPT_INSTALLED,
    is_global=False,
    needs_jac=True,
    needs_hess=False,
    needs_bounds=False,
    supports_parallelism=False,
    supports_bounds=True,
    supports_infinite_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=True,
    disable_history=False,
)
@dataclass(frozen=True)
class NloptMMA(Algorithm):
    """Minimize a scalar function using the method of moving asymptotes (MMA).

    MMA is a gradient-based local optimizer that was originally developed for
    structural optimization :cite:`Svanberg1987`. The implementation in NLopt
    :cite:`Johnson2007` is based on the globally convergent variant of the
    algorithm described in :cite:`Svanberg2002`.

    At each candidate point, an approximation of the objective function and
    constraints is constructed from the gradient and so-called moving asymptotes
    that are updated across iterations. A penalty term renders the approximation
    convex and conservative. The resulting convex separable subproblem is solved
    to obtain the next candidate point. The algorithm is "globally convergent" in
    the sense that it is guaranteed to converge to a local optimum from any
    feasible starting point; it does not find the global optimum.

    Use it for smooth scalar problems with bound and nonlinear constraints when
    gradients are available. The underlying implementation supports nonlinear
    inequality constraints, but no equality constraints; optimagic automatically
    converts equality constraints into pairs of inequality constraints, so both
    types can be specified.

    """

    convergence_xtol_rel: NonNegativeFloat = CONVERGENCE_XTOL_REL
    """Stop when an optimization step (or an estimate of the optimum) changes every
    parameter by less than this value multiplied by the absolute value of the
    parameter.

    This corresponds to NLopt's ``xtol_rel`` stopping criterion. A value of 0 means
    that this criterion is disabled.

    """

    convergence_xtol_abs: NonNegativeFloat = CONVERGENCE_XTOL_ABS
    """Stop when an optimization step (or an estimate of the optimum) changes every
    parameter by less than this absolute value.

    This corresponds to NLopt's ``xtol_abs`` stopping criterion. A value of 0 means
    that this criterion is disabled, which is the optimagic default.

    """

    convergence_ftol_rel: NonNegativeFloat = CONVERGENCE_FTOL_REL
    """Stop when an optimization step (or an estimate of the optimum) changes the
    objective function value by less than this value multiplied by the absolute
    value of the function value.

    This corresponds to NLopt's ``ftol_rel`` stopping criterion. A value of 0 means
    that this criterion is disabled.

    """

    convergence_ftol_abs: NonNegativeFloat = CONVERGENCE_FTOL_ABS
    """Stop when an optimization step (or an estimate of the optimum) changes the
    objective function value by less than this absolute value.

    This corresponds to NLopt's ``ftol_abs`` stopping criterion. A value of 0 means
    that this criterion is disabled, which is the optimagic default.

    """

    stopping_maxfun: PositiveInt = STOPPING_MAXFUN
    """Maximum number of objective function evaluations.

    If reached, the optimization stops, but this is not counted as successful
    convergence. This corresponds to NLopt's ``maxeval`` stopping criterion.

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        nonlinear_constraints = equality_as_inequality_constraints(
            problem.nonlinear_constraints
        )

        res = _minimize_nlopt(
            problem=problem,
            x0=x0,
            is_global=self.algo_info.is_global,
            convergence_xtol_rel=self.convergence_xtol_rel,
            convergence_xtol_abs=self.convergence_xtol_abs,
            convergence_ftol_rel=self.convergence_ftol_rel,
            convergence_ftol_abs=self.convergence_ftol_abs,
            stopping_max_eval=self.stopping_maxfun,
            algorithm=nlopt.LD_MMA,
            nonlinear_constraints=nonlinear_constraints,
        )

        return res


@mark.minimizer(
    name="nlopt_var",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_NLOPT_INSTALLED,
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
class NloptVAR(Algorithm):
    """Minimize a scalar function using a limited-memory variable-metric method.

    The algorithm is a shifted limited-memory variable-metric method by Vlcek and
    Luksan; a detailed description, including the rank-1 and rank-2 variants, is
    given in :cite:`Vlcek2006`. Like L-BFGS, it approximates the inverse Hessian
    from a limited number of stored past updates of the gradient. The larger the
    number of stored updates, the more memory is consumed.

    Use it for smooth, differentiable scalar problems, in particular with many
    parameters. The gradient of the objective function is required. Depending on
    the ``rank_1_update`` option, either a rank-1 (``LD_VAR1``) or a rank-2
    (``LD_VAR2``) update formula for the variable-metric approximation is used.

    The implementation in NLopt :cite:`Johnson2007` is based on a Fortran
    implementation written by Prof. Ladislav Luksan.

    """

    convergence_xtol_rel: NonNegativeFloat = CONVERGENCE_XTOL_REL
    """Stop when an optimization step (or an estimate of the optimum) changes every
    parameter by less than this value multiplied by the absolute value of the
    parameter.

    This corresponds to NLopt's ``xtol_rel`` stopping criterion. A value of 0 means
    that this criterion is disabled.

    """

    convergence_xtol_abs: NonNegativeFloat = CONVERGENCE_XTOL_ABS
    """Stop when an optimization step (or an estimate of the optimum) changes every
    parameter by less than this absolute value.

    This corresponds to NLopt's ``xtol_abs`` stopping criterion. A value of 0 means
    that this criterion is disabled, which is the optimagic default.

    """

    convergence_ftol_rel: NonNegativeFloat = CONVERGENCE_FTOL_REL
    """Stop when an optimization step (or an estimate of the optimum) changes the
    objective function value by less than this value multiplied by the absolute
    value of the function value.

    This corresponds to NLopt's ``ftol_rel`` stopping criterion. A value of 0 means
    that this criterion is disabled.

    """

    convergence_ftol_abs: NonNegativeFloat = CONVERGENCE_FTOL_ABS
    """Stop when an optimization step (or an estimate of the optimum) changes the
    objective function value by less than this absolute value.

    This corresponds to NLopt's ``ftol_abs`` stopping criterion. A value of 0 means
    that this criterion is disabled, which is the optimagic default.

    """

    stopping_maxfun: PositiveInt = STOPPING_MAXFUN
    """Maximum number of objective function evaluations.

    If reached, the optimization stops, but this is not counted as successful
    convergence. This corresponds to NLopt's ``maxeval`` stopping criterion.

    """

    rank_1_update: bool = True
    """Whether the rank-1 update method (``LD_VAR1``) is used.

    If False, the rank-2 update method (``LD_VAR2``) is used instead.

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if self.rank_1_update:
            algo = nlopt.LD_VAR1
        else:
            algo = nlopt.LD_VAR2
        res = _minimize_nlopt(
            problem=problem,
            x0=x0,
            is_global=self.algo_info.is_global,
            convergence_xtol_rel=self.convergence_xtol_rel,
            convergence_xtol_abs=self.convergence_xtol_abs,
            convergence_ftol_rel=self.convergence_ftol_rel,
            convergence_ftol_abs=self.convergence_ftol_abs,
            stopping_max_eval=self.stopping_maxfun,
            algorithm=algo,
        )

        return res


@mark.minimizer(
    name="nlopt_slsqp",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_NLOPT_INSTALLED,
    is_global=False,
    needs_jac=True,
    needs_hess=False,
    needs_bounds=False,
    supports_parallelism=False,
    supports_bounds=True,
    supports_infinite_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=True,
    disable_history=False,
)
@dataclass(frozen=True)
class NloptSLSQP(Algorithm):
    """Minimize a scalar function using the SLSQP algorithm.

    SLSQP (Sequential Least Squares Quadratic Programming) is a gradient-based
    local optimizer for nonlinearly constrained problems. It supports bound
    constraints as well as nonlinear equality and inequality constraints. The
    algorithm is a sequential quadratic programming (SQP) method that treats the
    optimization problem as a sequence of constrained least-squares problems.
    Despite its name, it is a general nonlinear optimizer for scalar objectives,
    not a least-squares solver.

    The implementation in NLopt :cite:`Johnson2007` is based on the procedure
    described in :cite:`Kraft1988` and :cite:`Kraft1994`.

    .. note::
        Because the underlying code uses dense-matrix methods, it requires O(n^2)
        storage and O(n^3) time in n dimensions, which makes it less practical for
        problems with more than a few thousand parameters.

    """

    convergence_xtol_rel: NonNegativeFloat = CONVERGENCE_XTOL_REL
    """Stop when an optimization step (or an estimate of the optimum) changes every
    parameter by less than this value multiplied by the absolute value of the
    parameter.

    This corresponds to NLopt's ``xtol_rel`` stopping criterion. A value of 0 means
    that this criterion is disabled.

    """

    convergence_xtol_abs: NonNegativeFloat = CONVERGENCE_XTOL_ABS
    """Stop when an optimization step (or an estimate of the optimum) changes every
    parameter by less than this absolute value.

    This corresponds to NLopt's ``xtol_abs`` stopping criterion. A value of 0 means
    that this criterion is disabled, which is the optimagic default.

    """

    convergence_ftol_rel: NonNegativeFloat = CONVERGENCE_FTOL_REL
    """Stop when an optimization step (or an estimate of the optimum) changes the
    objective function value by less than this value multiplied by the absolute
    value of the function value.

    This corresponds to NLopt's ``ftol_rel`` stopping criterion. A value of 0 means
    that this criterion is disabled.

    """

    convergence_ftol_abs: NonNegativeFloat = CONVERGENCE_FTOL_ABS
    """Stop when an optimization step (or an estimate of the optimum) changes the
    objective function value by less than this absolute value.

    This corresponds to NLopt's ``ftol_abs`` stopping criterion. A value of 0 means
    that this criterion is disabled, which is the optimagic default.

    """

    stopping_maxfun: PositiveInt = STOPPING_MAXFUN
    """Maximum number of objective function evaluations.

    If reached, the optimization stops, but this is not counted as successful
    convergence. This corresponds to NLopt's ``maxeval`` stopping criterion.

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        res = _minimize_nlopt(
            problem=problem,
            x0=x0,
            is_global=self.algo_info.is_global,
            convergence_xtol_rel=self.convergence_xtol_rel,
            convergence_xtol_abs=self.convergence_xtol_abs,
            convergence_ftol_rel=self.convergence_ftol_rel,
            convergence_ftol_abs=self.convergence_ftol_abs,
            stopping_max_eval=self.stopping_maxfun,
            algorithm=nlopt.LD_SLSQP,
        )

        return res


@mark.minimizer(
    name="nlopt_direct",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_NLOPT_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    needs_bounds=True,
    supports_parallelism=False,
    supports_bounds=True,
    supports_infinite_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class NloptDirect(Algorithm):
    """Minimize a scalar function using the DIRECT algorithm.

    DIRECT (DIviding RECTangles) is a deterministic derivative-free global
    optimizer :cite:`Jones1993`. It systematically divides the bounded search
    domain into smaller and smaller hyperrectangles and evaluates the objective
    function at their centers, balancing global exploration and local refinement.

    Use it for global optimization of scalar problems with a small number of
    parameters when no derivatives are available. Finite bounds on all parameters
    are required. The basic algorithm is deterministic, so results are
    reproducible without setting a seed.

    The implementation in NLopt :cite:`Johnson2007` provides several variants that
    are selected via the ``locally_biased``, ``random_search`` and
    ``unscaled_bounds`` options:

    - The locally biased variant DIRECT_L :cite:`Gablonsky2001` is more biased
      towards local search and tends to be more efficient for objective functions
      without too many local minima.
    - The randomized variants (only available for the locally biased version) use
      some randomization to decide which dimension to halve next in the case of
      near-ties.
    - By default, NLopt rescales the bounds to a unit hypercube. The unscaled
      ("NOSCAL") variants skip this rescaling, which can be better if the
      objective function varies on very different scales across dimensions.

    The available combinations are DIRECT, DIRECT_L, DIRECT_L_NOSCAL,
    DIRECT_L_RAND, DIRECT_L_RAND_NOSCAL and DIRECT_NOSCAL.

    """

    convergence_xtol_rel: NonNegativeFloat = CONVERGENCE_XTOL_REL
    """Stop when an optimization step (or an estimate of the optimum) changes every
    parameter by less than this value multiplied by the absolute value of the
    parameter.

    This corresponds to NLopt's ``xtol_rel`` stopping criterion. A value of 0 means
    that this criterion is disabled.

    """

    convergence_xtol_abs: NonNegativeFloat = CONVERGENCE_XTOL_ABS
    """Stop when an optimization step (or an estimate of the optimum) changes every
    parameter by less than this absolute value.

    This corresponds to NLopt's ``xtol_abs`` stopping criterion. A value of 0 means
    that this criterion is disabled, which is the optimagic default.

    """

    convergence_ftol_rel: NonNegativeFloat = CONVERGENCE_FTOL_REL
    """Stop when an optimization step (or an estimate of the optimum) changes the
    objective function value by less than this value multiplied by the absolute
    value of the function value.

    This corresponds to NLopt's ``ftol_rel`` stopping criterion. A value of 0 means
    that this criterion is disabled.

    """

    convergence_ftol_abs: NonNegativeFloat = CONVERGENCE_FTOL_ABS
    """Stop when an optimization step (or an estimate of the optimum) changes the
    objective function value by less than this absolute value.

    This corresponds to NLopt's ``ftol_abs`` stopping criterion. A value of 0 means
    that this criterion is disabled, which is the optimagic default.

    """

    stopping_maxfun: PositiveInt = STOPPING_MAXFUN_GLOBAL
    """Maximum number of objective function evaluations.

    If reached, the optimization stops, but this is not counted as successful
    convergence. This corresponds to NLopt's ``maxeval`` stopping criterion. The
    optimagic default for global optimizers is much lower than for local
    optimizers to keep runtimes manageable.

    """

    locally_biased: bool = False
    """Whether the locally biased variant DIRECT_L of the algorithm is used.

    The locally biased variant tends to be more efficient for objective functions
    without too many local minima. See :cite:`Gablonsky2001` for details.

    """

    random_search: bool = False
    """Whether the randomized variant of the locally biased algorithm is used.

    The randomized variant uses some randomization to decide which dimension to
    halve next in the case of near-ties. It is only available for the locally
    biased version of the algorithm, i.e. this option requires
    ``locally_biased=True``.

    """

    unscaled_bounds: bool = False
    """Whether the "NOSCAL" variant of the algorithm is used, which does not
    rescale the bounds to a unit hypercube.

    Skipping the rescaling can be beneficial if the objective function varies on
    very different scales across dimensions.

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if (
            not self.locally_biased
            and not self.random_search
            and not self.unscaled_bounds
        ):
            algo = nlopt.GN_DIRECT
        elif (
            self.locally_biased and not self.random_search and not self.unscaled_bounds
        ):
            algo = nlopt.GN_DIRECT_L
        elif self.locally_biased and not self.random_search and self.unscaled_bounds:
            algo = nlopt.GN_DIRECT_L_NOSCAL
        elif self.locally_biased and self.random_search and not self.unscaled_bounds:
            algo = nlopt.GN_DIRECT_L_RAND
        elif self.locally_biased and self.random_search and self.unscaled_bounds:
            algo = nlopt.GN_DIRECT_L_RAND_NOSCAL
        elif (
            not self.locally_biased and not self.random_search and self.unscaled_bounds
        ):
            algo = nlopt.GN_DIRECT_NOSCAL
        res = _minimize_nlopt(
            problem=problem,
            x0=x0,
            is_global=self.algo_info.is_global,
            convergence_xtol_rel=self.convergence_xtol_rel,
            convergence_xtol_abs=self.convergence_xtol_abs,
            convergence_ftol_rel=self.convergence_ftol_rel,
            convergence_ftol_abs=self.convergence_ftol_abs,
            stopping_max_eval=self.stopping_maxfun,
            algorithm=algo,
        )
        return res


@mark.minimizer(
    name="nlopt_esch",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_NLOPT_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    needs_bounds=True,
    supports_parallelism=False,
    supports_bounds=True,
    supports_infinite_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class NloptESCH(Algorithm):
    """Minimize a scalar function using the ESCH algorithm.

    ESCH is an evolutionary algorithm for global optimization developed by Carlos
    Henrique da Silva Santos :cite:`DaSilva2010`, :cite:`DaSilva2010a`. It is a
    modified evolution strategy :cite:`Beyer2002` that evolves a population of
    candidate points via mutation and selection; see also :cite:`Vent1975`.

    Use it for global optimization of scalar problems when no derivatives are
    available. Finite bounds on all parameters are required. The algorithm
    supports bound constraints only; nonlinear constraints are not supported.

    Like most evolutionary algorithms, it typically needs many function
    evaluations, and there is no guarantee that the global optimum is found within
    a finite budget.

    """

    convergence_xtol_rel: NonNegativeFloat = CONVERGENCE_XTOL_REL
    """Stop when an optimization step (or an estimate of the optimum) changes every
    parameter by less than this value multiplied by the absolute value of the
    parameter.

    This corresponds to NLopt's ``xtol_rel`` stopping criterion. A value of 0 means
    that this criterion is disabled.

    """

    convergence_xtol_abs: NonNegativeFloat = CONVERGENCE_XTOL_ABS
    """Stop when an optimization step (or an estimate of the optimum) changes every
    parameter by less than this absolute value.

    This corresponds to NLopt's ``xtol_abs`` stopping criterion. A value of 0 means
    that this criterion is disabled, which is the optimagic default.

    """

    convergence_ftol_rel: NonNegativeFloat = CONVERGENCE_FTOL_REL
    """Stop when an optimization step (or an estimate of the optimum) changes the
    objective function value by less than this value multiplied by the absolute
    value of the function value.

    This corresponds to NLopt's ``ftol_rel`` stopping criterion. A value of 0 means
    that this criterion is disabled.

    """

    convergence_ftol_abs: NonNegativeFloat = CONVERGENCE_FTOL_ABS
    """Stop when an optimization step (or an estimate of the optimum) changes the
    objective function value by less than this absolute value.

    This corresponds to NLopt's ``ftol_abs`` stopping criterion. A value of 0 means
    that this criterion is disabled, which is the optimagic default.

    """

    stopping_maxfun: PositiveInt = STOPPING_MAXFUN_GLOBAL
    """Maximum number of objective function evaluations.

    If reached, the optimization stops, but this is not counted as successful
    convergence. This corresponds to NLopt's ``maxeval`` stopping criterion. The
    optimagic default for global optimizers is much lower than for local
    optimizers to keep runtimes manageable.

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        res = _minimize_nlopt(
            problem=problem,
            x0=x0,
            is_global=self.algo_info.is_global,
            convergence_xtol_rel=self.convergence_xtol_rel,
            convergence_xtol_abs=self.convergence_xtol_abs,
            convergence_ftol_rel=self.convergence_ftol_rel,
            convergence_ftol_abs=self.convergence_ftol_abs,
            stopping_max_eval=self.stopping_maxfun,
            algorithm=nlopt.GN_ESCH,
        )

        return res


@mark.minimizer(
    name="nlopt_isres",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_NLOPT_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    needs_bounds=True,
    supports_parallelism=False,
    supports_bounds=True,
    supports_infinite_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=True,
    disable_history=False,
)
@dataclass(frozen=True)
class NloptISRES(Algorithm):
    """Minimize a scalar function using the ISRES algorithm.

    ISRES (Improved Stochastic Ranking Evolution Strategy) is an evolutionary
    algorithm for nonlinearly constrained global optimization
    :cite:`PhilipRunarsson2005`. Its evolution strategy combines a mutation rule
    (with a log-normal step-size update and exponential smoothing) with
    differential variation (a Nelder-Mead-like update rule). For constrained
    problems, individuals are ranked via the stochastic ranking procedure
    described in :cite:`Thomas2000`.

    Use it for global optimization of scalar problems when no derivatives are
    available; it is one of the few global optimizers that supports nonlinear
    equality and inequality constraints. Finite bounds on all parameters are
    required. The algorithm has heuristics to escape local optima, but no proof
    of global convergence is available.

    In the implementation in NLopt :cite:`Johnson2007`, the population size
    defaults to 20 * (number of parameters + 1).

    """

    convergence_xtol_rel: NonNegativeFloat = CONVERGENCE_XTOL_REL
    """Stop when an optimization step (or an estimate of the optimum) changes every
    parameter by less than this value multiplied by the absolute value of the
    parameter.

    This corresponds to NLopt's ``xtol_rel`` stopping criterion. A value of 0 means
    that this criterion is disabled.

    """

    convergence_xtol_abs: NonNegativeFloat = CONVERGENCE_XTOL_ABS
    """Stop when an optimization step (or an estimate of the optimum) changes every
    parameter by less than this absolute value.

    This corresponds to NLopt's ``xtol_abs`` stopping criterion. A value of 0 means
    that this criterion is disabled, which is the optimagic default.

    """

    convergence_ftol_rel: NonNegativeFloat = CONVERGENCE_FTOL_REL
    """Stop when an optimization step (or an estimate of the optimum) changes the
    objective function value by less than this value multiplied by the absolute
    value of the function value.

    This corresponds to NLopt's ``ftol_rel`` stopping criterion. A value of 0 means
    that this criterion is disabled.

    """

    convergence_ftol_abs: NonNegativeFloat = CONVERGENCE_FTOL_ABS
    """Stop when an optimization step (or an estimate of the optimum) changes the
    objective function value by less than this absolute value.

    This corresponds to NLopt's ``ftol_abs`` stopping criterion. A value of 0 means
    that this criterion is disabled, which is the optimagic default.

    """

    stopping_maxfun: PositiveInt = STOPPING_MAXFUN_GLOBAL
    """Maximum number of objective function evaluations.

    If reached, the optimization stops, but this is not counted as successful
    convergence. This corresponds to NLopt's ``maxeval`` stopping criterion. The
    optimagic default for global optimizers is much lower than for local
    optimizers to keep runtimes manageable.

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        res = _minimize_nlopt(
            problem=problem,
            x0=x0,
            is_global=self.algo_info.is_global,
            convergence_xtol_rel=self.convergence_xtol_rel,
            convergence_xtol_abs=self.convergence_xtol_abs,
            convergence_ftol_rel=self.convergence_ftol_rel,
            convergence_ftol_abs=self.convergence_ftol_abs,
            stopping_max_eval=self.stopping_maxfun,
            algorithm=nlopt.GN_ISRES,
        )

        return res


@mark.minimizer(
    name="nlopt_crs2_lm",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_NLOPT_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    needs_bounds=True,
    supports_parallelism=False,
    supports_bounds=True,
    supports_infinite_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class NloptCRS2LM(Algorithm):
    """Minimize a scalar function using the CRS2_LM algorithm.

    CRS2 (Controlled Random Search 2) with local mutation is a derivative-free
    global optimizer. The CRS class of algorithms, originally described in
    :cite:`Price1978` and :cite:`Price1983`, starts with a random population of
    points and evolves these points by heuristic rules; the evolution somewhat
    resembles a randomized Nelder-Mead algorithm. The implementation in NLopt
    :cite:`Johnson2007` uses the CRS2 variant with the local mutation modification
    described in :cite:`Kaelo2006`.

    Use it for global optimization of scalar problems when no derivatives are
    available. Finite bounds on all parameters are required. Like all stochastic
    global optimizers, it typically needs many function evaluations and offers no
    guarantee of finding the global optimum.

    """

    convergence_xtol_rel: NonNegativeFloat = CONVERGENCE_XTOL_REL
    """Stop when an optimization step (or an estimate of the optimum) changes every
    parameter by less than this value multiplied by the absolute value of the
    parameter.

    This corresponds to NLopt's ``xtol_rel`` stopping criterion. A value of 0 means
    that this criterion is disabled.

    """

    convergence_xtol_abs: NonNegativeFloat = CONVERGENCE_XTOL_ABS
    """Stop when an optimization step (or an estimate of the optimum) changes every
    parameter by less than this absolute value.

    This corresponds to NLopt's ``xtol_abs`` stopping criterion. A value of 0 means
    that this criterion is disabled, which is the optimagic default.

    """

    convergence_ftol_rel: NonNegativeFloat = CONVERGENCE_FTOL_REL
    """Stop when an optimization step (or an estimate of the optimum) changes the
    objective function value by less than this value multiplied by the absolute
    value of the function value.

    This corresponds to NLopt's ``ftol_rel`` stopping criterion. A value of 0 means
    that this criterion is disabled.

    """

    convergence_ftol_abs: NonNegativeFloat = CONVERGENCE_FTOL_ABS
    """Stop when an optimization step (or an estimate of the optimum) changes the
    objective function value by less than this absolute value.

    This corresponds to NLopt's ``ftol_abs`` stopping criterion. A value of 0 means
    that this criterion is disabled, which is the optimagic default.

    """

    stopping_maxfun: PositiveInt = STOPPING_MAXFUN_GLOBAL
    """Maximum number of objective function evaluations.

    If reached, the optimization stops, but this is not counted as successful
    convergence. This corresponds to NLopt's ``maxeval`` stopping criterion. The
    optimagic default for global optimizers is much lower than for local
    optimizers to keep runtimes manageable.

    """

    population_size: PositiveInt | None = None
    """Size of the random population of points that is evolved by the algorithm.

    If None, the population size is set to 10 * (number of parameters + 1), which
    is the default of the NLopt implementation.

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if self.population_size is None:
            population_size = 10 * (len(x0) + 1)
        else:
            population_size = self.population_size
        res = _minimize_nlopt(
            problem=problem,
            x0=x0,
            is_global=self.algo_info.is_global,
            convergence_xtol_rel=self.convergence_xtol_rel,
            convergence_xtol_abs=self.convergence_xtol_abs,
            convergence_ftol_rel=self.convergence_ftol_rel,
            convergence_ftol_abs=self.convergence_ftol_abs,
            stopping_max_eval=self.stopping_maxfun,
            algorithm=nlopt.GN_CRS2_LM,
            population_size=population_size,
        )

        return res


def _minimize_nlopt(
    problem,
    x0,
    algorithm,
    is_global,
    *,
    convergence_xtol_rel=None,
    convergence_xtol_abs=None,
    convergence_ftol_rel=None,
    convergence_ftol_abs=None,
    stopping_max_eval=None,
    population_size=None,
    nonlinear_constraints=None,
):
    """Run actual nlopt optimization argument, set relevant attributes."""

    def func(x, grad):
        if grad.size > 0:
            fun, jac = problem.fun_and_jac(x)
            grad[:] = jac
        else:
            fun = problem.fun(x)
        return fun

    if nonlinear_constraints is None:
        nonlinear_constraints = problem.nonlinear_constraints
    opt = nlopt.opt(algorithm, x0.shape[0])
    if convergence_ftol_rel is not None:
        opt.set_ftol_rel(convergence_ftol_rel)
    if convergence_ftol_abs is not None:
        opt.set_ftol_abs(convergence_ftol_abs)
    if convergence_xtol_rel is not None:
        opt.set_xtol_rel(convergence_xtol_rel)
    if convergence_xtol_abs is not None:
        opt.set_xtol_abs(convergence_xtol_abs)
    if problem.bounds.lower is not None:
        opt.set_lower_bounds(problem.bounds.lower)
    if problem.bounds.upper is not None:
        opt.set_upper_bounds(problem.bounds.upper)
    if stopping_max_eval is not None:
        opt.set_maxeval(stopping_max_eval)
    if population_size is not None:
        opt.set_population(population_size)
    if nonlinear_constraints:
        for constr in _get_nlopt_constraints(nonlinear_constraints, filter_type="eq"):
            opt.add_equality_mconstraint(constr["fun"], constr["tol"])
        for constr in _get_nlopt_constraints(nonlinear_constraints, filter_type="ineq"):
            opt.add_inequality_mconstraint(constr["fun"], constr["tol"])
    opt.set_min_objective(func)
    solution_x = opt.optimize(x0)
    return _process_nlopt_results(opt, solution_x, is_global)


def _process_nlopt_results(nlopt_obj, solution_x, is_global):
    messages = {
        1: "Convergence achieved ",
        2: (
            "Optimizer stopped because maximum value of criterion function was reached"
        ),
        3: (
            "Optimizer stopped because convergence_ftol_rel or "
            "convergence_ftol_abs was reached"
        ),
        4: (
            "Optimizer stopped because convergence_xtol_rel or "
            "convergence_xtol_abs was reached"
        ),
        5: "Optimizer stopped because max_criterion_evaluations was reached",
        6: "Optimizer stopped because max running time was reached",
        -1: "Optimizer failed",
        -2: "Invalid arguments were passed",
        -3: "Memory error",
        -4: "Halted because roundoff errors limited progress",
        -5: "Halted because of user specified forced stop",
    }
    success = nlopt_obj.last_optimize_result() in [1, 2, 3, 4]
    if is_global and not success:
        success = None
    processed = InternalOptimizeResult(
        x=solution_x,
        fun=nlopt_obj.last_optimum_value(),
        n_fun_evals=nlopt_obj.get_numevals(),
        success=success,
        message=messages[nlopt_obj.last_optimize_result()],
    )

    return processed


def _get_nlopt_constraints(constraints, filter_type):
    """Transform internal nonlinear constraints to NLOPT readable format."""
    filtered = [c for c in constraints if c["type"] == filter_type]
    nlopt_constraints = [_internal_to_nlopt_constaint(c) for c in filtered]
    return nlopt_constraints


def _internal_to_nlopt_constaint(c):
    """Sign flip description:

    In optimagic, inequality constraints are internally defined as g(x) >= 0. NLOPT uses
    h(x) <= 0, which is why we need to flip the sign.

    """
    tol = c["tol"]
    if np.isscalar(tol):
        tol = np.tile(tol, c["n_constr"])

    def _constraint(result, x, grad):
        result[:] = -c["fun"](x)  # see docstring for sign flip
        if grad.size > 0:
            grad[:] = -c["jac"](x)  # see docstring for sign flip

    new_constr = {
        "fun": _constraint,
        "tol": tol,
    }
    return new_constr
