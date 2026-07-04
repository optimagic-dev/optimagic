"""Implement scipy algorithms.

The following ``scipy`` algorithms are not supported because they
require the specification of the Hessian:

- dogleg
- trust-ncg
- trust-exact
- trust-krylov

The following arguments are not supported as part of ``algo_options``:

- ``disp``
    If set to True would print a convergence message.
    In optimagic it's always set to its default False.
    Refer to optimagic's result dictionary's "success" entry for the convergence
    message.
- ``return_all``
    If set to True, a list of the best solution at each iteration is returned.
    In optimagic it's always set to its default False.
- ``tol``
    This argument of minimize (not an options key) is passed as different types of
    tolerance (gradient, parameter or criterion, as well as relative or absolute)
    depending on the selected algorithm. We require the user to explicitely input
    the tolerance criteria or use our defaults instead.
- ``args``
    This argument of minimize (not an options key) is partialed into the function
    for the user. Specify ``criterion_kwargs`` in ``maximize`` or ``minimize`` to
    achieve the same behavior.
- ``callback``
    This argument would be called after each iteration and the algorithm would
    terminate if it returned True.

"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any, Callable, List, Literal, SupportsInt, Tuple

import numpy as np
import scipy
import scipy.optimize
from numpy.typing import NDArray
from scipy.optimize import Bounds as ScipyBounds
from scipy.optimize import NonlinearConstraint
from scipy.optimize import OptimizeResult as ScipyOptimizeResult

from optimagic import mark
from optimagic.batch_evaluators import process_batch_evaluator
from optimagic.optimization.algo_options import (
    CONVERGENCE_FTOL_ABS,
    CONVERGENCE_FTOL_REL,
    CONVERGENCE_GTOL_ABS,
    CONVERGENCE_GTOL_REL,
    CONVERGENCE_SECOND_BEST_FTOL_ABS,
    CONVERGENCE_SECOND_BEST_XTOL_ABS,
    CONVERGENCE_XTOL_ABS,
    CONVERGENCE_XTOL_REL,
    LIMITED_MEMORY_STORAGE_LENGTH,
    MAX_LINE_SEARCH_STEPS,
    STOPPING_MAXFUN,
    STOPPING_MAXFUN_GLOBAL,
    STOPPING_MAXITER,
)
from optimagic.optimization.algorithm import Algorithm, InternalOptimizeResult
from optimagic.optimization.internal_optimization_problem import (
    InternalBounds,
    InternalOptimizationProblem,
)
from optimagic.parameters.nonlinear_constraints import (
    equality_as_inequality_constraints,
    vector_as_list_of_scalar_constraints,
)
from optimagic.typing import (
    AggregationLevel,
    BatchEvaluator,
    BatchEvaluatorLiteral,
    NegativeFloat,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
)
from optimagic.utilities import calculate_trustregion_initial_radius


@mark.minimizer(
    name="scipy_lbfgsb",
    solver_type=AggregationLevel.SCALAR,
    is_available=True,
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
class ScipyLBFGSB(Algorithm):
    """Minimize a scalar differentiable function using the L-BFGS-B algorithm.

    The optimizer is taken from scipy, which calls the Fortran code written by the
    original authors of the algorithm. The Fortran code includes the corrections
    and improvements that were introduced in a follow up paper.

    lbfgsb is a limited memory version of the original bfgs algorithm, that deals with
    lower and upper bounds via an active set approach.

    The lbfgsb algorithm is well suited for differentiable scalar optimization problems
    with up to several hundred parameters.

    It is a quasi-newton line search algorithm. At each trial point it evaluates the
    criterion function and its gradient to find a search direction. It then approximates
    the hessian using the stored history of gradients and uses the hessian to calculate
    a candidate step size. Then it uses a gradient based line search algorithm to
    determine the actual step length. Since the algorithm always evaluates the gradient
    and criterion function jointly, the user should provide a ``fun_and_jac`` function
    that exploits the synergies in the calculation of criterion and gradient.

    The lbfgsb algorithm is almost perfectly scale invariant. Thus, it is not necessary
    to scale the parameters.

    """

    convergence_ftol_rel: NonNegativeFloat = CONVERGENCE_FTOL_REL
    r"""Converge if the relative change in the objective function is less than this
    value. More formally, this is expressed as.

    .. math::

        \frac{f^k - f^{k+1}}{\max\{{|f^k|, |f^{k+1}|, 1}\}} \leq
        \textsf{convergence_ftol_rel}.

    """

    convergence_gtol_abs: NonNegativeFloat = CONVERGENCE_GTOL_ABS
    """Converge if the absolute values in the gradient of the objective function are
    less than this value."""

    stopping_maxfun: PositiveInt = STOPPING_MAXFUN
    """Maximum number of function evaluations."""

    stopping_maxiter: PositiveInt = STOPPING_MAXITER
    """Maximum number of iterations."""

    limited_memory_storage_length: PositiveInt = LIMITED_MEMORY_STORAGE_LENGTH
    """The maximum number of variable metric corrections used to define the limited
    memory matrix. This is the 'maxcor' parameter in the SciPy documentation.

    The default value is taken from SciPy's L-BFGS-B implementation. Larger values use
    more memory but may converge faster for some problems.

    """

    max_line_search_steps: PositiveInt = MAX_LINE_SEARCH_STEPS
    """The maximum number of line search steps. This is the 'maxls' parameter in the
    SciPy documentation.

    The default value is taken from SciPy's L-BFGS-B implementation.

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        options = {
            "maxcor": self.limited_memory_storage_length,
            "ftol": self.convergence_ftol_rel,
            "gtol": self.convergence_gtol_abs,
            "maxfun": self.stopping_maxfun,
            "maxiter": self.stopping_maxiter,
            "maxls": self.max_line_search_steps,
        }
        raw_res = scipy.optimize.minimize(
            fun=problem.fun_and_jac,
            x0=x0,
            method="L-BFGS-B",
            jac=True,
            bounds=_get_scipy_bounds(problem.bounds),
            options=options,
        )
        res = process_scipy_result(raw_res)
        return res


@mark.minimizer(
    name="scipy_slsqp",
    solver_type=AggregationLevel.SCALAR,
    is_available=True,
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
class ScipySLSQP(Algorithm):
    """Minimize a scalar function of one or more variables using the SLSQP algorithm.

    SLSQP stands for Sequential Least Squares Programming. It is a sequential
    quadratic programming (SQP) line search algorithm that solves a quadratic
    approximation of the problem in each iteration. It uses first derivatives of
    the objective function and the constraints.

    SLSQP is well suited for continuously differentiable scalar optimization
    problems with up to several hundred parameters. It supports bounds as well as
    nonlinear equality and inequality constraints.

    The optimizer is taken from SciPy, which wraps the SLSQP optimization
    subroutine originally implemented by :cite:`Kraft1988`.

    """

    convergence_ftol_abs: NonNegativeFloat = CONVERGENCE_SECOND_BEST_FTOL_ABS
    """Absolute precision goal for the value of the objective function in the
    stopping criterion.

    optimagic's default is 1e-8, which is stricter than SciPy's default of 1e-6.

    """

    stopping_maxiter: PositiveInt = STOPPING_MAXITER
    """If the maximum number of iterations is reached, the optimization stops, but we
    do not count this as convergence."""

    display: bool = False
    """Set to True to print convergence messages.

    This is the 'disp' parameter in the SciPy documentation.

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        options = {
            "maxiter": self.stopping_maxiter,
            "ftol": self.convergence_ftol_abs,
            "disp": self.display,
        }
        raw_res = scipy.optimize.minimize(
            fun=problem.fun_and_jac,
            x0=x0,
            method="SLSQP",
            jac=True,
            bounds=_get_scipy_bounds(problem.bounds),
            constraints=problem.nonlinear_constraints,
            options=options,
        )
        res = process_scipy_result(raw_res)
        return res


@mark.minimizer(
    name="scipy_neldermead",
    solver_type=AggregationLevel.SCALAR,
    is_available=True,
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
class ScipyNelderMead(Algorithm):
    """Minimize a scalar function using the Nelder-Mead simplex algorithm.

    The Nelder-Mead algorithm (:cite:`Nelder1965`) is a direct search method (based
    on function comparison) and is often applied to nonlinear optimization problems
    for which derivatives are not known. It maintains a simplex of ``n + 1`` points
    (where ``n`` is the number of parameters) that is updated via reflection,
    expansion, contraction and shrinkage operations. Bounds are supported.

    Unlike most modern optimization methods, the Nelder-Mead heuristic can converge
    to a non-stationary point, unless the problem satisfies stronger conditions than
    are necessary for modern methods.

    Nelder-Mead is never the best algorithm to solve a problem but rarely the worst.
    Its popularity is likely due to historic reasons and much larger than its
    properties warrant.

    .. note::
        The SciPy argument ``initial_simplex`` is not supported by optimagic because
        it is not compatible with optimagic's handling of constraints.

    """

    stopping_maxiter: PositiveInt = STOPPING_MAXITER
    """If the maximum number of iterations is reached, the optimization stops, but we
    do not count this as convergence."""

    stopping_maxfun: PositiveInt = STOPPING_MAXFUN
    """If the maximum number of function evaluations is reached, the optimization
    stops, but we do not count this as convergence."""

    convergence_ftol_abs: NonNegativeFloat = CONVERGENCE_SECOND_BEST_FTOL_ABS
    """Absolute difference in the criterion value between iterations that is tolerated
    to declare convergence.

    As no relative tolerances can be passed to Nelder-Mead, optimagic sets a non-zero
    default of 1e-8, which is stricter than SciPy's default of 1e-4.

    """

    convergence_xtol_abs: NonNegativeFloat = CONVERGENCE_SECOND_BEST_XTOL_ABS
    """Absolute difference in parameters between iterations that is tolerated to
    declare convergence.

    As no relative tolerances can be passed to Nelder-Mead, optimagic sets a non-zero
    default of 1e-8, which is stricter than SciPy's default of 1e-4.

    """

    adaptive: bool = False
    """Adapt the algorithm parameters (reflection, expansion, contraction and
    shrinkage) to the dimensionality of the problem.

    This is useful for high-dimensional minimization (:cite:`Gao2012`). The default
    value False is taken from SciPy.

    """

    display: bool = False
    """Set to True to print convergence messages.

    This is the 'disp' parameter in the SciPy documentation.

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        options = {
            "maxiter": self.stopping_maxiter,
            "maxfev": self.stopping_maxfun,
            "xatol": self.convergence_xtol_abs,
            "fatol": self.convergence_ftol_abs,
            # TODO: Benchmark if adaptive = True works better
            "adaptive": self.adaptive,
            "disp": self.display,
        }
        raw_res = scipy.optimize.minimize(
            fun=problem.fun,
            x0=x0,
            bounds=_get_scipy_bounds(problem.bounds),
            method="Nelder-Mead",
            options=options,
        )
        res = process_scipy_result(raw_res)
        return res


@mark.minimizer(
    name="scipy_powell",
    solver_type=AggregationLevel.SCALAR,
    is_available=True,
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
class ScipyPowell(Algorithm):
    """Minimize a scalar function using the modified Powell method.

    Powell's method (:cite:`Powell1964`) is a conjugate direction method that
    minimizes the function by a sequence of one-dimensional searches: In each
    iteration it performs a bidirectional line search along each vector of a
    direction set and updates the direction set such that the search directions
    become approximately conjugate.

    The algorithm is derivative-free: In contrast to gradient-based algorithms like
    ``scipy_bfgs`` or ``scipy_conjugate_gradient``, the criterion function need not
    be differentiable and no derivatives are evaluated. Bounds are supported.

    .. warning::
        In our benchmark using a quadratic objective function, the Powell algorithm
        did not find the optimum very precisely (less than 4 decimal places). If you
        require high precision, you should refine an optimum found with Powell with
        another local optimizer.

    .. note::
        The SciPy argument ``direc``, which is the initial set of direction vectors,
        is not supported by optimagic because it is incompatible with how optimagic
        handles constraints.

    """

    convergence_xtol_rel: NonNegativeFloat = CONVERGENCE_XTOL_REL
    """Stop when the relative movement between parameter vectors is smaller than
    this."""

    convergence_ftol_rel: NonNegativeFloat = CONVERGENCE_FTOL_REL
    r"""Stop when the relative improvement between two iterations is smaller than
    this. More formally, this is expressed as

    .. math::

        \frac{f^k - f^{k+1}}{\max\{{|f^k|, |f^{k+1}|, 1}\}} \leq
        \textsf{convergence_ftol_rel}.

    """

    stopping_maxfun: PositiveInt = STOPPING_MAXFUN
    """If the maximum number of function evaluations is reached, the optimization
    stops, but we do not count this as convergence."""

    stopping_maxiter: PositiveInt = STOPPING_MAXITER
    """If the maximum number of iterations is reached, the optimization stops, but we
    do not count this as convergence."""

    display: bool = False
    """Set to True to print convergence messages.

    This is the 'disp' parameter in the SciPy documentation.

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        options = {
            "xtol": self.convergence_xtol_rel,
            "ftol": self.convergence_ftol_rel,
            "maxfev": self.stopping_maxfun,
            "maxiter": self.stopping_maxiter,
            "disp": self.display,
        }
        raw_res = scipy.optimize.minimize(
            fun=problem.fun,
            x0=x0,
            method="Powell",
            bounds=_get_scipy_bounds(problem.bounds),
            options=options,
        )
        res = process_scipy_result(raw_res)
        return res


@mark.minimizer(
    name="scipy_bfgs",
    solver_type=AggregationLevel.SCALAR,
    is_available=True,
    is_global=False,
    needs_jac=True,
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
class ScipyBFGS(Algorithm):
    """Minimize a scalar differentiable function using the BFGS algorithm.

    BFGS stands for Broyden-Fletcher-Goldfarb-Shanno algorithm (see
    :cite:`Nocedal2006`). It is a quasi-Newton line search algorithm for
    unconstrained nonlinear optimization problems. At each iteration it computes a
    search direction from the gradient and an approximation of the inverse Hessian
    that is built up from the history of gradient evaluations, and then performs a
    line search that satisfies the strong Wolfe conditions.

    BFGS is not guaranteed to converge unless the function has a quadratic Taylor
    expansion near an optimum. However, BFGS can have acceptable performance even
    for non-smooth optimization instances.

    Bounds and other constraints are not supported. For problems with bound
    constraints or with a very large number of parameters, consider the
    limited-memory variant ``scipy_lbfgsb`` instead.

    """

    convergence_gtol_abs: NonNegativeFloat = CONVERGENCE_GTOL_ABS
    """Stop if the norm of the gradient (as defined by the ``norm`` option) is
    smaller than this value.

    With the default infinity norm, this means that the optimization stops when all
    elements of the gradient are smaller than this value. The default value is
    taken from SciPy.

    """

    stopping_maxiter: PositiveInt = STOPPING_MAXITER
    """If the maximum number of iterations is reached, the optimization stops, but we
    do not count this as convergence."""

    norm: NonNegativeFloat = np.inf
    """Order of the vector norm that is used to calculate the gradient's "score" that
    is compared to the gradient tolerance to determine convergence.

    The default is infinity, which means that the largest entry of the gradient
    vector is compared to the gradient tolerance.

    """

    convergence_xtol_rel: NonNegativeFloat = CONVERGENCE_XTOL_REL
    """Relative tolerance in the parameters. Terminate successfully if the step size
    is less than ``xk * convergence_xtol_rel`` where ``xk`` is the current parameter
    vector. This is the 'xrtol' parameter in the SciPy documentation.

    optimagic's default is 1e-5, while SciPy's default is 0.

    """

    display: bool = False
    """Set to True to print convergence messages.

    This is the 'disp' parameter in the SciPy documentation.

    """

    armijo_condition: NonNegativeFloat = 1e-4
    r"""Parameter for the Armijo condition rule of the line search. This is the 'c1'
    parameter in the SciPy documentation. It ensures

    .. math::

        f(x_k + \alpha p_k) \leq f(x_k) + \textsf{armijo_condition} \cdot \alpha
        \nabla f(x_k)^\top p_k,

    so each step yields at least a fraction ``armijo_condition`` of the predicted
    decrease. Smaller values allow more aggressive steps, larger values enforce more
    conservative ones. The default value is taken from SciPy.

    """

    curvature_condition: NonNegativeFloat = 0.9
    r"""Parameter for the curvature condition rule of the line search. This is the
    'c2' parameter in the SciPy documentation. It ensures

    .. math::

        \nabla f(x_k + \alpha p_k)^\top p_k \geq \textsf{curvature_condition} \cdot
        \nabla f(x_k)^\top p_k,

    so the slope at the new point is not too negative. Smaller values enforce a
    stricter curvature reduction (smaller steps), larger values are looser (bigger
    steps). SciPy requires ``0 < armijo_condition < curvature_condition < 1``. The
    default value is taken from SciPy.

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        options = {
            "gtol": self.convergence_gtol_abs,
            "maxiter": self.stopping_maxiter,
            "norm": self.norm,
            "xrtol": self.convergence_xtol_rel,
            "disp": self.display,
            "c1": self.armijo_condition,
            "c2": self.curvature_condition,
        }
        raw_res = scipy.optimize.minimize(
            fun=problem.fun_and_jac, x0=x0, method="BFGS", jac=True, options=options
        )
        res = process_scipy_result(raw_res)
        return res


@mark.minimizer(
    name="scipy_conjugate_gradient",
    solver_type=AggregationLevel.SCALAR,
    is_available=True,
    is_global=False,
    needs_jac=True,
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
class ScipyConjugateGradient(Algorithm):
    """Minimize a scalar function using a nonlinear conjugate gradient algorithm.

    The conjugate gradient method finds local optima of a differentiable function
    using only first derivatives; no Hessian information is required, which keeps
    the memory requirements low. SciPy's implementation is based on the
    Polak-Ribiere variant of the algorithm, described in :cite:`Nocedal2006`,
    pp. 120-122. Bounds and other constraints are not supported.

    Conjugate gradient methods tend to work better when:

    - the criterion has a unique global minimizing point, and no local minima or
      other stationary points,
    - the criterion is, at least locally, reasonably well approximated by a
      quadratic function,
    - the criterion is continuous and has a continuous gradient,
    - the gradient is not too large, e.g., has a norm less than 1000,
    - the initial guess is reasonably close to the criterion's global minimizer.

    """

    convergence_gtol_abs: NonNegativeFloat = CONVERGENCE_GTOL_ABS
    """Stop if the norm of the gradient (as defined by the ``norm`` option) is
    smaller than this value.

    With the default infinity norm, this means that the optimization stops when all
    elements of the gradient are smaller than this value. The default value is
    taken from SciPy.

    """

    stopping_maxiter: PositiveInt = STOPPING_MAXITER
    """If the maximum number of iterations is reached, the optimization stops, but we
    do not count this as convergence."""

    norm: NonNegativeFloat = np.inf
    """Order of the vector norm that is used to calculate the gradient's "score" that
    is compared to the gradient tolerance to determine convergence.

    The default is infinity, which means that the largest entry of the gradient
    vector is compared to the gradient tolerance.

    """

    display: bool = False
    """Set to True to print convergence messages.

    This is the 'disp' parameter in the SciPy documentation.

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        options = {
            "gtol": self.convergence_gtol_abs,
            "maxiter": self.stopping_maxiter,
            "norm": self.norm,
            "disp": self.display,
        }
        raw_res = scipy.optimize.minimize(
            fun=problem.fun_and_jac, x0=x0, method="CG", jac=True, options=options
        )
        res = process_scipy_result(raw_res)
        return res


@mark.minimizer(
    name="scipy_newton_cg",
    solver_type=AggregationLevel.SCALAR,
    is_available=True,
    is_global=False,
    needs_jac=True,
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
class ScipyNewtonCG(Algorithm):
    """Minimize a scalar function using Newton's conjugate gradient algorithm.

    Newton's conjugate gradient algorithm computes the search direction in each
    iteration by approximately solving the Newton equations with a conjugate
    gradient method that only requires an approximation of the Hessian. It is
    practical for both small and large problems (see :cite:`Nocedal2006`, p. 140).

    .. warning::
        In our benchmark using a quadratic objective function, the newton_cg
        algorithm did not find the optimum very precisely (less than 4 decimal
        places). If you require high precision, you should refine an optimum found
        with newton_cg with another local optimizer.

    Newton-CG methods are also called truncated Newton methods.
    ``scipy_newton_cg`` differs from ``scipy_truncated_newton`` because

    - ``scipy_newton_cg``'s algorithm is written purely in Python using NumPy
      and SciPy, while ``scipy_truncated_newton``'s algorithm calls a C function,
    - ``scipy_newton_cg``'s algorithm is only for unconstrained minimization,
      while ``scipy_truncated_newton``'s algorithm supports bounds.

    Conjugate gradient methods tend to work better when:

    - the criterion has a unique global minimizing point, and no local minima or
      other stationary points,
    - the criterion is, at least locally, reasonably well approximated by a
      quadratic function,
    - the criterion is continuous and has a continuous gradient,
    - the gradient is not too large, e.g., has a norm less than 1000,
    - the initial guess is reasonably close to the criterion's global minimizer.

    """

    convergence_xtol_rel: NonNegativeFloat = CONVERGENCE_XTOL_REL
    """Stop when the relative movement between parameter vectors is smaller than
    this.

    Newton-CG uses the average relative change in the parameters for determining
    convergence.

    """

    stopping_maxiter: PositiveInt = STOPPING_MAXITER
    """If the maximum number of iterations is reached, the optimization stops, but we
    do not count this as convergence."""

    display: bool = False
    """Set to True to print convergence messages.

    This is the 'disp' parameter in the SciPy documentation.

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        options = {
            "xtol": self.convergence_xtol_rel,
            "maxiter": self.stopping_maxiter,
            "disp": self.display,
        }
        raw_res = scipy.optimize.minimize(
            fun=problem.fun_and_jac,
            x0=x0,
            method="Newton-CG",
            jac=True,
            options=options,
        )
        res = process_scipy_result(raw_res)
        return res


@mark.minimizer(
    name="scipy_cobyla",
    solver_type=AggregationLevel.SCALAR,
    is_available=True,
    is_global=False,
    needs_jac=False,
    needs_hess=False,
    needs_bounds=False,
    supports_parallelism=False,
    supports_bounds=False,
    supports_infinite_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=True,
    disable_history=False,
)
@dataclass(frozen=True)
class ScipyCOBYLA(Algorithm):
    """Minimize a scalar function of one or more variables using the COBYLA
    algorithm.

    COBYLA stands for Constrained Optimization By Linear Approximation. It is a
    derivative-free trust-region method that models the objective and constraint
    functions by linear interpolation at the vertices of a simplex. For more
    information on COBYLA see :cite:`Powell1994`, :cite:`Powell1998` and
    :cite:`Powell2007`.

    COBYLA supports nonlinear constraints. Natively, it can only handle inequality
    constraints; optimagic converts each equality constraint into two inequality
    constraints before passing the problem to SciPy. Bounds are not supported by
    this optimizer in optimagic.

    SciPy's implementation wraps Powell's original Fortran implementation of the
    algorithm.

    """

    convergence_xtol_rel: NonNegativeFloat = CONVERGENCE_XTOL_REL
    """Stop when the relative movement between parameter vectors is smaller than
    this.

    In case of COBYLA this is a lower bound on the size of the trust region and can
    be seen as the required accuracy in the variables, but this accuracy is not
    guaranteed.

    """

    stopping_maxiter: PositiveInt = STOPPING_MAXITER
    """If the maximum number of iterations is reached, the optimization stops, but we
    do not count this as convergence."""

    trustregion_initial_radius: PositiveFloat | None = None
    """Initial value of the trust region radius.

    Since a linear approximation is likely only good near the current simplex, the
    linear program is given the further requirement that the solution, which will
    become the next evaluation point, must be within a radius RHO_j from x_j. RHO_j
    only decreases, never increases. The initial RHO_j is the
    ``trustregion_initial_radius``. In this way COBYLA's iterations behave like a
    trust region algorithm. If None (the default), the radius is determined from
    the magnitude of the start parameters.

    """

    display: bool = False
    """Set to True to print convergence messages.

    This is the 'disp' parameter in the SciPy documentation.

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        # TODO: Maybe we should leave the radius at their default
        if self.trustregion_initial_radius is None:
            radius = calculate_trustregion_initial_radius(x0)
        else:
            radius = self.trustregion_initial_radius

        options = {
            "maxiter": self.stopping_maxiter,
            "rhobeg": radius,
            "disp": self.display,
        }

        # cannot handle equality constraints
        nonlinear_constraints = equality_as_inequality_constraints(
            problem.nonlinear_constraints
        )

        raw_res = scipy.optimize.minimize(
            fun=problem.fun,
            x0=x0,
            method="COBYLA",
            constraints=nonlinear_constraints,
            options=options,
            tol=self.convergence_xtol_rel,
        )
        res = process_scipy_result(raw_res)
        return res


@mark.minimizer(
    name="scipy_ls_trf",
    solver_type=AggregationLevel.LEAST_SQUARES,
    is_available=True,
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
class ScipyLSTRF(Algorithm):
    """Minimize a nonlinear least-squares problem using the Trust Region Reflective
    method.

    The Trust Region Reflective (TRF) algorithm (:cite:`Branch1999`,
    :cite:`Coleman1996`) is an interior trust-region method for bound-constrained
    minimization. In each iteration it solves a trust-region subproblem augmented
    by a special diagonal quadratic term, where the shape of the trust region is
    determined by the distance from the bounds and the direction of the gradient.

    The algorithm can only be used for least-squares problems, i.e. the objective
    function must be marked with the ``mark.least_squares`` decorator. It is
    particularly suitable for large problems with bounds and is a generally robust
    method; it is SciPy's default least-squares method. For small problems with
    bounds, ``scipy_ls_dogbox`` can be more efficient, and for small unconstrained
    problems ``scipy_ls_lm`` is usually the best choice.

    """

    convergence_ftol_rel: NonNegativeFloat = CONVERGENCE_FTOL_REL
    """Stop when the relative improvement of the cost function between two iterations
    is below this value.

    More precisely, the optimization stops when ``dF < convergence_ftol_rel * F``
    and there was an adequate agreement between a local quadratic model and the
    true model in the last step.

    """

    convergence_gtol_rel: NonNegativeFloat = CONVERGENCE_GTOL_REL
    """Stop when the uniform norm of the gradient, scaled to account for the presence
    of bounds, is below this value.

    The default value is taken from SciPy.

    """

    stopping_maxfun: PositiveInt = STOPPING_MAXFUN
    """If the maximum number of function evaluations is reached, the optimization
    stops, but we do not count this as convergence."""

    relative_step_size_diff_approx: NonNegativeFloat | None = None
    """Relative step size for SciPy's finite difference approximation of the
    Jacobian. This is the 'diff_step' parameter in the SciPy documentation.

    The actual step is computed as ``x * relative_step_size_diff_approx``. If None
    (default), it is chosen automatically. Note that optimagic always passes a
    Jacobian function (either the user-provided derivative or optimagic's numerical
    derivative) to SciPy, in which case this option has no effect.

    """

    tr_solver: Literal["exact", "lsmr"] | None = None
    """Method for solving the trust-region subproblems.

    - 'exact' is suitable for not very large problems with dense Jacobian matrices.
      The computational complexity per iteration is comparable to a singular value
      decomposition of the Jacobian matrix.
    - 'lsmr' is suitable for problems with sparse and large Jacobian matrices. It
      uses the iterative procedure ``scipy.sparse.linalg.lsmr`` for finding a
      solution of a linear least-squares problem and only requires matrix-vector
      product evaluations.

    If None (default), the solver is chosen based on the type of Jacobian returned
    on the first iteration.

    """

    tr_solver_options: dict[str, Any] | None = None
    """Keyword options passed to the trust-region solver. This is the 'tr_options'
    parameter in the SciPy documentation.

    - If ``tr_solver='exact'``, the options are ignored.
    - If ``tr_solver='lsmr'``, the options are passed to
      ``scipy.sparse.linalg.lsmr``. Additionally, the 'regularize' option (bool,
      default True) adds a regularization term to the normal equation, which
      improves convergence if the Jacobian is rank-deficient.

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if self.tr_solver_options is None:
            tr_solver_options = {}
        else:
            tr_solver_options = self.tr_solver_options

        lower_bounds = -np.inf if problem.bounds.lower is None else problem.bounds.lower
        upper_bounds = np.inf if problem.bounds.upper is None else problem.bounds.upper

        raw_res = scipy.optimize.least_squares(
            fun=problem.fun,
            x0=x0,
            # This optimizer does not work with fun_and_jac
            jac=problem.jac,
            bounds=(lower_bounds, upper_bounds),
            method="trf",
            max_nfev=self.stopping_maxfun,
            ftol=self.convergence_ftol_rel,
            gtol=self.convergence_gtol_rel,
            diff_step=self.relative_step_size_diff_approx,
            tr_solver=self.tr_solver,
            tr_options=tr_solver_options,
        )
        res = process_scipy_result(raw_res)
        return res


@mark.minimizer(
    name="scipy_ls_dogbox",
    solver_type=AggregationLevel.LEAST_SQUARES,
    is_available=True,
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
class ScipyLSDogbox(Algorithm):
    """Minimize a nonlinear least-squares problem using a rectangular trust region
    method.

    The dogbox algorithm (:cite:`Voglis2004`) operates in a trust-region framework,
    but uses rectangular trust regions as opposed to conventional ellipsoids. The
    intersection of the trust region and the bounds is again rectangular, so in
    each iteration a bound-constrained quadratic subproblem is solved approximately
    by Powell's dogleg method.

    The algorithm can only be used for least-squares problems, i.e. the objective
    function must be marked with the ``mark.least_squares`` decorator. The typical
    use case is small problems with bounds, where it often outperforms
    ``scipy_ls_trf``. It is not recommended for problems with a rank-deficient
    Jacobian.

    """

    convergence_ftol_rel: NonNegativeFloat = CONVERGENCE_FTOL_REL
    """Stop when the relative improvement of the cost function between two iterations
    is below this value.

    More precisely, the optimization stops when ``dF < convergence_ftol_rel * F``
    and there was an adequate agreement between a local quadratic model and the
    true model in the last step.

    """

    convergence_gtol_rel: NonNegativeFloat = CONVERGENCE_GTOL_REL
    """Stop when the uniform norm of the gradient with respect to the variables that
    are not at their bounds is below this value.

    The default value is taken from SciPy.

    """

    stopping_maxfun: PositiveInt = STOPPING_MAXFUN
    """If the maximum number of function evaluations is reached, the optimization
    stops, but we do not count this as convergence."""

    relative_step_size_diff_approx: NonNegativeFloat | None = None
    """Relative step size for SciPy's finite difference approximation of the
    Jacobian. This is the 'diff_step' parameter in the SciPy documentation.

    The actual step is computed as ``x * relative_step_size_diff_approx``. If None
    (default), it is chosen automatically. Note that optimagic always passes a
    Jacobian function (either the user-provided derivative or optimagic's numerical
    derivative) to SciPy, in which case this option has no effect.

    """

    tr_solver: Literal["exact", "lsmr"] | None = None
    """Method for solving the trust-region subproblems.

    - 'exact' is suitable for not very large problems with dense Jacobian matrices.
      The computational complexity per iteration is comparable to a singular value
      decomposition of the Jacobian matrix.
    - 'lsmr' is suitable for problems with sparse and large Jacobian matrices. It
      uses the iterative procedure ``scipy.sparse.linalg.lsmr`` for finding a
      solution of a linear least-squares problem and only requires matrix-vector
      product evaluations.

    If None (default), the solver is chosen based on the type of Jacobian returned
    on the first iteration.

    """

    tr_solver_options: dict[str, Any] | None = None
    """Keyword options passed to the trust-region solver. This is the 'tr_options'
    parameter in the SciPy documentation.

    - If ``tr_solver='exact'``, the options are ignored.
    - If ``tr_solver='lsmr'``, the options are passed to
      ``scipy.sparse.linalg.lsmr``.

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if self.tr_solver_options is None:
            tr_solver_options = {}
        else:
            tr_solver_options = self.tr_solver_options

        lower_bounds = -np.inf if problem.bounds.lower is None else problem.bounds.lower
        upper_bounds = np.inf if problem.bounds.upper is None else problem.bounds.upper

        raw_res = scipy.optimize.least_squares(
            fun=problem.fun,
            x0=x0,
            # This optimizer does not work with fun_and_jac
            jac=problem.jac,
            bounds=(lower_bounds, upper_bounds),
            method="dogbox",
            max_nfev=self.stopping_maxfun,
            ftol=self.convergence_ftol_rel,
            gtol=self.convergence_gtol_rel,
            diff_step=self.relative_step_size_diff_approx,
            tr_solver=self.tr_solver,
            tr_options=tr_solver_options,
        )
        res = process_scipy_result(raw_res)
        return res


@mark.minimizer(
    name="scipy_ls_lm",
    solver_type=AggregationLevel.LEAST_SQUARES,
    is_available=True,
    is_global=False,
    needs_jac=True,
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
class ScipyLSLM(Algorithm):
    """Minimize a nonlinear least-squares problem using the Levenberg-Marquardt
    method.

    SciPy's implementation wraps the classic Levenberg-Marquardt algorithm as
    implemented in MINPACK (:cite:`More1978`). It is usually the most efficient
    method for small, unconstrained least-squares problems.

    The algorithm can only be used for least-squares problems, i.e. the objective
    function must be marked with the ``mark.least_squares`` decorator. It does not
    handle bounds or sparse Jacobians and requires that the number of residuals is
    at least as large as the number of parameters. For problems with bounds, use
    ``scipy_ls_trf`` or ``scipy_ls_dogbox`` instead.

    """

    convergence_ftol_rel: NonNegativeFloat = CONVERGENCE_FTOL_REL
    """Stop when the relative improvement of the cost function between two iterations
    is below this value.

    More precisely, the optimization stops when ``dF < convergence_ftol_rel * F``
    and there was an adequate agreement between a local quadratic model and the
    true model in the last step.

    """

    convergence_gtol_rel: NonNegativeFloat = CONVERGENCE_GTOL_REL
    """Stop when the maximum absolute value of the cosine of the angles between the
    columns of the Jacobian and the residual vector is below this value.

    The default value is taken from SciPy.

    """

    stopping_maxfun: PositiveInt = STOPPING_MAXFUN
    """If the maximum number of function evaluations is reached, the optimization
    stops, but we do not count this as convergence."""

    relative_step_size_diff_approx: NonNegativeFloat | None = None
    """Relative step size for SciPy's finite difference approximation of the
    Jacobian. This is the 'diff_step' parameter in the SciPy documentation.

    The actual step is computed as ``x * relative_step_size_diff_approx``. If None
    (default), it is chosen automatically. Note that optimagic always passes a
    Jacobian function (either the user-provided derivative or optimagic's numerical
    derivative) to SciPy, in which case this option has no effect.

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        raw_res = scipy.optimize.least_squares(
            fun=problem.fun,
            x0=x0,
            # This optimizer does not work with fun_and_jac
            jac=problem.jac,
            method="lm",
            max_nfev=self.stopping_maxfun,
            ftol=self.convergence_ftol_rel,
            gtol=self.convergence_gtol_rel,
            diff_step=self.relative_step_size_diff_approx,
        )
        res = process_scipy_result(raw_res)
        return res


@mark.minimizer(
    name="scipy_truncated_newton",
    solver_type=AggregationLevel.SCALAR,
    is_available=True,
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
class ScipyTruncatedNewton(Algorithm):
    """Minimize a scalar function using the truncated Newton (TNC) algorithm.

    A truncated Newton method computes the search direction in each iteration by
    approximately solving the Newton equations with a conjugate gradient iteration
    that is truncated after a limited number of steps (:cite:`Dembo1983`,
    :cite:`Nash1984`). This makes the algorithm suitable for differentiable scalar
    problems with many parameters. Bounds are supported.

    ``scipy_truncated_newton`` differs from ``scipy_newton_cg`` because

    - ``scipy_newton_cg``'s algorithm is written purely in Python using NumPy
      and SciPy, while ``scipy_truncated_newton`` wraps a C implementation,
    - ``scipy_newton_cg``'s algorithm is only for unconstrained minimization,
      while ``scipy_truncated_newton``'s algorithm supports bounds.

    Conjugate gradient methods tend to work better when:

    - the criterion has a unique global minimizing point, and no local minima or
      other stationary points,
    - the criterion is, at least locally, reasonably well approximated by a
      quadratic function,
    - the criterion is continuous and has a continuous gradient,
    - the gradient is not too large, e.g., has a norm less than 1000,
    - the initial guess is reasonably close to the criterion's global minimizer.

    .. note::
        optimagic does not support the SciPy arguments ``scale`` and ``offset``
        because they are not compatible with the way optimagic handles constraints.
        It also does not support ``messg_num``, which is an additional way to
        control the verbosity of the optimizer.

    """

    convergence_ftol_abs: NonNegativeFloat = CONVERGENCE_FTOL_ABS
    """Absolute difference in the criterion value between iterations after scaling
    that is tolerated to declare convergence."""

    convergence_xtol_abs: NonNegativeFloat = CONVERGENCE_XTOL_ABS
    """Absolute difference in parameters between iterations after scaling that is
    tolerated to declare convergence."""

    convergence_gtol_abs: NonNegativeFloat = CONVERGENCE_GTOL_ABS
    """Stop if the value of the projected gradient (after applying x scaling factors)
    is smaller than this value.

    In SciPy, a negative value means that the tolerance is set to
    ``1e-2 * sqrt(finite_difference_precision)``.

    """

    stopping_maxfun: PositiveInt = STOPPING_MAXFUN
    """If the maximum number of function evaluations is reached, the optimization
    stops, but we do not count this as convergence."""

    max_hess_evaluations_per_iteration: int = -1
    """Maximum number of hessian*vector evaluations per main iteration. This is the
    'maxCGit' parameter in the SciPy documentation.

    If ``max_hess_evaluations_per_iteration == 0``, the direction chosen is
    ``-gradient``. If ``max_hess_evaluations_per_iteration < 0``, it is set to
    ``max(1, min(50, n/2))`` where n is the length of the parameter vector. This is
    also the default.

    """

    max_step_for_line_search: NonNegativeFloat = 0
    """Maximum step for the line search. This is the 'stepmx' parameter in the SciPy
    documentation.

    It may be increased during the optimization. If too small, it is set to 10.0.
    The default value is taken from SciPy.

    """

    line_search_severity: float = -1
    """Severity of the line search. This is the 'eta' parameter in the SciPy
    documentation.

    If < 0 or > 1, it is set to 0.25. The default value is taken from SciPy.

    """

    finite_difference_precision: NonNegativeFloat = 0
    """Relative precision for finite difference calculations. This is the 'accuracy'
    parameter in the SciPy documentation.

    If it is smaller than the machine precision, it is set to the square root of
    the machine precision. The default value is taken from SciPy.

    """

    criterion_rescale_factor: float = -1
    """Scaling factor (in log10) used to trigger rescaling of the criterion function.
    This is the 'rescale' parameter in the SciPy documentation.

    If 0, rescale at each iteration. If a large value, never rescale. If < 0, it is
    set to 1.3. The default value is taken from SciPy.

    """

    # TODO: Check type hint for `func_min_estimate`
    func_min_estimate: float = 0
    """Minimum function value estimate. This is the 'minfev' parameter in the SciPy
    documentation.

    The default value of 0 is taken from SciPy.

    """

    display: bool = False
    """Set to True to print convergence messages.

    This is the 'disp' parameter in the SciPy documentation.

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        options = {
            "ftol": self.convergence_ftol_abs,
            "xtol": self.convergence_xtol_abs,
            "gtol": self.convergence_gtol_abs,
            "maxfun": self.stopping_maxfun,
            "maxCGit": self.max_hess_evaluations_per_iteration,
            "stepmx": self.max_step_for_line_search,
            "minfev": self.func_min_estimate,
            "eta": self.line_search_severity,
            "accuracy": self.finite_difference_precision,
            "rescale": self.criterion_rescale_factor,
            "disp": self.display,
        }

        raw_res = scipy.optimize.minimize(
            fun=problem.fun_and_jac,
            x0=x0,
            method="TNC",
            jac=True,
            bounds=_get_scipy_bounds(problem.bounds),
            options=options,
        )
        res = process_scipy_result(raw_res)
        return res


@mark.minimizer(
    name="scipy_trust_constr",
    solver_type=AggregationLevel.SCALAR,
    is_available=True,
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
class ScipyTrustConstr(Algorithm):
    """Minimize a scalar function of one or more variables subject to constraints.

    ``scipy_trust_constr`` is a trust-region algorithm for constrained
    optimization. It is the most versatile constrained minimization algorithm
    implemented in SciPy and the most appropriate for large-scale problems. It
    supports bounds as well as nonlinear constraints. Natively, it can handle both
    equality and inequality constraints; optimagic converts each equality
    constraint into two inequality constraints before passing the problem to SciPy.

    It switches between two implementations depending on the problem definition.
    For equality-constrained problems it is an implementation of the
    Byrd-Omojokun trust-region SQP method described in :cite:`Lalee1998` and in
    :cite:`Conn2000`, p. 549. When inequality constraints are imposed as well, it
    switches to the trust-region interior point method described in
    :cite:`Byrd1999`. This interior point algorithm, in turn, solves inequality
    constraints by introducing slack variables and solving a sequence of
    equality-constrained barrier problems for progressively smaller values of the
    barrier parameter. The previously described equality-constrained SQP method is
    used to solve the subproblems with increasing levels of accuracy as the iterate
    gets closer to a solution.

    It approximates the Hessian using the Broyden-Fletcher-Goldfarb-Shanno (BFGS)
    Hessian update strategy, so no Hessian needs to be provided.

    .. warning::
        In our benchmark using a quadratic objective function, the trust_constr
        algorithm did not find the optimum very precisely (less than 4 decimal
        places). If you require high precision, you should refine an optimum found
        with trust_constr with another local optimizer.

    """

    # TODO: Check if can be set to CONVERGENCE_GTOL_ABS
    convergence_gtol_abs: NonNegativeFloat = 1e-08
    """Tolerance for termination by the norm of the Lagrangian gradient.

    The algorithm will terminate when both the infinity norm (i.e., max abs value)
    of the Lagrangian gradient and the constraint violation are smaller than this
    value. The default value of 1e-8 is taken from SciPy.

    """

    convergence_xtol_rel: NonNegativeFloat = CONVERGENCE_XTOL_REL
    """Tolerance for termination by the change of the independent variable.

    The algorithm will terminate when the radius of the trust region used in the
    algorithm is smaller than this value. optimagic's default is 1e-5, while
    SciPy's default is 1e-8.

    """

    stopping_maxiter: PositiveInt = STOPPING_MAXITER
    """If the maximum number of iterations is reached, the optimization stops, but we
    do not count this as convergence."""

    trustregion_initial_radius: PositiveFloat | None = None
    """Initial value of the trust region radius.

    The trust radius gives the maximum distance between solution points in
    consecutive iterations. It reflects the trust the algorithm puts in the local
    approximation of the optimization problem. For an accurate local approximation
    the trust region should be large and for an approximation valid only close to
    the current point it should be a small one. The trust radius is automatically
    updated throughout the optimization process, with ``trustregion_initial_radius``
    being its initial value. If None (the default), the initial radius is
    determined from the magnitude of the start parameters.

    """

    display: bool = False
    """Set to True to print convergence messages.

    This is the 'disp' parameter in the SciPy documentation.

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if self.trustregion_initial_radius is None:
            trustregion_initial_radius = calculate_trustregion_initial_radius(x0)
        else:
            trustregion_initial_radius = self.trustregion_initial_radius

        options = {
            "gtol": self.convergence_gtol_abs,
            "maxiter": self.stopping_maxiter,
            "xtol": self.convergence_xtol_rel,
            "initial_tr_radius": trustregion_initial_radius,
            "disp": self.display,
        }

        # cannot handle equality constraints
        nonlinear_constraints = equality_as_inequality_constraints(
            problem.nonlinear_constraints
        )

        raw_res = scipy.optimize.minimize(
            fun=problem.fun_and_jac,
            jac=True,
            x0=x0,
            method="trust-constr",
            bounds=_get_scipy_bounds(problem.bounds),
            constraints=_get_scipy_constraints(nonlinear_constraints),
            options=options,
        )
        res = process_scipy_result(raw_res)
        return res


def process_scipy_result(scipy_res: ScipyOptimizeResult) -> InternalOptimizeResult:
    res = InternalOptimizeResult(
        x=scipy_res.x,
        fun=scipy_res.fun,
        success=bool(scipy_res.success),
        message=str(scipy_res.message),
        n_fun_evals=_int_if_not_none(scipy_res.get("nfev")),
        n_jac_evals=_int_if_not_none(scipy_res.get("njev")),
        n_hess_evals=_int_if_not_none(scipy_res.get("nhev")),
        n_iterations=_int_if_not_none(scipy_res.get("nit")),
        # TODO: Pass on more things once we can convert them to external
        status=None,
        jac=None,
        hess=None,
        hess_inv=None,
        max_constraint_violation=None,
        info=None,
        history=None,
    )
    return res


def _int_if_not_none(value: SupportsInt | None) -> int | None:
    if value is None:
        return None
    return int(value)


def _get_scipy_constraints(constraints):
    """Transform internal nonlinear constraints to scipy readable format.

    This format is currently only used by scipy_trust_constr.

    """
    scipy_constraints = [_internal_to_scipy_constraint(c) for c in constraints]
    return scipy_constraints


def _internal_to_scipy_constraint(c):
    new_constr = NonlinearConstraint(
        fun=c["fun"],
        lb=np.zeros(c["n_constr"]),
        ub=np.tile(np.inf, c["n_constr"]),
        jac=c["jac"],
    )
    return new_constr


@mark.minimizer(
    name="scipy_basinhopping",
    solver_type=AggregationLevel.SCALAR,
    is_available=True,
    is_global=True,
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
class ScipyBasinhopping(Algorithm):
    """Find the global minimum of a scalar function using the basin-hopping
    algorithm.

    Basin-hopping (:cite:`Wales1997`) is a two-phase method that combines a global
    stepping algorithm with local minimization at each step. Designed to mimic the
    natural process of energy minimization of clusters of atoms, it works well for
    similar problems with "funnel-like, but rugged" energy landscapes.

    This algorithm is mainly supported for completeness. Consider using optimagic's
    built-in multistart optimization for a similar approach that can run multiple
    optimizations in parallel, supports all local algorithms in optimagic (as
    opposed to just those from SciPy) and allows for a better visualization of the
    multistart history.

    When a derivative is provided, it is passed to the local minimization method.

    """

    local_algorithm: (
        Literal[
            "Nelder-Mead",
            "Powell",
            "CG",
            "BFGS",
            "Newton-CG",
            "L-BFGS-B",
            "TNC",
            "COBYLA",
            "SLSQP",
            "trust-constr",
            "dogleg",
            "trust-ncg",
            "trust-exact",
            "trust-krylov",
        ]
        | Callable
    ) = "L-BFGS-B"
    """The local optimization algorithm to be used.

    Can be the name of any SciPy local minimizer or a custom function for local
    minimization. The default is "L-BFGS-B", as in SciPy.

    """

    n_local_optimizations: PositiveInt = 100
    """The number of local optimizations that are run.

    The default is 100, as in SciPy.

    """

    temperature: NonNegativeFloat = 1.0
    """The "temperature" parameter for the acceptance or rejection criterion.

    Higher temperatures mean that larger jumps in the function value will be
    accepted. For best results, it should be comparable to the separation (in
    function value) between local minima. The default is 1.0, as in SciPy.

    """

    stepsize: NonNegativeFloat = 0.5
    """Maximum step size for use in the random displacement.

    The default is 0.5, as in SciPy.

    """

    local_algo_options: dict[str, Any] | None = None
    """Additional keyword arguments for the local minimizer.

    Check the documentation of the local SciPy algorithms for details on what is
    supported.

    """

    take_step: Callable | None = None
    """Replace the default step-taking routine with this callable.

    The default is None, which uses SciPy's default step-taking routine.

    """

    accept_test: Callable | None = None
    """Define a test to judge the acceptance of steps.

    The default is None, which uses SciPy's default acceptance test.

    """

    interval: PositiveInt = 50
    """Determines how often the step size is updated.

    The default is 50, as in SciPy.

    """

    convergence_n_unchanged_iterations: PositiveInt | None = None
    """Stop the run if the global minimum candidate remains the same for this number
    of iterations. This is the 'niter_success' parameter in the SciPy
    documentation.

    The default is None, as in SciPy.

    """

    seed: int | np.random.Generator | np.random.RandomState | None = None
    """Seed or random number generator that makes the stochastic parts of the
    algorithm reproducible.

    The default is None, as in SciPy.

    """

    target_accept_rate: NonNegativeFloat = 0.5
    """The target acceptance rate that is used to adjust the step size.

    If the current acceptance rate is greater than the target, the step size is
    increased; otherwise, it is decreased. The range is (0, 1) and the default is
    0.5, as in SciPy.

    """

    stepwise_factor: NonNegativeFloat = 0.9
    """The step size is multiplied or divided by this factor upon each update.

    The range is (0, 1) and the default is 0.9, as in SciPy.

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        n_local_optimizations = max(1, self.n_local_optimizations - 1)
        if self.local_algo_options is None:
            local_algo_options = {}
        else:
            local_algo_options = self.local_algo_options
        minimizer_kwargs = {
            "method": self.local_algorithm,
            "bounds": _get_scipy_bounds(problem.bounds),
            "jac": problem.jac,
        }
        minimizer_kwargs = {**minimizer_kwargs, **local_algo_options}

        res = scipy.optimize.basinhopping(
            func=problem.fun,
            x0=x0,
            minimizer_kwargs=minimizer_kwargs,
            niter=n_local_optimizations,
            T=self.temperature,
            stepsize=self.stepsize,
            take_step=self.take_step,
            accept_test=self.accept_test,
            interval=self.interval,
            niter_success=self.convergence_n_unchanged_iterations,
            seed=self.seed,
            target_accept_rate=self.target_accept_rate,
            stepwise_factor=self.stepwise_factor,
        )

        return process_scipy_result(res)


@mark.minimizer(
    name="scipy_brute",
    solver_type=AggregationLevel.SCALAR,
    is_available=True,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    needs_bounds=True,
    supports_parallelism=True,
    supports_bounds=True,
    supports_infinite_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=True,
)
@dataclass(frozen=True)
class ScipyBrute(Algorithm):
    """Find the global minimum of a scalar function over a given range by brute
    force.

    Brute force evaluates the criterion function at each point of a rectangular
    grid defined by the bounds. Since the number of grid points grows exponentially
    with the number of parameters, the algorithm is only suited for problems with
    very few parameters. Finite bounds are required for all parameters.

    The start values are not actually used because the grid is defined solely by
    the bounds. They are still necessary for optimagic to infer the number and
    format of the parameters.

    The function evaluations can be parallelized over multiple cores. Due to the
    parallelization, this algorithm cannot collect a history of parameters and
    criterion evaluations.

    """

    n_grid_points: PositiveInt = 20
    """The number of grid points per parameter dimension used for the brute force
    search. This is the 'Ns' parameter in the SciPy documentation.

    The default is 20, as in SciPy.

    """

    polishing_function: Callable | None = None
    """Function to seek a more precise minimum near the best grid point, taking the
    best grid point as initial guess as a positional argument. This is the 'finish'
    parameter in the SciPy documentation.

    The default is None, which means that no polishing is performed.

    """

    n_cores: PositiveInt = 1
    """The number of cores on which the function is evaluated in parallel.

    The default is 1.

    """

    batch_evaluator: BatchEvaluatorLiteral | BatchEvaluator = "joblib"
    """An optimagic batch evaluator that is used for the parallel function
    evaluations.

    The default is "joblib".

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        workers = _get_workers(self.n_cores, self.batch_evaluator)
        if problem.bounds.lower is None or problem.bounds.upper is None:
            raise ValueError(
                """Global algorithms like scipy_brute need finite bounds
                 for all parameters"""
            )
        raw_res = scipy.optimize.brute(
            func=problem.fun,
            ranges=tuple(zip(problem.bounds.lower, problem.bounds.upper, strict=True)),
            Ns=self.n_grid_points,
            full_output=True,
            finish=self.polishing_function,
            workers=workers,
        )
        res = InternalOptimizeResult(
            x=raw_res[0],
            fun=raw_res[1],
            n_fun_evals=raw_res[2].size,
            n_iterations=raw_res[2].size,
            success=True,
            message="brute force optimization terminated successfully",
        )

        return res


@mark.minimizer(
    name="scipy_differential_evolution",
    solver_type=AggregationLevel.SCALAR,
    is_available=True,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    needs_bounds=True,
    supports_parallelism=True,
    supports_bounds=True,
    supports_infinite_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=True,
    disable_history=True,
)
@dataclass(frozen=True)
class ScipyDifferentialEvolution(Algorithm):
    """Find the global minimum of a scalar function using differential evolution.

    Differential evolution (:cite:`Storn1997`) is a stochastic, population-based
    evolutionary algorithm. It does not use gradient information, so the criterion
    function does not need to be differentiable. In each generation, candidate
    solutions are mutated by mixing them with other candidate solutions of the
    population and are kept if they improve the criterion value.

    The algorithm can search large areas of the candidate space, but this often
    comes at the cost of requiring more criterion evaluations than gradient-based
    methods. It is a good choice for global optimization of non-smooth or noisy
    problems with a moderate number of parameters. Finite bounds are required for
    all parameters. Nonlinear constraints are supported.

    .. note::
        Due to optimagic's general parameter format, the SciPy arguments
        ``integrality`` and ``vectorized`` are not supported. The SciPy argument
        ``updating`` is always set to "deferred" so that the population can be
        evaluated in parallel.

    """

    strategy: (
        Literal[
            "best1bin",
            "best1exp",
            "rand1exp",
            "randtobest1exp",
            "currenttobest1exp",
            "best2exp",
            "rand2exp",
            "randtobest1bin",
            "currenttobest1bin",
            "best2bin",
            "rand2bin",
            "rand1bin",
        ]
        | Callable
    ) = "best1bin"
    """The differential evolution strategy that is used to create trial candidates.

    The default is "best1bin", as in SciPy.

    """

    stopping_maxiter: PositiveInt = STOPPING_MAXFUN_GLOBAL
    """The maximum number of generations over which the entire population is evolved.

    The maximum number of criterion evaluations (without polishing) is
    ``(stopping_maxiter + 1) * population_size_multiplier * number of parameters``.

    """

    population_size_multiplier: NonNegativeInt = 15
    """A multiplier for setting the total population size. This is the 'popsize'
    parameter in the SciPy documentation.

    The number of individuals in the population is
    ``population_size_multiplier * number of parameters``. The default is 15, as in
    SciPy.

    """

    convergence_ftol_rel: NonNegativeFloat = 0.01
    """Relative tolerance for convergence. This is the 'tol' parameter in the SciPy
    documentation.

    The optimization stops when the standard deviation of the criterion values in
    the population is smaller than
    ``convergence_ftol_abs + convergence_ftol_rel * |mean criterion value|``. The
    default is 0.01, as in SciPy.

    """

    # TODO: Refine type to add ranges [0,2] if float.
    mutation_constant: NonNegativeFloat | Tuple[NonNegativeFloat, NonNegativeFloat] = (
        0.5,
        1,
    )
    """The differential weight, denoted by F in the literature. This is the
    'mutation' parameter in the SciPy documentation.

    Should be within [0, 2]. If a tuple ``(min, max)`` is given, dithering is used:
    the mutation constant is randomly changed on a generation by generation basis,
    which can help speed up convergence significantly. The default is (0.5, 1), as
    in SciPy.

    """

    # TODO: Refine type to add ranges [0,1].
    recombination_constant: NonNegativeFloat = 0.7
    """The crossover probability, denoted by CR in the literature. This is the
    'recombination' parameter in the SciPy documentation.

    It determines the probability that two solution vectors are combined to produce
    a new solution vector. Should be between 0 and 1. Increasing this value allows
    a larger number of mutants to progress into the next generation, but at the
    risk of population stability. The default is 0.7, as in SciPy.

    """

    seed: int | np.random.Generator | np.random.RandomState | None = None
    """Seed or random number generator that makes the stochastic parts of the
    algorithm reproducible.

    The default is None, as in SciPy.

    """

    polish: bool = True
    """If True, the best population member is polished at the end.

    SciPy uses L-BFGS-B for unconstrained problems and trust-constr for constrained
    problems to slightly improve the minimum. The default is True, as in SciPy.

    """

    sampling_method: (
        Literal["latinhypercube", "random", "sobol", "halton"] | NDArray[np.float64]
    ) = "latinhypercube"
    """Method used to generate the initial population. This is the 'init' parameter
    in the SciPy documentation.

    Can be "latinhypercube", "sobol", "halton", "random", or an array specifying
    the initial population of shape (total population size, number of parameters).
    The initial population is clipped to the bounds before use. The default is
    "latinhypercube", as in SciPy.

    """

    convergence_ftol_abs: NonNegativeFloat = CONVERGENCE_SECOND_BEST_FTOL_ABS
    """Absolute tolerance for convergence. This is the 'atol' parameter in the SciPy
    documentation.

    See ``convergence_ftol_rel`` for the exact convergence criterion. optimagic's
    default is 1e-8, while SciPy's default is 0.

    """

    n_cores: PositiveInt = 1
    """The number of cores on which the population is evaluated in parallel.

    The default is 1.

    """

    batch_evaluator: BatchEvaluatorLiteral | BatchEvaluator = "joblib"
    """An optimagic batch evaluator that is used for the parallel function
    evaluations.

    The default is "joblib".

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        workers = _get_workers(self.n_cores, self.batch_evaluator)
        res = scipy.optimize.differential_evolution(
            func=problem.fun,
            bounds=_get_scipy_bounds(problem.bounds),
            strategy=self.strategy,
            maxiter=self.stopping_maxiter,
            popsize=self.population_size_multiplier,
            tol=self.convergence_ftol_rel,
            mutation=self.mutation_constant,
            recombination=self.recombination_constant,
            seed=self.seed,
            polish=self.polish,
            init=self.sampling_method,
            atol=self.convergence_ftol_abs,
            updating="deferred",
            workers=workers,
            constraints=_get_scipy_constraints(problem.nonlinear_constraints),
        )

        return process_scipy_result(res)


@mark.minimizer(
    name="scipy_shgo",
    solver_type=AggregationLevel.SCALAR,
    is_available=True,
    is_global=True,
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
class ScipySHGO(Algorithm):
    """Find the global minimum of a scalar function using simplicial homology global
    optimization (SHGO).

    SHGO (:cite:`Endres2018`) is appropriate for solving general purpose nonlinear
    programming and black-box optimization problems to global optimality
    (low-dimensional problems). It samples the search space, builds a simplicial
    complex on the sampled points from which promising candidate minimizers are
    derived, and refines the candidates with a local optimization algorithm. Bounds
    and nonlinear constraints are supported.

    With the default "simplicial" sampling method, the algorithm has theoretical
    guarantees of convergence to the global minimum in finite time for Lipschitz
    smooth functions.

    .. note::
        If nonlinear constraints are used, only the local algorithms "COBYLA" and
        "SLSQP" take the constraints into account during the local refinement; the
        other local algorithms ignore them.

    """

    local_algorithm: (
        Literal[
            "Nelder-Mead",
            "Powell",
            "CG",
            "BFGS",
            "Newton-CG",
            "L-BFGS-B",
            "TNC",
            "COBYLA",
            "SLSQP",
            "trust-constr",
            "dogleg",
            "trust-ncg",
            "trust-exact",
            "trust-krylov",
        ]
        | Callable
    ) = "L-BFGS-B"
    """The local optimization algorithm to be used.

    Can be the name of any SciPy local minimizer or a custom function for local
    minimization. Only "COBYLA" and "SLSQP" support constraints. The default is
    "L-BFGS-B", as in SciPy.

    """

    local_algo_options: dict[str, Any] | None = None
    """Additional keyword arguments for the local minimizer.

    Check the documentation of the local SciPy algorithms for details on what is
    supported.

    """

    n_sampling_points: PositiveInt = 128
    """The number of sampling points used in the construction of the simplicial
    complex. This is the 'n' parameter in the SciPy documentation.

    optimagic's default is 128.

    """

    n_simplex_iterations: PositiveInt = 1
    """The number of iterations used in the construction of the simplicial complex.
    This is the 'iters' parameter in the SciPy documentation.

    The default is 1, as in SciPy.

    """

    sampling_method: Literal["simplicial", "halton", "sobol"] | Callable = "simplicial"
    """The method used for sampling the search space.

    The default "simplicial" provides the theoretical guarantee of convergence to
    the global minimum in finite time. The "halton" and "sobol" methods are faster
    in terms of sampling point generation at the cost of losing this guarantee. A
    custom sampling function can also be passed.

    """

    max_sampling_evaluations: PositiveInt | None = None
    """The maximum number of criterion evaluations in the feasible domain during the
    sampling phase. This is the 'maxfev' option in the SciPy documentation."""

    convergence_minimum_criterion_value: float | None = None
    """Specify the global minimum of the criterion function when it is known. This is
    the 'f_min' option in the SciPy documentation.

    For maximization problems, flip the sign.

    """

    convergence_minimum_criterion_tolerance: NonNegativeFloat = 1e-4
    """Specify the relative error between the current best minimum and the supplied
    global criterion minimum that is allowed. This is the 'f_tol' option in the
    SciPy documentation.

    The default is 1e-4, as in SciPy.

    """

    stopping_maxiter: PositiveInt | None = None
    """The maximum number of iterations.

    If reached, the optimization stops, but we do not count this as convergence.

    """

    stopping_maxfun: PositiveInt = STOPPING_MAXFUN_GLOBAL
    """The maximum number of sampling evaluations, including points outside the
    feasible domain. This is the 'maxev' option in the SciPy documentation."""

    stopping_max_processing_time: PositiveFloat | None = None
    """The maximum time allowed for the optimization. This is the 'maxtime' option in
    the SciPy documentation."""

    minimum_homology_group_rank_differential: PositiveInt | None = None
    """The minimum difference in the rank of the homology group between iterations.
    This is the 'minhgrd' option in the SciPy documentation.

    The rank of the homology group approximately corresponds to the number of
    locally convex subdomains (and thus local minima) found so far. The algorithm
    terminates when the rank has not grown for this many iterations.

    """

    symmetry: List | bool = False
    """Specify whether the criterion function contains symmetric variables.

    The search space (and therefore performance) can be improved by exploiting
    symmetries.

    """

    minimize_every_iteration: bool = True
    """Specify whether promising global sampling points are passed to the local
    algorithm in every iteration."""

    max_local_minimizations_per_iteration: PositiveInt | bool = False
    """The maximum number of local optimizations per iteration. This is the
    'local_iter' option in the SciPy documentation.

    If False (the default), all promising candidates from the minimizer pool are
    refined, i.e. there is no limit.

    """

    infinity_constraints: bool = True
    """Specify whether sampling points outside the feasible domain are saved. This is
    the 'infty_constraints' option in the SciPy documentation.

    If True, the infeasible points are stored with a criterion value of infinity,
    which requires more memory but fewer criterion evaluations. The default is
    True, as in SciPy.

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if self.local_algorithm == "COBYLA":
            nonlinear_constraints = equality_as_inequality_constraints(
                problem.nonlinear_constraints
            )

        nonlinear_constraints = vector_as_list_of_scalar_constraints(
            problem.nonlinear_constraints
        )

        local_algo_options = (
            {} if self.local_algo_options is None else self.local_algo_options
        )
        default_minimizer_kwargs = {
            "method": self.local_algorithm,
            "bounds": _get_scipy_bounds(problem.bounds),
            "jac": problem.jac,
        }

        minimizer_kwargs = {**default_minimizer_kwargs, **local_algo_options}
        options = {
            "maxfev": self.max_sampling_evaluations,
            "f_min": self.convergence_minimum_criterion_value,
            "f_tol": self.convergence_minimum_criterion_tolerance,
            "maxiter": self.stopping_maxiter,
            "maxev": self.stopping_maxfun,
            "maxtime": self.stopping_max_processing_time,
            "minhgrd": self.minimum_homology_group_rank_differential,
            "symmetry": self.symmetry,
            "jac": problem.jac,
            "minimize_every_iter": self.minimize_every_iteration,
            "local_iter": self.max_local_minimizations_per_iteration,
            "infty_constraints": self.infinity_constraints,
        }

        if any(options.values()) is False:
            options_used = None
        else:
            options_used = options

        res = scipy.optimize.shgo(
            func=problem.fun,
            bounds=_get_scipy_bounds(problem.bounds),
            constraints=nonlinear_constraints,
            minimizer_kwargs=minimizer_kwargs,
            n=self.n_sampling_points,
            iters=self.n_simplex_iterations,
            sampling_method=self.sampling_method,
            options=options_used,
        )

        return process_scipy_result(res)


@mark.minimizer(
    name="scipy_dual_annealing",
    solver_type=AggregationLevel.SCALAR,
    is_available=True,
    is_global=True,
    needs_jac=True,
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
class ScipyDualAnnealing(Algorithm):
    """Find the global minimum of a scalar function using dual annealing.

    Dual annealing is a stochastic global optimization algorithm that implements
    the generalized simulated annealing (GSA) algorithm (:cite:`Tsallis1988`,
    :cite:`TsallisStariolo1996`, :cite:`Xiang1997`), which couples classical and
    fast simulated annealing via a distorted Cauchy-Lorentz visiting distribution,
    and combines it with a local search that is applied to accepted candidate
    solutions.

    The algorithm does not require derivatives of the criterion function for the
    annealing phase (derivatives are passed to the local search algorithm) and is
    suited for noisy or multi-modal problems. Finite bounds are required for all
    parameters.

    """

    stopping_maxiter: PositiveInt = STOPPING_MAXFUN_GLOBAL
    """The maximum number of global search iterations.

    The default is 1000, as in SciPy.

    """

    local_algorithm: (
        Literal[
            "Nelder-Mead",
            "Powell",
            "CG",
            "BFGS",
            "Newton-CG",
            "L-BFGS-B",
            "TNC",
            "COBYLA",
            "SLSQP",
            "trust-constr",
            "dogleg",
            "trust-ncg",
            "trust-exact",
            "trust-krylov",
        ]
        | Callable
    ) = "L-BFGS-B"
    """The local optimization algorithm to be used.

    Can be the name of any SciPy local minimizer or a custom function for local
    minimization. The default is "L-BFGS-B", as in SciPy.

    """

    local_algo_options: dict[str, Any] | None = None
    """Additional keyword arguments for the local minimizer.

    Check the documentation of the local SciPy algorithms for details on what is
    supported.

    """

    # TODO: Refine type to add ranges (0.01, 5e4]
    initial_temperature: PositiveFloat = 5230.0
    """The temperature the algorithm starts with.

    Higher values facilitate a wider search of the energy landscape and allow the
    algorithm to escape local minima that it is trapped in. The range is
    (0.01, 5e4] and the default is 5230.0, as in SciPy.

    """

    # TODO: Refine type to add ranges (0,1)
    restart_temperature_ratio: PositiveFloat = 2e-05
    """Reannealing starts when the temperature has decreased to
    ``initial_temperature * restart_temperature_ratio``.

    The range is (0, 1) and the default is 2e-5, as in SciPy.

    """

    # TODO: Refine type to add ranges (1, 3]
    visit: PositiveFloat = 2.62
    """Specify the thickness of the visiting distribution's tails.

    Higher values give the visiting distribution a heavier tail, which makes the
    algorithm jump to more distant regions. The range is (1, 3] and the default is
    2.62, as in SciPy.

    """

    # TODO: Refine type to add ranges (-1e4, -5]
    accept: NegativeFloat = -5.0
    """Controls the probability of acceptance.

    Lower values lead to a smaller acceptance probability. The range is (-1e4, -5]
    and the default is -5.0, as in SciPy.

    """

    stopping_maxfun: PositiveInt = STOPPING_MAXFUN
    """Soft limit for the number of criterion evaluations.

    If the algorithm is in the middle of a local search when this limit is reached,
    the limit is exceeded and the algorithm stops after the local search is done.
    optimagic's default is 1,000,000, while SciPy's default is 1e7.

    """

    seed: int | np.random.Generator | np.random.RandomState | None = None
    """Seed or random number generator that makes the stochastic parts of the
    algorithm reproducible.

    The default is None, as in SciPy.

    """

    no_local_search: bool = False
    """If set to True, a traditional Generalized Simulated Annealing is performed
    without applying any local search strategy.

    The default is False, as in SciPy.

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        local_algo_options = (
            {} if self.local_algo_options is None else self.local_algo_options
        )
        default_minimizer_kwargs = {
            "method": self.local_algorithm,
            "bounds": _get_scipy_bounds(problem.bounds),
            "jac": problem.jac,
        }

        minimizer_kwargs = {**default_minimizer_kwargs, **local_algo_options}

        res = scipy.optimize.dual_annealing(
            func=problem.fun,
            bounds=_get_scipy_bounds(problem.bounds),
            maxiter=self.stopping_maxiter,
            minimizer_kwargs=minimizer_kwargs,
            initial_temp=self.initial_temperature,
            restart_temp_ratio=self.restart_temperature_ratio,
            visit=self.visit,
            accept=self.accept,
            maxfun=self.stopping_maxfun,
            seed=self.seed,
            no_local_search=self.no_local_search,
            x0=x0,
        )

        return process_scipy_result(res)


@mark.minimizer(
    name="scipy_direct",
    solver_type=AggregationLevel.SCALAR,
    is_available=True,
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
class ScipyDirect(Algorithm):
    """Find the global minimum of a scalar function using the DIRECT algorithm.

    DIRECT stands for DIviding RECTangles (:cite:`Jones1993`). It is a
    deterministic, derivative-free global optimization algorithm based on
    Lipschitzian optimization without knowledge of the Lipschitz constant. The
    algorithm normalizes the search space to a unit hypercube, samples the
    criterion function at the center of the hypercube and then iteratively divides
    the most promising hyperrectangles.

    By default, the locally biased variant DIRECT_L (:cite:`Gablonsky2001`) is
    used, which is often faster on problems without too many local minima.

    Finite bounds are required for all parameters. The start values are not used by
    the algorithm, but they are still necessary for optimagic to infer the number
    and format of the parameters.

    """

    convergence_ftol_rel: NonNegativeFloat = CONVERGENCE_FTOL_REL
    """The minimum required difference of the criterion values between the current
    best hyperrectangle and the next potentially optimal hyperrectangle to be
    divided. This is the 'eps' parameter in the SciPy documentation.

    It determines the trade-off between local and global search: the larger the
    value, the more the search is biased towards local exploration. optimagic's
    default is 2e-9, while SciPy's default is 1e-4.

    """

    stopping_maxfun: PositiveInt = STOPPING_MAXFUN
    """Approximate upper bound on the number of criterion evaluations.

    If reached, the optimization stops, but we do not count this as convergence.
    optimagic's default is 1,000,000. Note that SciPy caps this value if necessary
    to limit DIRECT's memory usage to approximately 1 GiB.

    """

    stopping_maxiter: PositiveInt = STOPPING_MAXFUN_GLOBAL
    """The maximum number of iterations.

    If reached, the optimization stops, but we do not count this as convergence.
    The default is 1000, as in SciPy.

    """

    locally_biased: bool = True
    """If True (default), use the locally biased variant of the algorithm known as
    DIRECT_L (:cite:`Gablonsky2001`).

    Set it to False to use the original, unbiased DIRECT algorithm, which is
    recommended for very difficult problems with many local minima.

    """

    convergence_minimum_criterion_value: float = -np.inf
    """Specify the global minimum of the criterion function when it is known. This is
    the 'f_min' parameter in the SciPy documentation.

    The default is minus infinity, meaning that the global minimum is unknown. For
    maximization problems, flip the sign.

    """

    # TODO: must be between 0 and 1
    convergence_minimum_criterion_tolerance: NonNegativeFloat = 1e-4
    """Specify the relative error between the current best minimum and the supplied
    global minimum that is allowed. This is the 'f_min_rtol' parameter in the SciPy
    documentation.

    It is only used if ``convergence_minimum_criterion_value`` is set. The default
    is 1e-4, as in SciPy.

    """

    # TODO: must be between 0 and 1
    volume_hyperrectangle_tolerance: NonNegativeFloat = 1e-16
    """Stop when the volume of the hyperrectangle containing the lowest criterion
    value is smaller than this fraction of the complete search space. This is the
    'vol_tol' parameter in the SciPy documentation.

    The range is (0, 1) and the default is 1e-16, as in SciPy.

    """

    # TODO: must be between 0 and 1
    length_hyperrectangle_tolerance: NonNegativeFloat = 1e-6
    """Stopping criterion based on the length of the hyperrectangle containing the
    lowest criterion value. This is the 'len_tol' parameter in the SciPy
    documentation.

    If ``locally_biased`` is True, terminate when half of the normalized maximal
    side length of this hyperrectangle is smaller than this value. If
    ``locally_biased`` is False, terminate when half of the normalized diagonal of
    this hyperrectangle is smaller than this value. The range is (0, 1) and the
    default is 1e-6, as in SciPy.

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        res = scipy.optimize.direct(
            func=problem.fun,
            bounds=_get_scipy_bounds(problem.bounds),
            eps=self.convergence_ftol_rel,
            maxfun=self.stopping_maxfun,
            maxiter=self.stopping_maxiter,
            locally_biased=self.locally_biased,
            f_min=self.convergence_minimum_criterion_value,
            f_min_rtol=self.convergence_minimum_criterion_tolerance,
            vol_tol=self.volume_hyperrectangle_tolerance,
            len_tol=self.length_hyperrectangle_tolerance,
        )

        return process_scipy_result(res)


def _get_workers(n_cores, batch_evaluator):
    batch_evaluator = process_batch_evaluator(batch_evaluator)
    out = functools.partial(
        batch_evaluator,
        n_cores=n_cores,
        error_handling="raise",
    )
    return out


def _get_scipy_bounds(bounds: InternalBounds) -> ScipyBounds | None:
    if bounds.lower is None and bounds.upper is None:
        return None

    lower = bounds.lower if bounds.lower is not None else -np.inf
    upper = bounds.upper if bounds.upper is not None else np.inf
    return ScipyBounds(lb=lower, ub=upper)


def process_scipy_result_old(scipy_results_obj):
    # using get with defaults to access dict elements is just a safety measure
    raw_res = {**scipy_results_obj}
    processed = {
        "solution_x": raw_res.get("x"),
        "solution_criterion": raw_res.get("fun"),
        "solution_derivative": raw_res.get("jac"),
        "solution_hessian": raw_res.get("hess"),
        "n_fun_evals": raw_res.get("nfev"),
        "n_jac_evals": raw_res.get("njac") or raw_res.get("njev"),
        "n_iterations": raw_res.get("nit"),
        "success": raw_res.get("success"),
        "reached_convergence_criterion": None,
        "message": raw_res.get("message"),
    }
    return processed
