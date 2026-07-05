"""This module implements the POUNDERs algorithm."""

from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from optimagic import mark
from optimagic.config import IS_PETSC4PY_INSTALLED
from optimagic.exceptions import NotInstalledError
from optimagic.optimization.algo_options import (
    CONVERGENCE_GTOL_ABS,
    CONVERGENCE_GTOL_REL,
    CONVERGENCE_GTOL_SCALED,
    STOPPING_MAXITER,
)
from optimagic.optimization.algorithm import Algorithm, InternalOptimizeResult
from optimagic.optimization.internal_optimization_problem import (
    InternalOptimizationProblem,
)
from optimagic.typing import AggregationLevel, NonNegativeFloat, PositiveInt
from optimagic.utilities import calculate_trustregion_initial_radius


@mark.minimizer(
    name="tao_pounders",
    solver_type=AggregationLevel.LEAST_SQUARES,
    is_available=IS_PETSC4PY_INSTALLED,
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
class TAOPounders(Algorithm):
    r"""Minimize a nonlinear least-squares problem using the POUNDERS algorithm.

    POUNDERS (Practical Optimization Using No Derivatives for sums of Squares,
    :cite:`Wild2015`) is a derivative-free trust-region algorithm that is tailored
    to nonlinear least-squares problems. It is part of the Toolkit for Advanced
    Optimization (TAO, :cite:`Benson2017`), which is distributed together with
    PETSc, and is wrapped in optimagic via
    `petsc4py <https://pypi.org/project/petsc4py/>`_.

    Instead of working only with the scalar sum of squares, POUNDERS exploits the
    availability of the full vector of residuals: it maintains an interpolation-based
    quadratic model of each residual around the current iterate (building on the
    minimal norm Hessian models of :cite:`Wild2008`) and combines these models
    according to the known least-squares structure of the objective. Candidate steps
    are obtained by minimizing the combined model inside a dynamically resized trust
    region.

    Use POUNDERS if your criterion function is a nonlinear sum of squares, evaluating
    the residuals is expensive (e.g., each evaluation requires a simulation),
    derivatives are not available, and the residuals are possibly slightly noisy.
    This makes it a useful tool for economists who estimate structural models using
    indirect inference or the method of simulated moments. Compared to
    Levenberg-Marquardt style solvers, which need the Jacobian of the residuals
    (either analytically or via finite differences, which costs many extra
    evaluations per iteration and is unreliable for noisy functions), POUNDERS gets
    by without any derivative information. Compared to derivative-free algorithms
    that ignore the least-squares structure, such as Nelder-Mead, it typically
    requires far fewer criterion evaluations to arrive at a local optimum. If the
    residuals are smooth, cheap to evaluate, and derivatives are available, a
    derivative-based least-squares solver is usually the better choice.

    Scale the problem such that the bounds correspond to the unit hypercube
    :math:`[0, 1]^n`. For unconstrained problems, scale each parameter such that
    unit changes in the parameters result in similar order-of-magnitude changes in
    the criterion value(s).

    .. note::
        This algorithm requires the petsc4py package, which is not available on
        Windows. Windows users can use optimagic's own ``pounders`` algorithm
        instead.

    """

    convergence_gtol_abs: NonNegativeFloat = CONVERGENCE_GTOL_ABS
    r"""Stop if the norm of the gradient falls below this value.

    .. math::

        \lVert g(X) \rVert \leq \textsf{convergence_gtol_abs}

    This is the absolute gradient tolerance ``gatol`` in the TAO documentation.
    Set it to zero to disable this criterion.

    """

    convergence_gtol_rel: NonNegativeFloat = CONVERGENCE_GTOL_REL
    r"""Stop if the norm of the gradient relative to the criterion value falls below
    this value.

    .. math::

        \frac{\lVert g(X) \rVert}{|f(X)|} \leq \textsf{convergence_gtol_rel}

    This is the relative gradient tolerance ``grtol`` in the TAO documentation.
    Set it to zero to disable this criterion.

    """

    convergence_gtol_scaled: NonNegativeFloat = CONVERGENCE_GTOL_SCALED
    r"""Stop if the norm of the gradient falls below this fraction of the norm of the
    gradient at the start parameters :math:`X_0`.

    .. math::

        \frac{\lVert g(X) \rVert}{\lVert g(X_0) \rVert} \leq
        \textsf{convergence_gtol_scaled}

    This is the gradient reduction tolerance ``gttol`` in the TAO documentation.
    Set it to zero to disable this criterion.

    """

    trustregion_initial_radius: NonNegativeFloat | None = None
    r"""Initial value of the trust region radius. It must be larger than zero.

    If None (the default), the radius is set to
    :math:`0.1 \max(\lVert x_0 \rVert_\infty, 1)`, where :math:`x_0` are the start
    parameters.

    """

    stopping_maxiter: PositiveInt = STOPPING_MAXITER
    """Alternative stopping criterion.

    If set, the routine stops after the specified number of iterations or after the
    step size is sufficiently small. If the variable is set, the default convergence
    criteria are all ignored.

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        raw = tao_pounders(
            criterion=problem.fun,
            x=x0,
            lower_bounds=problem.bounds.lower,
            upper_bounds=problem.bounds.upper,
            convergence_gtol_abs=self.convergence_gtol_abs,
            convergence_gtol_rel=self.convergence_gtol_rel,
            convergence_gtol_scaled=self.convergence_gtol_scaled,
            trustregion_initial_radius=self.trustregion_initial_radius,
            stopping_maxiter=self.stopping_maxiter,
        )

        res = InternalOptimizeResult(
            x=raw["solution_x"],
            fun=raw["solution_criterion"],
            success=raw["success"],
            message=raw["message"],
            n_fun_evals=raw["n_fun_evals"],
            n_jac_evals=0,
            n_hess_evals=0,
            n_iterations=raw["n_iterations"],
            info={
                "gradient_norm": raw["gradient_norm"],
                "criterion_norm": raw["criterion_norm"],
                "convergence_code": raw["convergence_code"],
                "convergence_reason": raw["reached_convergence_criterion"],
            },
        )

        return res


def tao_pounders(
    criterion,
    x,
    lower_bounds,
    upper_bounds,
    *,
    convergence_gtol_abs=CONVERGENCE_GTOL_ABS,
    convergence_gtol_rel=CONVERGENCE_GTOL_REL,
    convergence_gtol_scaled=CONVERGENCE_GTOL_SCALED,
    trustregion_initial_radius=None,
    stopping_maxiter=STOPPING_MAXITER,
):
    r"""Minimize a function using the POUNDERs algorithm.

    For details see
    :ref: `tao_algorithm`.

    """
    if not IS_PETSC4PY_INSTALLED:
        raise NotInstalledError(
            "The 'tao_pounders' algorithm requires the petsc4py package to be "
            "installed. If you are using Linux or MacOS, install the package with "
            "'conda install -c conda-forge petsc4py'. The package is not available on "
            "Windows. Windows users can use optimagics 'pounders' algorithm instead."
        )
    from petsc4py import PETSc

    first_eval = criterion(x)
    n_errors = len(first_eval)
    _x = _initialise_petsc_array(x)
    # We need to know the number of contributions of the criterion value to allocate the
    # array.
    residuals_out = _initialise_petsc_array(n_errors)

    # Create the solver object.
    tao = PETSc.TAO().create(PETSc.COMM_WORLD)

    # Set the solver type.
    tao.setType("pounders")

    tao.setFromOptions()

    def func_tao(tao, x, resid_out):  # noqa: ARG001
        """Evaluate objective and attach result to an petsc object f.

        This is required to use the pounders solver from tao.

        Args:
             tao: The tao object we created for the optimization task.
             x (PETSc.array): Current parameter values.
             f: Petsc object in which we save the current function value.

        """
        resid_out.array = criterion(x.array)

    # Set the procedure for calculating the objective. This part has to be changed if we
    # want more than pounders.
    tao.setResidual(func_tao, residuals_out)

    if trustregion_initial_radius is None:
        trustregion_initial_radius = calculate_trustregion_initial_radius(_x)
    elif trustregion_initial_radius <= 0:
        raise ValueError("The initial trust region radius must be > 0.")
    tao.setInitialTrustRegionRadius(trustregion_initial_radius)

    # Add bounds if provided.
    if lower_bounds is not None or upper_bounds is not None:
        if lower_bounds is None:
            lower_bounds = np.full(len(x), -np.inf)
        if upper_bounds is None:
            upper_bounds = np.full(len(x), np.inf)
        lower_bounds = _initialise_petsc_array(lower_bounds)
        upper_bounds = _initialise_petsc_array(upper_bounds)
        tao.setVariableBounds(lower_bounds, upper_bounds)

    # Put the starting values into the container and pass them to the optimizer.
    tao.setInitial(_x)

    # Obtain tolerances for the convergence criteria. Since we can not create
    # scaled_gradient_tolerance manually we manually set absolute_gradient_tolerance and
    # or relative_gradient_tolerance to zero once a subset of these two is turned off
    # and scaled_gradient_tolerance is still turned on.
    default_gatol = convergence_gtol_abs if convergence_gtol_abs else -1
    default_gttol = convergence_gtol_scaled if convergence_gtol_scaled else -1
    default_grtol = convergence_gtol_rel if convergence_gtol_rel else -1
    # Set tolerances for default convergence tests.
    tao.setTolerances(
        gatol=default_gatol,
        grtol=default_grtol,
        gttol=default_gttol,
    )

    # Set user defined convergence tests. Beware that specifying multiple tests could
    # overwrite others or lead to unclear behavior.
    if stopping_maxiter is not None:
        tao.setConvergenceTest(functools.partial(_max_iters, stopping_maxiter))
    elif convergence_gtol_scaled is False and convergence_gtol_abs is False:
        tao.setConvergenceTest(functools.partial(_grtol_conv, convergence_gtol_rel))
    elif convergence_gtol_rel is False and convergence_gtol_scaled is False:
        tao.setConvergenceTest(functools.partial(_gatol_conv, convergence_gtol_abs))
    elif convergence_gtol_scaled is False:
        tao.setConvergenceTest(
            functools.partial(
                _grtol_gatol_conv,
                convergence_gtol_rel,
                convergence_gtol_abs,
            )
        )

    # Run the problem.
    tao.solve()

    results = _process_pounders_results(residuals_out, tao)

    # Destroy petsc objects for memory reasons.
    petsc_bounds = [b for b in (lower_bounds, upper_bounds) if b is not None]
    for obj in [tao, _x, residuals_out, *petsc_bounds]:
        obj.destroy()

    return results


def _initialise_petsc_array(len_or_array):
    """Initialize an empty array or fill in provided values.

    Args:
        len_or_array (int or numpy.ndarray): If the value is an integer, allocate an
            empty array with the given length. If the value is an array, allocate an
            array of equal length and fill in the values.

    """
    from petsc4py import PETSc

    length = len_or_array if isinstance(len_or_array, int) else len(len_or_array)

    array = PETSc.Vec().create(PETSc.COMM_WORLD)
    array.setSizes(length)
    array.setFromOptions()

    if isinstance(len_or_array, np.ndarray):
        array.array = len_or_array

    return array


def _max_iters(max_iterations, tao):
    if tao.getSolutionStatus()[0] < max_iterations:
        return 0
    elif tao.getSolutionStatus()[0] >= max_iterations:
        tao.setConvergedReason(8)


def _gatol_conv(absolute_gradient_tolerance, tao):
    if tao.getSolutionStatus()[2] >= absolute_gradient_tolerance:
        return 0
    elif tao.getSolutionStatus()[2] < absolute_gradient_tolerance:
        tao.setConvergedReason(3)


def _grtol_conv(relative_gradient_tolerance, tao):
    if (
        tao.getSolutionStatus()[2] / tao.getSolutionStatus()[1]
        >= relative_gradient_tolerance
    ):
        return 0
    elif (
        tao.getSolutionStatus()[2] / tao.getSolutionStatus()[1]
        < relative_gradient_tolerance
    ):
        tao.setConvergedReason(4)


def _grtol_gatol_conv(relative_gradient_tolerance, absolute_gradient_tolerance, tao):
    if (
        tao.getSolutionStatus()[2] / tao.getSolutionStatus()[1]
        >= relative_gradient_tolerance
    ):
        return 0
    elif (
        tao.getSolutionStatus()[2] / tao.getSolutionStatus()[1]
        < relative_gradient_tolerance
    ):
        tao.setConvergedReason(4)

    elif tao.getSolutionStatus()[2] < absolute_gradient_tolerance:
        tao.setConvergedReason(3)


def _translate_tao_convergence_reason(tao_resaon):
    mapping = {
        3: "absolute_gradient_tolerance below critical value",
        4: "relative_gradient_tolerance below critical value",
        5: "scaled_gradient_tolerance below critical value",
        6: "step size small",
        7: "objective below min value",
        8: "user defined",
        -2: "maxits reached",
        -4: "numerical problems",
        -5: "max funcevals reached",
        -6: "line search failure",
        -7: "trust region failure",
        -8: "user defined",
    }
    return mapping[tao_resaon]


def _process_pounders_results(residuals_out, tao):
    convergence_code = tao.getConvergedReason()
    convergence_reason = _translate_tao_convergence_reason(convergence_code)

    results = {
        "solution_x": tao.solution.array,
        "solution_criterion": tao.function,
        "solution_derivative": None,
        "solution_hessian": None,
        "n_fun_evals": tao.getIterationNumber(),
        "n_jac_evals": None,
        "n_iterations": None,
        "success": bool(convergence_code >= 0),
        "reached_convergence_criterion": (
            convergence_reason if convergence_code >= 0 else None
        ),
        "message": convergence_reason,
        # Further results.
        "solution_criterion_values": residuals_out.array,
        "gradient_norm": tao.gnorm,
        "criterion_norm": tao.cnorm,
        "convergence_code": convergence_code,
    }

    return results
