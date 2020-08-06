"""This module implements the POUNDERs algorithm."""
import functools
import sys

import numpy as np

if sys.platform != "win32":
    from petsc4py import PETSc


POUNDERS_ALGO_INFO = {
    "primary_criterion_entry": "root_contributions",
    "parallelizes": False,
    "needs_scaling": True,
    "name": "tao_pounders",
}


def tao_pounders(
    criterion_and_derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    gradient_absolute_tolerance=1e-8,
    gradient_relative_tolerance=1e-8,
    gradient_total_tolerance=1e-10,
    initial_trust_region_radius=100,
    max_iterations=None,
):
    r"""Minimize a function using the POUNDERs algorithm.

    POUNDERs can be a useful tool for economists who estimate structural models using
    indirect inference, because unlike commonly used algorithms such as Nelder-Mead,
    POUNDERs is tailored for minimizing a non-linear sum of squares objective function,
    and therefore may require fewer iterations to arrive at a local optimum than
    Nelder-Mead.

    The criterion function :func:`func` should return an array of the errors NOT an
    array of the squared errors or anything else.

    Scaling the problem is necessary such that bounds correspond to the unit hypercube
    :mat:`[0, 1]^n`. For unconstrained problems, scale each parameter such that unit
    changes in parameters result in similar order-of-magnitude changes in the criterion
    value(s).

    POUNDERs has several convergence criteria. Let :math:`X` be the current parameter
    vector, :math:`X_0` the initial parameter vector, :math:`g` the gradient, and
    :math:`f` the criterion function.

    ``gradient_absolute_tolerance`` stops the optimization if the norm of the gradient
    falls below :math:`\epsilon`.

    .. math::

        ||g(X)|| < \epsilon

    ``gradient_relative_tolerance`` stops the optimization if the norm of the gradient
    relative to the criterion value falls below :math:`epsilon`.

    .. math::

        ||g(X)|| / |f(X)| < \epsilon

    ``gradient_total_tolerance`` stops the optimization if the norm of the gradient is
    lower than some fraction :math:`epsilon` of the norm of the gradient at the initial
    parameters.

    .. math::

        ||g(X)|| / ||g(X0)|| < \epsilon

    Args:
        gradient_absolute_tolerance (float): Stop if relative norm of gradient is less
            than this. If set to False the algorithm will not consider
            gradient_absolute_tolerance. Default is 1e-8.
        gradient_relative_tolerance (float): Stop if norm of gradient is less than this.
            If set to False the algorithm will not consider gradient_relative_tolerance.
            Default is 1e-8.
        gradient_total_tolerance (float): Stop if norm of gradient is reduced by this
            factor. If set to False the algorithm will not consider
            gradient_relative_tolerance. Default is 1e-10.
        initial_trust_region_radius (float): Sets the radius for the initial trust
            region that the optimizer employs. It must be :math:`> 0`.
        max_iterations (int): Alternative Stopping criterion. If set the routine will
            stop after the number of specified iterations or after the step size is
            sufficiently small. If the variable is set the default criteria will all be
            ignored. Default is `None`.

    Returns:
        results (dict): Dictionary with processed optimization results.

    References:
    .. _TAO Users Manual (Revision 3.7):
        http://web.mit.edu/tao-petsc_v3.7/tao_manual.pdf
    .. _Solving Derivative-Free Nonlinear Least Squares Problems with POUNDERS:
        https://www.mcs.anl.gov/papers/P5120-0414.pdf
    .. _petsc4py on BitBucket:
        https://bitbucket.org/petsc/petsc4py

    """
    if sys.platform == "win32":
        raise NotImplementedError("The pounders algorithm is not available on Windows.")

    func = functools.partial(
        criterion_and_derivative, task="criterion", algorithm_info=POUNDERS_ALGO_INFO,
    )

    x = _initialise_petsc_array(x)
    # We need to know the number of contributions of the criterion value to allocate the
    # array.
    n_errors = len(
        criterion_and_derivative.keywords["first_criterion_evaluation"]["output"][
            "root_contributions"
        ]
    )
    residuals_out = _initialise_petsc_array(n_errors)

    # Create the solver object.
    tao = PETSc.TAO().create(PETSc.COMM_WORLD)

    # Set the solver type.
    tao.setType("pounders")

    tao.setFromOptions()

    def func_tao(tao, x, resid_out):
        """Evaluate objective and attach result to an petsc object f.

        This is required to use the pounders solver from tao.

        Args:
             tao: The tao object we created for the optimization task.
             x (PETSc.array): Current parameter values.
             f: Petsc object in which we save the current function value.

        """
        resid_out.array = func(x.array)

    # Set the procedure for calculating the objective. This part has to be changed if we
    # want more than pounders.
    tao.setResidual(func_tao, residuals_out)

    if initial_trust_region_radius <= 0:
        raise ValueError("The initial trust region radius must be > 0.")
    tao.setInitialTrustRegionRadius(initial_trust_region_radius)

    # Add bounds.
    lower_bounds = _initialise_petsc_array(lower_bounds)
    upper_bounds = _initialise_petsc_array(upper_bounds)
    tao.setVariableBounds(lower_bounds, upper_bounds)

    # Put the starting values into the container and pass them to the optimizer.
    tao.setInitial(x)

    # Obtain tolerances for the convergence criteria. Since we can not create
    # gradient_total_tolerance manually we manually set gradient_absolute_tolerance and
    # or gradient_relative_tolerance to zero once a subset of these two is turned off
    # and gradient_total_tolerance is still turned on.
    default_gatol = gradient_absolute_tolerance if gradient_absolute_tolerance else -1
    default_gttol = gradient_total_tolerance if gradient_total_tolerance else -1
    default_grtol = gradient_relative_tolerance if gradient_relative_tolerance else -1
    # Set tolerances for default convergence tests.
    tao.setTolerances(
        gatol=default_gatol, grtol=default_grtol, gttol=default_gttol,
    )

    # Set user defined convergence tests. Beware that specifying multiple tests could
    # overwrite others or lead to unclear behavior.
    if max_iterations is not None:
        tao.setConvergenceTest(functools.partial(_max_iters, max_iterations))
    elif gradient_total_tolerance is False and gradient_absolute_tolerance is False:
        tao.setConvergenceTest(
            functools.partial(_grtol_conv, gradient_relative_tolerance)
        )
    elif gradient_relative_tolerance is False and gradient_total_tolerance is False:
        tao.setConvergenceTest(
            functools.partial(_gatol_conv, gradient_absolute_tolerance)
        )
    elif gradient_total_tolerance is False:
        tao.setConvergenceTest(
            functools.partial(
                _grtol_gatol_conv,
                gradient_relative_tolerance,
                gradient_absolute_tolerance,
            )
        )

    # Run the problem.
    tao.solve()

    results = _process_pounders_results(residuals_out, tao)

    # Destroy petsc objects for memory reasons.
    for obj in [tao, x, residuals_out, lower_bounds, upper_bounds]:
        obj.destroy()

    return results


def _initialise_petsc_array(len_or_array):
    """Initialize an empty array or fill in provided values.

    Args:
        len_or_array (int or numpy.ndarray): If the value is an integer, allocate an
            empty array with the given length. If the value is an array, allocate an
            array of equal length and fill in the values.

    """
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


def _gatol_conv(gradient_absolute_tolerance, tao):
    if tao.getSolutionStatus()[2] >= gradient_absolute_tolerance:
        return 0
    elif tao.getSolutionStatus()[2] < gradient_absolute_tolerance:
        tao.setConvergedReason(3)


def _grtol_conv(gradient_relative_tolerance, tao):
    if (
        tao.getSolutionStatus()[2] / tao.getSolutionStatus()[1]
        >= gradient_relative_tolerance
    ):
        return 0
    elif (
        tao.getSolutionStatus()[2] / tao.getSolutionStatus()[1]
        < gradient_relative_tolerance
    ):
        tao.setConvergedReason(4)


def _grtol_gatol_conv(gradient_relative_tolerance, gradient_absolute_tolerance, tao):
    if (
        tao.getSolutionStatus()[2] / tao.getSolutionStatus()[1]
        >= gradient_relative_tolerance
    ):
        return 0
    elif (
        tao.getSolutionStatus()[2] / tao.getSolutionStatus()[1]
        < gradient_relative_tolerance
    ):
        tao.setConvergedReason(4)

    elif tao.getSolutionStatus()[2] < gradient_absolute_tolerance:
        tao.setConvergedReason(3)


def _translate_tao_convergence_reason(tao_resaon):
    mapping = {
        3: "gradient_absolute_tolerance below critical value",
        4: "gradient_relative_tolerance below critical value",
        5: "gradient_total_tolerance below critical value",
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
        "n_criterion_evaluations": tao.getIterationNumber(),
        "n_derivative_evaluations": None,
        "n_iterations": None,
        "success": True if convergence_code >= 0 else False,
        "reached_convergence_criterion": convergence_reason
        if convergence_code >= 0
        else None,
        "message": convergence_reason,
        # Further results.
        "solution_criterion_values": residuals_out.array,
        "gradient_norm": tao.gnorm,
        "criterion_norm": tao.cnorm,
    }

    return results
