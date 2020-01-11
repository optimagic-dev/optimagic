"""Wrapper for pounders in the tao package."""
import sys
from functools import partial

import numpy as np

if sys.platform != "win32":
    from petsc4py import PETSc


def minimize_pounders_np(
    func,
    x0,
    bounds,
    gatol=1e-8,
    grtol=1e-8,
    gttol=1e-10,
    init_tr=None,
    max_iterations=None,
    n_errors=None,
):
    """Minimize a function using the Pounders algorithm.

    Pounders can be a useful tool for economists who estimate structural models using
    indirect inference, because unlike commonly used algorithms such as Nelder-Mead,
    Pounders is tailored for minimizing a non-linear sum of squares objective function,
    and therefore may require fewer iterations to arrive at a local optimum than
    Nelder-Mead.

    The criterion function :func:`func` should return an array of the errors NOT an
    array of the squared errors or anything else.

    Args:
        func (callable): Objective function.
        x0 (np.ndarray): Starting values of the parameters.
        bounds (Tuple[np.ndarray]): A tuple containing two NumPy arrays where the first
            corresponds to the lower and the second to the upper bound. Unbounded
            parameters are represented by infinite values. The arrays have the same
            length as the parameter vector.
        gatol (float): Stop if relative norm of gradient is less than this. If set to
            False the algorithm will not consider gatol. Default is 1e-8.
        grtol (float): Stop if norm of gradient is less than this. If set to False the
            algorithm will not consider grtol. Default is 1e-8.
        gttol (float): Stop if norm of gradient is reduced by this factor. If set to
            False the algorithm will not consider grtol. Default is 1e-10.
        init_tr (float): Sets the radius for the initial trust region that the optimizer
            employs. If `None` the algorithm uses 100 as initial  trust region radius.
            Default is `None`.
        max_iterations (int): Alternative Stopping criterion. If set the routine will
            stop after the number of specified iterations or after the step size is
            sufficiently small. If the variable is set the default criteria will all be
            ignored. Default is `None`.
        n_errors (int or None): The number of outputs of `func` are necessary to
            pre-allocate the results array. If the argument is ``None``, evaluate the
            function once. This might be undesirable during dashboard optimizations.

    Returns:
        results (dict): Dictionary with processed optimization results.

    .. _TAO Users Manual:
        https://www.mcs.anl.gov/petsc/petsc-current/docs/tao_manual.pdf
    .. _Solving Derivative-Free Nonlinear Least Squares Problems with POUNDERS:
        https://www.mcs.anl.gov/papers/P5120-0414.pdf

    """
    if sys.platform == "win32":
        raise NotImplementedError("The pounders algorithm is not available on Windows.")

    # We need to know the dimension of the output of the criterion function. Evaluate
    # plain `criterion` to prevent logging.
    if n_errors is None:
        n_errors = len(func(x0))

    # We want to get containers for the func vector and the paras.
    x0 = _initialise_petsc_array(x0)
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

    # We try to set user defined convergence tests.
    if init_tr is not None:
        tao.setInitialTrustRegionRadius(init_tr)

    # Add bounds.
    n_params = len(x0.array)
    processed_bounds = []
    for bound in bounds:
        bound = np.full(n_params, bound) if isinstance(bound, (int, float)) else bound
        processed_bounds.append(_initialise_petsc_array(bound))
    tao.setVariableBounds(processed_bounds)

    # Put the starting values into the container and pass them to the optimizer.
    tao.setInitial(x0)

    # Obtain tolerances for the convergence criteria. Since we can not create gttol
    # manually we manually set gatol and or grtol to zero once a subset of these two is
    # turned off and gttol is still turned on.
    default_gatol = gatol if gatol else -1
    default_gttol = gttol if gttol else -1
    default_grtol = grtol if grtol else -1
    # Set tolerances for default convergence tests.
    tao.setTolerances(gatol=default_gatol, gttol=default_gttol, grtol=default_grtol)

    # Set user defined convergence tests. Beware that specifying multiple tests could
    # overwrite others or lead to unclear behavior.
    if max_iterations is not None:
        tao.setConvergenceTest(partial(_max_iters, max_iterations))
    elif gttol is False and gatol is False:
        tao.setConvergenceTest(partial(_grtol_conv, grtol))
    elif grtol is False and gttol is False:
        tao.setConvergenceTest(partial(_gatol_conv, gatol))
    elif gttol is False:
        tao.setConvergenceTest(partial(_grtol_gatol_conv, grtol, gatol))

    # Run the problem.
    tao.solve()

    results = _process_pounders_results(residuals_out, tao)

    # Destroy petsc objects for memory reasons.
    tao.destroy()
    x0.destroy()
    residuals_out.destroy()

    return results


def _initialise_petsc_array(len_or_array):
    """Initialize an empty array or fill in provided values.

    Args:
        len_or_array (int or np.ndarray): If the value is an integer, allocate an empty
            array with the given length. If the value is an array, allocate an array of
            equal length and fill in the values.

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


def _gatol_conv(gatol, tao):
    if tao.getSolutionStatus()[2] >= gatol:
        return 0
    elif tao.getSolutionStatus()[2] < gatol:
        tao.setConvergedReason(3)


def _grtol_conv(grtol, tao):
    if tao.getSolutionStatus()[2] / tao.getSolutionStatus()[1] >= grtol:
        return 0
    elif tao.getSolutionStatus()[2] / tao.getSolutionStatus()[1] < grtol:
        tao.setConvergedReason(4)


def _grtol_gatol_conv(grtol, gatol, tao):
    if tao.getSolutionStatus()[2] / tao.getSolutionStatus()[1] >= grtol:
        return 0
    elif tao.getSolutionStatus()[2] / tao.getSolutionStatus()[1] < grtol:
        tao.setConvergedReason(4)

    elif tao.getSolutionStatus()[2] < gatol:
        tao.setConvergedReason(3)


def _translate_tao_convergence_reason(tao_resaon):
    mapping = {
        3: "gatol below critical value",
        4: "grtol below critical value",
        5: "gttol below critical value",
        6: "step size small",
        7: "objective below min value",
        8: "user defined",
        -2: "maxits reached",
        -4: "numerical porblems",
        -5: "max funcevals reached",
        -6: "line search failure",
        -7: "trust region failure",
        -8: "user defined",
    }
    return mapping[tao_resaon]


def _process_pounders_results(residuals_out, tao):
    results = {
        # Harmonized results.
        "status": "success",
        "fitness": tao.function,
        "x": tao.solution.array,
        "n_evaluations": tao.getIterationNumber(),
        # Other results.
        "fitness_values": residuals_out.array,
        "conv": _translate_tao_convergence_reason(tao.getConvergedReason()),
        "gnorm": tao.gnorm,
        "cnorm": tao.cnorm,
    }

    return results
