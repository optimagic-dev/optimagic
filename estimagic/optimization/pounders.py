"""Wrapper for pounders in the tao package."""
import sys
from functools import partial

import numpy as np

if sys.platform != "win32":
    from petsc4py import PETSc


def minimize_pounders(
    internal_criterion, internal_params, criterion, params, algo_options
):
    # Prepare arguments
    bounds = tuple(params.query("_internal_free")[["lower", "upper"]].to_numpy().T)
    n_squared_errors = len(criterion(params))
    x0 = internal_params["value"].to_numpy()

    return minimize_pounders_np(
        x0,
        internal_criterion,
        bounds,
        **algo_options,
        n_squared_errors=n_squared_errors,
    )


def minimize_pounders_np(
    x0,
    fun,
    bounds=(-np.inf, np.inf),
    gatol=1e-8,
    grtol=1e-8,
    gttol=1e-10,
    init_tr=None,
    max_iterations=None,
    n_squared_errors=None,
):
    """Minimize a function using the Pounders algorithm.

    Pounders can be a useful tool for economists who estimate structural models using
    indirect inference, because unlike commonly used algorithms such as Nelder-Mead,
    Pounders is tailored for minimizing a non-linear sum of squares objective function,
    and therefore may require fewer iterations to arrive at a local optimum than
    Nelder-Mead.

    Args:
        x0 (np.ndarray): Starting values of parameters.
        fun (callable): Function to be minimized.
        bounds (tuple): Bounds are either a tuple of number or arrays. The first
            elements specifies the lower and the second the upper bound of parameters.
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
        n_squared_errors (int or None): The number of outputs of `fun` are necessary to
            pre-allocate the results array. If the argument is ``None``, evaluate the
            function once. This might be undesirable during dashboard optimizations.

    Returns:
        result (dict): Dictionary with the following key-value pairs:

            - `"solution"`: solution vector as `np.ndarray`.
            - `"func_values"`: `np.ndarray` of value of the objective at the solution.
            - `"x0"`: `np.ndarray` of the start values.
            - `"conv"`: string indicating the termination reason.
            - `"sol"`: `list` containing ...
              - current iterate as integer.
              - current value of the objective as float
              - current value of the approximated
              - jacobian as float
              - infeasability norm as float
              - step length as float
              - termination reason as int.

    .. _TAO Users Manual:
        https://www.mcs.anl.gov/petsc/petsc-current/docs/tao_manual.pdf
    .. _Solving Derivative-Free Nonlinear Least Squares Problems with POUNDERS:
        https://www.mcs.anl.gov/papers/P5120-0414.pdf

    """
    if sys.platform == "win32":
        raise NotImplementedError("The pounders algorithm is not available on Windows.")

    # We need to know the dimension of the output of the criterion function. Evaluate
    # plain `criterion` to prevent logging.
    if n_squared_errors is None:
        n_squared_errors = len(fun(x0))

    # We want to get containers for the func vector and the paras.
    x0 = initialise_petsc_array(x0)
    squared_errors = initialise_petsc_array(n_squared_errors)

    # Create the solver object.
    tao = PETSc.TAO().create(PETSc.COMM_WORLD)

    # Set the solver type.
    tao.setType("pounders")

    tao.setFromOptions()

    def func_tao(tao, squared_errors, f):
        """Evaluate objective and attach result to an petsc object f.

        This is required to use the pounders solver from tao.

        Args:
             tao: The tao object we created for the optimization task.
             squared_errors (np.ndarray): 1d NumPy array of the current values at which
                we want to evaluate the function.
             f: Petsc object in which we save the current function value.

        """
        f.array = fun(squared_errors.array)

    # Set the procedure for calculating the objective. This part has to be changed if we
    # want more than pounders.
    tao.setResidual(func_tao, squared_errors)

    # We try to set user defined convergence tests.
    if init_tr is not None:
        tao.setInitialTrustRegionRadius(init_tr)

    # Add bounds.
    n_params = len(x0.array)
    processed_bounds = []
    for bound in bounds:
        bound = np.full(n_params, bound) if isinstance(bound, (int, float)) else bound
        processed_bounds.append(initialise_petsc_array(bound))
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

    results = _process_pounders_results(squared_errors, tao)

    # Destroy petsc objects for memory reasons.
    tao.destroy()
    x0.destroy()
    squared_errors.destroy()

    return results


def initialise_petsc_array(len_or_array):
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


def _process_pounders_results(squared_errors, tao):
    results = {
        "fitness": squared_errors.array.sum(),
        "fitness_values": squared_errors.array,
        "x": tao.solution.array,
        "conv": _translate_tao_convergence_reason(tao.getConvergedReason()),
        "n_evaluations": tao.getIterationNumber(),
        "gnorm": tao.gnorm,
        "cnorm": tao.cnorm,
    }

    return results
