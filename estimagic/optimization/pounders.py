"""Wrapper for pounders in the tao package."""
import sys
from functools import partial

if sys.platform != "win32":
    from petsc4py import PETSc


def minimize_pounders(
    internal_criterion, internal_params, criterion, params, algo_options,
):
    """Minimize a function using the pounders algortihm.

    Args:
        internal_criterion (callable): Internal criterion function.
        internal_params (pd.DataFrame): See :ref:`params`.
        criterion (callable): External criterion function.
        params (pd.DataFrame): See :ref:`params`.
        algo_options (dict): Contains options for the algorithm. They are ...
            - init_tr (float): Sets the radius for the initial trust region that the
              optimizer employs. If `None` the algorithm uses 100 as initial  trust
              region radius. Default is `None`.
            - max_iterations (int): Alternative Stopping criterion. If set the routine
              will stop after the number of specified iterations or after the step size
              is sufficiently small. If the variable is set the default criteria will
              all be ignored. Default is `None`.
            - gatol (float): Stop if relative norm of gradient is less than this. If set
              to False the algorithm will not consider gatol. Default is 1e-8.
            - grtol (float): Stop if norm of gradient is less than this. If set to False
              the algorithm will not consider grtol. Default is 1e-8.
            - gttol (float): Stop if norm of gradient is reduced by this factor. If set
              to False the algorithm will not consider grtol. Default is 1e-10.

    Returns:
        out (dict): Dictionary with the following key-value pairs:

            - `"solution"`: solution vector as `np.ndarray`.
            - `"func_values"`: `np.ndarray` of value of the objective at the solution.
            - `"x"`: `np.ndarray` of the start values.
            - `"conv"`: string indicating the termination reason.
            - `"sol"`: `list` containing ...
              - current iterate as integer.
              - current value of the objective as float
              - current value of the approximated
              - jacobian as float
              - infeasability norm as float
              - step length as float
              - termination reason as int.

    """
    if sys.platform == "win32":
        raise NotImplementedError("The pounders algorithm is not available on Windows.")

    # Set defaults for algo_options.
    gatol = algo_options.get("gatol", 1e-8)
    grtol = algo_options.get("grtol", 1e-8)
    gttol = algo_options.get("gttol", 1e-10)
    init_tr = algo_options.get("init_tr", None)
    max_iterations = algo_options.get("max_iterations", None)

    # We need to know the dimension of the output of the criterion function. Evaluate
    # plain `criterion` to prevent logging.
    len_output = algo_options.pop("len_output", None)
    if len_output is None:
        len_output = len(criterion(params))

    # We want to get containers for the func vector and the paras.
    n_params = len(internal_params)
    paras = initialise_petsc_array(n_params)
    crit = initialise_petsc_array(len_output)

    # Create the solver object.
    tao = PETSc.TAO().create(PETSc.COMM_WORLD)

    # Set the solver type.
    tao.setType("pounders")

    tao.setFromOptions()

    def func_tao(tao, crit, f):
        """Evaluate objective and attach result to an petsc object f.

        This is required to use the pounders solver from tao.

        Args:
             tao: The tao object we created for the optimization task.
             crit (np.ndarray): 1d NumPy array of the current values at which we want
                to evaluate the function.
             f: Petsc object in which we save the current function value.

        """
        f.array = internal_criterion(crit.array)

    # Set the procedure for calculating the objective. This part has to be changed if we
    # want more than pounders.
    tao.setResidual(func_tao, crit)

    # We try to set user defined convergence tests.
    if init_tr is not None:
        tao.setInitialTrustRegionRadius(init_tr)

    # Add bounds.
    bounds = params.query("_internal_free")[["lower", "upper"]].to_numpy().T
    low = initialise_petsc_array(n_params)
    up = initialise_petsc_array(n_params)
    low.array = bounds[0]
    up.array = bounds[1]
    tao.setVariableBounds([low, up])

    # Put the starting values into the container and pass them to the optimizer.
    paras.array = internal_params["value"].to_numpy()
    tao.setInitial(paras)

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

    results = _process_pounders_results(crit, tao)

    # Destroy petsc objects for memory reasons.
    tao.destroy()
    paras.destroy()
    crit.destroy()

    return results


def initialise_petsc_array(length):
    array = PETSc.Vec().create(PETSc.COMM_WORLD)
    array.setSizes(length)
    array.setFromOptions()

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


def _process_pounders_results(crit, tao):
    results = {
        "fitness": crit.array.sum(),
        "fitness_values": crit.array,
        "x": tao.solution.array,
        "conv": _translate_tao_convergence_reason(tao.getConvergedReason()),
        "n_evaluations": tao.getIterationNumber(),
        "gnorm": tao.gnorm,
        "cnorm": tao.cnorm,
    }

    return results
