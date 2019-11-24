"""Wrapper for pounders in the tao package."""
import sys
from functools import partial

if sys.platform != "win32":
    from petsc4py import PETSc


def minimize_pounders(
    func,
    x,
    len_out,
    bounds=None,
    init_tr=None,
    max_iterations=None,
    gatol=1e-8,
    grtol=1e-8,
    gttol=1e-10,
):
    """Minimize a function using the pounders algortihm.

    Args:
        func (callable): Function that takes a 1d NumPy array and returns a 1d NumPy
            array.
        x (np.ndarray): Contains the start values of the variables of interest
        bounds (list or tuple of lists): Contains the bounds for the variable of
            interest. The first list contains the lower value for each parameter and the
            upper list the upper value. The object has to contain two elements of which
            one represents the upper and the other one the lower bound.
        init_tr (float): Sets the radius for the initial trust region that the optimizer
            employs. If none the algorithm uses 100 as initial  trust region radius.
        max_iterations (int): Alternative Stopping criterion. If set the routine will
            stop after the number of specified iterations or after the step size is
            sufficiently small. If the variable is set the default criteria will all be
            ignored.
        gatol (int): Stop if relative norm of gradient is less than this. If set to
            False the algorithm will not consider gatol.
        grtol (int): Stop if norm of gradient is less than this. If set to False the
            algorithm will not consider grtol.
        gttol (int): Stop if norm of gradient is reduced by this factor. If set to False
            the algorithm will not consider grtol.

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

    # We want to get containers for the func verctor and the paras.
    size_paras = len(x)
    size_objective = len_out
    paras, crit = _prep_args(size_paras, size_objective)

    # Set the start value.
    paras[:] = x

    def func_tao(tao, paras, f):
        """Evaluate objective and attach result to an petsc object f.

        This is required to use the pounders solver from tao.

        Args:
             tao: The tao object we created for the optimization task.
             paras (np.ndarray): 1d NumPy array of the current values at which we want
                to evaluate the function.
             f: Petsc object in which we save the current function value.
        """
        dev = func(paras.array)
        # Attach to PETSc object.
        f.array = dev

    # Create the solver object.
    tao = PETSc.TAO().create(PETSc.COMM_WORLD)

    # Set the solver type.
    tao.setType("pounders")

    tao.setFromOptions()

    # Set the procedure for calculating the objective. This part has to be changed if we
    # want more than pounders.
    tao.setResidual(func_tao, crit)

    # We try to set user defined convergence tests.
    if init_tr is not None:
        tao.setInitialTrustRegionRadius(init_tr)

    # Change they need to be in a container
    # Set the variable sounds if existing
    if bounds is not None:
        low, up = _prep_args(len(x), len(x))
        low.array = bounds[0]
        up.array = bounds[1]
        tao.setVariableBounds([low, up])

    # Set the container over which we optimize that already contians start values
    tao.setInitial(paras)

    # Obtain tolerances for the convergence criteria. Since we can not create gttol
    # manually we manually set gatol and or grtol to zero once a subset of these two is
    # turned off and gttol is still turned on.
    tol_real = _get_tolerances(gttol, gatol, grtol)

    # Set tolerances for default convergence tests.
    tao.setTolerances(
        gatol=tol_real["gatol"], gttol=tol_real["gttol"], grtol=tol_real["grtol"]
    )

    # Set user defined convergence tests. Beware that specifiying multiple tests could
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

    # Create a dict that contains relevant information.
    out = {}
    out["x"] = paras.array
    out["fun"] = crit.array[-1]
    out["func_values"] = crit.array
    out["start_values"] = x
    out["conv"] = _translate_tao_convergence_reason(tao.getConvergedReason())
    out["sol"] = tao.getSolutionStatus()

    # Destroy petsc objects for memory reasons.
    tao.destroy()
    paras.destroy()
    crit.destroy()

    return out


def _prep_args(size_paras, size_objective):
    """Prepare the arguments for tao.

    Args:
        size_paras: int containing the size of the pram vector
        size_prob: int containing the size of the
    """
    # create container for variable of interest
    paras = PETSc.Vec().create(PETSc.COMM_WORLD)
    paras.setSizes(size_paras)

    # Create container for criterion function
    crit = PETSc.Vec().create(PETSc.COMM_WORLD)
    crit.setSizes(size_objective)

    # Initialize
    crit.setFromOptions()
    paras.setFromOptions()

    return paras, crit


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


def _get_tolerances(gttol, gatol, grtol):
    out = {}
    out["gatol"] = gatol
    out["grtol"] = grtol
    out["gttol"] = gttol
    for x in out.keys():
        if out[x] is False:
            out[x] = -1
    return out


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
