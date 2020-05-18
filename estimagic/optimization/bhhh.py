# flake8: noqa
import numpy as np
from scipy.optimize._numdiff import approx_derivative
from scipy.optimize.optimize import _check_unknown_options
from scipy.optimize.optimize import _line_search_wolfe12
from scipy.optimize.optimize import _LineSearchError
from scipy.optimize.optimize import vecnorm
from scipy.optimize.optimize import wrap_function

from estimagic.optimization.scipy import _process_scipy_results


def _minimize_bhhh(func, x0, bounds, algo_options=None, gradient=None):
    """
    Interface for BHHH.
    Args:
        func (callable): Objective function.
        x0 (np.ndarray): Starting values of the parameters.
        bounds (Tuple[np.ndarray]): A tuple containing two NumPy arrays where the first
            corresponds to the lower and the second to the upper bound. Unbounded
            parameters are represented by infinite values. The arrays have the same
            length as the parameter vector.
        algo_name (str): One of the optimizers of the scipy package which receives the
            same inputs as the ``"method"`` keyword of the original function.
        algo_options (dict): Options for the optimizer.
        gradient (callable): Gradient function.
    Returns:
        results (dict): Dictionary with processed optimization results.
    """

    bounds = np.array(bounds)

    if algo_options is None:
        scipy_results_obj = fmin_bhhh(func, x0, bounds=bounds, fprime=gradient)
    else:
        scipy_results_obj = fmin_bhhh(
            func, x0, bounds=bounds, fprime=gradient, **algo_options
        )

    results = _process_scipy_results(scipy_results_obj)

    return results


def wrap_function_aggregate(function, args):
    """
    Wrap the objective function in the BHHH.

    """
    ncalls = [0]
    if function is None:
        return ncalls, None

    def function_wrapper(*wrapper_args):
        ncalls[0] += 1
        return function(*(wrapper_args + args)).sum()

    return ncalls, function_wrapper


def wrap_function_num_derivative(function, args):
    """
    Wrap the numerical Jacobian in the BHHH.

    """
    ncalls = [0]

    def function_wrapper(x0):
        ncalls[0] += 1
        return approx_derivative(function, x0, args=args)

    return ncalls, function_wrapper


def fmin_bhhh(
    fun,
    x0,
    bounds=None,
    fprime=None,
    args=(),
    tol={"abs": 1e-05, "rel": 1e-08},
    norm=np.Inf,
    max_iterations=None,
    retall=False,
    callback=None,
):
    """
    Minimize a function using the BHHH algorithm.

    Parameters
    ----------
    fun : callable fun(x, data, *args, **kwargs)
        Objective function to be minimized.
    x0 : ndarray
        Initial guess.
    bounds : ndarray, optional
        ``(min, max)`` pairs for each element along the rows``x``, defining
        the bounds on that parameter. Use +-inf for one of ``min`` or
        ``max`` when there is no bound in that direction.
    fprime : callable f'(x,*args), optional
        Gradient of f.
    args : tuple, optional
        Extra arguments passed to f and fprime.
    tol : dict, optional
        Dict that contains the absolute and relative tolerance parameters.
        Form should be tol = {"abs" : x, "rel" : y}. Both parameters must be
        strictly positive and relative tolerance must be bigger equal to the
        absolute tolerance.
    norm : float, optional
        Order of norm (Inf is max, -Inf is min)
    callback : callable, optional
        An optional user-supplied function to call after each
        iteration.  Called as callback(xk), where xk is the
        current parameter vector.
    max_iterations : int, optional
        Maximum number of iterations to perform.
    disp : bool, optional
        Print convergence message if True.
    retall : bool, optional
        Return a list of results at each iteration if True.

    Returns
    -------
    x : ndarray
        Parameters which minimize f, i.e. f(xopt) == fopt.
    fun : float
        Minimum value.
    jacobian : ndarray
        Value of gradient at minimum, f'(xopt), which should be near 0.
    hess_inv : ndarray
        Value of 1/f''(xopt), i.e. the inverse hessian matrix.
    nfev : int
        Number of function_calls made.
    njev : int
        Number of gradient calls made.
    warnflag : integer
        1 : Maximum number of iterations exceeded.
        2 : Gradient and/or function calls not changing.
        3 : NaN result encountered.
    message : str
        Status message on whether the optimizer converged.
    status : bool
        Indicating whether optimizer converged.
    allvecs  :  list
        The value of x at each iteration.  Only returned if retall is True.

    Notes
    -----
    Optimize the function, f, whose gradient is given by fprime
    using the quasi-Newton method of Berndt, Hall, Hall,
    and Hubert (BHHH). Box constraints are implemented by using a simple
    gradient approach at each step to identify active and inactive variables.
    The standard BHHH approach is then used on the inactive subset.

    References
    ----------
    Berndt, E.; Hall, B.; Hall, R.; Hausman, J. (1974). "Estimation and
    Inference in Nonlinear Structural Models". Annals of Economic and Social
    Measurement. 3 (4): 653–665.
    Buchwald, S. "Implementierung des L-BFGS-B-Verfahrens in Python".
    Bachelor-Thesis University of Konstanz.

    """

    opts = {
        "tol": tol,
        "norm": norm,
        "max_iterations": max_iterations,
        "return_all": retall,
    }

    res = _minimize_bhhh(fun, x0, bounds, args, fprime, callback=callback, **opts)

    return res


def minimize_bhhh(
    func,
    x0,
    bounds=None,
    args=(),
    jacobian=None,
    callback=None,
    tol={"abs": 1e-05, "rel": 1e-08},
    norm=np.Inf,
    max_iterations=None,
    **unknown_options
):
    """
    Minimization of a scalar function of one or more variables using the
    BHHH algorithm.

    Options
    -------
    disp : bool
        Set to True to print convergence messages.
    max_iterations : int
        Maximum number of iterations to perform.
    tol : dict
        Absolute and relative tolerance values.
    norm : float
        Order of norm (Inf is max, -Inf is min).

    """

    _check_unknown_options(unknown_options)

    if not isinstance(args, tuple):
        args = (args,)

    num_iterations = 0
    num_params = len(x0)
    status = True
    # Probably taken care of by estimagic...
    # x0 = np.asarray(x0).flatten()
    # if x0.ndim == 0:
    #     x0.shape = (1,)

    bounds = np.array(bounds)
    lower_bounds = bounds[0, :]
    upper_bounds = bounds[1, :]
    x0 = np.clip(x0, lower_bounds, upper_bounds)

    if max_iterations is None:
        max_iterations = num_params * 200

    # Need the aggregate function to take only x0 as an argument
    func_calls, aggregate_func = wrap_function_aggregate(func, args)

    if not callable(jacobian):
        grad_calls, jacobian_wrapped = wrap_function_num_derivative(func, args)
    else:
        grad_calls, jacobian_wrapped = wrap_function(jacobian, args)

    def aggregate_jacobian_wrapped(x0):
        return jacobian_wrapped(x0).sum(axis=0)

    # Setup for iteration
    func_value = aggregate_func(x0)

    agg_jacobian_start_value = aggregate_jacobian_wrapped(x0)
    norm_start = vecnorm(
        x0 - np.clip(x0 - agg_jacobian_start_value, lower_bounds, upper_bounds),
        ord=norm,
    )
    # Set the initial step guess to dx ~ 1
    old_func_value = func_value + np.linalg.norm(agg_jacobian_start_value) / 2

    xk = x0

    for _ in range(max_iterations):

        # Individual
        jacobian_value = jacobian_wrapped(xk)

        # Aggregate fprime.
        agg_jacobian_value = jacobian_value.sum(axis=0)
        norm_current = vecnorm(
            xk - np.clip(xk - agg_jacobian_value, lower_bounds, upper_bounds), ord=norm
        )

        # Check tolerance of gradient norm
        if norm_current <= tol["abs"] + tol["rel"] * norm_start:
            break

        # Calculate BHHH hessian and step
        hessian = np.dot(jacobian_value.T, jacobian_value)
        hessian_inverse = np.linalg.inv(hessian)
        step = -np.dot(hessian_inverse, agg_jacobian_value)

        try:
            alpha, fc, gc, func_value, old_func_value, gfkp1 = _line_search_wolfe12(
                aggregate_func,
                aggregate_jacobian_wrapped,
                xk,
                step,
                agg_jacobian_value,
                func_value,
                old_func_value,
                amin=1e-100,
                amax=1e100,
            )
        except _LineSearchError:
            message = "Line search algorithm failed to find a better solution."
            status = False
            break

        xk = np.clip(xk + alpha * step, lower_bounds, upper_bounds)

        if callback is not None:
            callback(xk)
        num_iterations += 1

    if status is True:
        if num_iterations >= max_iterations:
            message = "Maximum number of iterations has been exceeded."
            status = False
        elif np.isnan(func_value) or np.isnan(xk).any():
            message = "NaN result encountered."
            status = False
        else:
            message = "Optimization terminated successfully."

    results = {}
    results["status"] = status
    results["message"] = message
    results["n_iterations"] = num_iterations
    results["fitness"] = func_value
    results["n_evaluations"] = func_calls
    results["n_evaluations_jacobian"] = grad_calls
    results["jacobian"] = agg_jacobian_value
    results["hessian"] = hessian
    results["hessian_inverse"] = hessian_inverse
    results["x"] = xk

    return results
