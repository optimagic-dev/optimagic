# flake8: noqa
import numpy as np
from scipy.optimize._numdiff import approx_derivative
from scipy.optimize.optimize import _check_unknown_options
from scipy.optimize.optimize import _line_search_wolfe12
from scipy.optimize.optimize import _LineSearchError
from scipy.optimize.optimize import _status_message
from scipy.optimize.optimize import OptimizeResult
from scipy.optimize.optimize import vecnorm
from scipy.optimize.optimize import wrap_function

from estimagic.optimization.scipy import _process_scipy_results


def minimize_bhhh(func, x0, bounds, algo_options=None, gradient=None):
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


def wrap_function_agg(function, args):
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


def wrap_function_num_dev(objective_fun, args):
    """
    Wrap the numerical Jacobian in the BHHH.

    """
    ncalls = [0]

    def function_wrapper(x0):
        ncalls[0] += 1
        return approx_derivative(objective_fun, x0, args=args)

    return ncalls, function_wrapper


def fmin_bhhh(
    fun,
    x0,
    bounds=None,
    fprime=None,
    args=(),
    tol={"abs": 1e-05, "rel": 1e-08},
    norm=np.Inf,
    maxiter=None,
    disp=1,
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
    maxiter : int, optional
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
    jac : ndarray
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
    success : bool
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
    if not isinstance(args, tuple):
        args = (args,)

    opts = {
        "tol": tol,
        "norm": norm,
        "disp": disp,
        "maxiter": maxiter,
        "return_all": retall,
    }

    res = _minimize_bhhh(fun, x0, bounds, args, fprime, callback=callback, **opts)

    return res


def _minimize_bhhh(
    fun,
    x0,
    bounds=None,
    args=(),
    jac=None,
    callback=None,
    tol={"abs": 1e-05, "rel": 1e-08},
    norm=np.Inf,
    maxiter=None,
    disp=False,
    return_all=False,
    **unknown_options
):
    """
    Minimization of a scalar function of one or more variables using the
    BHHH algorithm.

    Options
    -------
    disp : bool
        Set to True to print convergence messages.
    maxiter : int
        Maximum number of iterations to perform.
    tol : dict
        Absolute and relative tolerance values.
    norm : float
        Order of norm (Inf is max, -Inf is min).

    """

    _check_unknown_options(unknown_options)

    f = fun
    fprime = jac
    retall = return_all
    k = 0
    n = len(x0)

    x0 = np.asarray(x0).flatten()
    if x0.ndim == 0:
        x0.shape = (1,)

    if bounds is None:
        bounds = np.array([np.inf] * n * 2).reshape((2, n))
        bounds[0, :] = -bounds[0, :]
    if bounds.shape[1] != n:
        raise ValueError("length of x0 != length of bounds")

    low = bounds[0, :]
    up = bounds[1, :]
    x0 = np.clip(x0, low, up)

    if maxiter is None:
        maxiter = len(x0) * 200

    # Need the aggregate functions to take only x0 as an argument
    func_calls, agg_fun = wrap_function_agg(f, args)

    if not callable(fprime):
        grad_calls, myfprime = wrap_function_num_dev(f, args)
    else:
        grad_calls, myfprime = wrap_function(fprime, args)

    def agg_fprime(x0):
        return myfprime(x0).sum(axis=0)

    # Setup for iteration
    old_fval = agg_fun(x0)

    gf0 = agg_fprime(x0)
    norm_pg0 = vecnorm(x0 - np.clip(x0 - gf0, low, up), ord=norm)

    xk = x0
    norm_pgk = norm_pg0

    if retall:
        allvecs = [x0]
    warnflag = 0

    for _ in range(maxiter):

        # Individual
        gfk_obs = myfprime(xk)

        # Aggregate fprime. Might replace by simply summing up gfk_obs
        gfk = gfk_obs.sum(axis=0)
        norm_pgk = vecnorm(xk - np.clip(xk - gfk, low, up), ord=norm)

        # Check tolerance of gradient norm
        if norm_pgk <= tol["abs"] + tol["rel"] * norm_pg0:
            break

        # Sets the initial step guess to dx ~ 1
        old_old_fval = old_fval + np.linalg.norm(gfk) / 2

        # Calculate BHHH hessian and step
        hk = np.dot(gfk_obs.T, gfk_obs)
        bk = np.linalg.inv(hk)
        pk = np.empty(n)
        pk = -np.dot(bk, gfk)

        try:
            alpha_k, fc, gc, old_fval, old_old_fval, gfkp1 = _line_search_wolfe12(
                agg_fun,
                agg_fprime,
                xk,
                pk,
                gfk,
                old_fval,
                old_old_fval,
                amin=1e-100,
                amax=1e100,
            )
        except _LineSearchError:
            # Line search failed to find a better solution.
            warnflag = 2
            break

        xkp1 = np.clip(xk + alpha_k * pk, low, up)
        if retall:
            allvecs.append(xkp1)
        xk = xkp1
        if callback is not None:
            callback(xk)
        k += 1

        if np.isinf(old_fval):
            # We correctly found +-Inf as optimal value, or something went
            # wrong.
            warnflag = 2
            break

    fval = old_fval

    if warnflag == 2:
        msg = _status_message["pr_loss"]
    elif k >= maxiter:
        warnflag = 1
        msg = _status_message["maxiter"]
    elif np.isnan(fval) or np.isnan(xk).any():
        warnflag = 3
        msg = _status_message["nan"]
    else:
        msg = _status_message["success"]

    if disp:
        print("{}{}".format("Warning: " if warnflag != 0 else "", msg))
        print("         Current function value: %f" % fval)
        print("         Iterations: %d" % k)

    result = OptimizeResult(
        fun=fval,
        jac=gfk,
        hess_inv=bk,
        nfev=func_calls[0],
        njev=grad_calls[0],
        status=warnflag,
        success=(warnflag == 0),
        message=msg,
        x=xk,
        nit=k,
    )
    if retall:
        result["allvecs"] = allvecs
    return result
