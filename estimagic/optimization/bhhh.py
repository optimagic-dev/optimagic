# flake8: noqa
import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize._numdiff import approx_derivative
from scipy.optimize.optimize import _check_unknown_options
from scipy.optimize.optimize import _line_search_wolfe12
from scipy.optimize.optimize import _LineSearchError
from scipy.optimize.optimize import vecnorm
from scipy.optimize.optimize import wrap_function


def wrap_function_aggregate(function, args):
    """
    Sum the outputs of a vector-valued function fixed at specific values of
    arguments.

    Parameters
    ----------
    function : callable function(x, *args)
        A vector-valued function giving out a numpy.array having arguments x
        as flexible arguments.
    args : tuple
        Addtional arguments passed to function. The function will be evaluated
        at those fixed arguments.

    Returns
    -------
    ncalls : int
        Number of times the wrap function has been called.
    function_wrapper : callable function_wrapper(x)
        Function return the value of function(x, *args)

    """
    ncalls = [0]

    def function_wrapper(*wrapper_args):
        ncalls[0] += 1
        return function(*(wrapper_args + args)).sum()

    return ncalls, function_wrapper


def wrap_function_num_derivative(function, args):
    """
    Return the numerical Jacobian of a function at given values for certain
    arguments (args).

    Parameters
    ----------
    function : callable function(x, *args)
        Vector-valued function that returns a numpy.array.
    args : tuple
        tuple of arguments at which the Jacobian is supposed to be numerically
        approximated.

    Returns
    -------
    ncalls : int
        Number of times the wrap function has been called.
    function_wrapper : callable function_wrapper(x)
        Function that gives out the numerical Jacobian at the specified level
        of x and the pre set level of args.

    """
    ncalls = [0]

    def function_wrapper(x0):
        ncalls[0] += 1
        return approx_derivative(function, x0, args=args)

    return ncalls, function_wrapper


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
    Minimize the objective function func(x, *args) using the the quasi-Newton
    method and the Berndt, Hall, Hall and Hubert (BHHH) algorithm to approximate
    the Hessian.
    This method works only on objective functions that take the general form
    f = sum(f_i).

    Parameters
    ----------
    func : callable func(x, *args)
        Objective function that returns a numpy.array corresponding to
        the individual values of f_i. The numpy.array, hence, has the length
        corresponding to the amount of individual summands f_i.
    x0 : numpy.array
        Starting values of the parameters.
    bounds : Tuple[np.ndarray], optional
        A tuple containing two NumPy arrays where the first
        corresponds to the lower and the second to the upper bounds. Unbounded
        parameters are represented by None. The arrays have the same
        length as the parameter vector.
    args : tuple, optional
        Extra arguments passed to the objective function and its derivative.
    jacobian : callable func'(x, *args), optional
        Jacobian of the objective function. If None is given then the jacobian
        will be numerically approximated using scipy.
    callback : callable callback(xk), optional
        An optional function to call after each iteration.
        xk is the current guess of the parameter vector.
    tol : dict, optional
        Dictionairy that contains the absolute and relative stopping tolerance.
        The Form is tol = {"abs" : x, "rel" : y}. Both parameters must be
        strictly positive and relative tolerance must be bigger equal to the
        absolute tolerance. The default is {"abs": 1e-05, "rel": 1e-08}.
    norm : float, optional
        Order of norm (Inf is max, -Inf is min). Default is Inf.
    max_iterations : int, optional
        The maximum number of iterations. The default is None.

    Returns
    -------
    results : dict
        Return a dictionairy containing the following keys.
        x : numpy.array
            The parameter vector found by the algorithm.
        status : bool
            True if the algorithm conerved.
        message : str
            Print a message describing that algorithm converged and if not,
            why it did not converge.
        n_iterations : int
            State the number of iterations the algorithm needed.
        fitness : float
            The value of the aggregated objective function at x.
        n_evaluations : int
            Number of calls of the objective function.
        n_evaluations_jacobian : int
            Number of calls of the jacobian function.
        jacobian : numpy.array
            Value of the aggregate Jacobian at x.
        hessian : numpy.array
            Value of the Hessian matrix at x.
        hessian_inverse : numpy.array
            Value of the inverse Hessian at x.

    References
    ----------
    Berndt, E.; Hall, B.; Hall, R.; Hausman, J. (1974). "Estimation and
    Inference in Nonlinear Structural Models". Annals of Economic and Social
    Measurement. 3 (4): 653–665.
    Buchwald, S. "Implementierung des L-BFGS-B-Verfahrens in Python".
    Bachelor-Thesis University of Konstanz.

    """
    # process inputs
    _check_unknown_options(unknown_options)

    if not isinstance(args, tuple):
        args = (args,)

    num_iterations = 0
    num_params = len(x0)
    status = True

    if bounds is None:
        bounds = np.array([np.inf] * num_params * 2).reshape((2, num_params))
        bounds[0, :] = -bounds[0, :]
    else:
        bounds = np.array(bounds)

    lower_bounds = bounds[0, :]
    upper_bounds = bounds[1, :]
    x0 = np.clip(x0, lower_bounds, upper_bounds)

    if max_iterations is None:
        max_iterations = num_params * 200

    func_calls, aggregate_func = wrap_function_aggregate(func, args)

    if not callable(jacobian):
        grad_calls, jacobian_wrapped = wrap_function_num_derivative(func, args)
    else:
        grad_calls, jacobian_wrapped = wrap_function(jacobian, args)

    def aggregate_jacobian_wrapped(x0):
        return jacobian_wrapped(x0).sum(axis=0)

    # Setup for iteration
    func_value = aggregate_func(x0)

    jacobian_value = jacobian_wrapped(x0)
    agg_jacobian_value = jacobian_value.sum(axis=0)
    norm_start = vecnorm(
        x0 - np.clip(x0 - agg_jacobian_value, lower_bounds, upper_bounds), ord=norm,
    )

    xk = x0

    # Set the initial step guess to dx ~ 1
    old_func_value = func_value + np.linalg.norm(agg_jacobian_value) / 2

    for iteration in range(max_iterations):

        if iteration != 0:
            # Jacobian at current guess
            jacobian_value = jacobian_wrapped(xk)

            # Aggregate Jacobian at current guess
            agg_jacobian_value = jacobian_value.sum(axis=0)

        norm_current = vecnorm(
            xk - np.clip(xk - agg_jacobian_value, lower_bounds, upper_bounds), ord=norm
        )

        # Calculate BHHH hessian and the initial step size
        hessian = np.dot(jacobian_value.T, jacobian_value)

        # get numerically stable inverse
        lower_triangle = np.linalg.cholesky(hessian)
        lower_triangle_inverse = solve_triangular(
            lower_triangle, np.eye(hessian.shape[0]), lower=True
        )
        hessian_inverse = np.dot(lower_triangle_inverse.T, lower_triangle_inverse)

        step = -np.dot(hessian_inverse, agg_jacobian_value)

        # Check tolerance of gradient norm
        if norm_current <= tol["abs"] + tol["rel"] * norm_start:
            break
        # Start line searching
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

        # Make sure that the new guess is within bounds
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
    results["n_evaluations"] = func_calls[0]
    results["n_evaluations_jacobian"] = grad_calls[0]
    results["jacobian"] = agg_jacobian_value
    results["hessian"] = hessian
    results["hessian_inverse"] = hessian_inverse
    results["x"] = xk

    return results
