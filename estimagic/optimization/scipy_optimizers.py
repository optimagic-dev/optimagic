import functools

import numpy as np
import scipy

from estimagic.optimization.default_algo_options import ABSOLUTE_CRITERION_TOLERANCE
from estimagic.optimization.default_algo_options import ABSOLUTE_PARAMS_TOLERANCE
from estimagic.optimization.default_algo_options import GRADIENT_TOLERANCE
from estimagic.optimization.default_algo_options import LIMITED_MEMORY_STORAGE_LENGTH
from estimagic.optimization.default_algo_options import MAX_CRITERION_EVALUATIONS
from estimagic.optimization.default_algo_options import MAX_ITERATIONS
from estimagic.optimization.default_algo_options import MAX_LINE_SEARCH_STEPS
from estimagic.optimization.default_algo_options import RELATIVE_CRITERION_TOLERANCE
from estimagic.optimization.default_algo_options import RELATIVE_PARAMS_TOLERANCE


DEFAULT_ALGO_INFO = {
    "primary_criterion_entry": "value",
    "parallelizes": False,
    "needs_scaling": False,
}


def scipy_lbfgsb(
    criterion_and_derivative,
    x,
    lower_bounds=None,
    upper_bounds=None,
    *,
    relative_criterion_tolerance=RELATIVE_CRITERION_TOLERANCE,
    gradient_tolerance=GRADIENT_TOLERANCE,
    max_criterion_evaluations=MAX_CRITERION_EVALUATIONS,
    max_iterations=MAX_ITERATIONS,
    limited_memory_storage_length=LIMITED_MEMORY_STORAGE_LENGTH,
    max_line_search_steps=MAX_LINE_SEARCH_STEPS,
):
    """Minimize a scalar function of one or more variables using the L-BFGS-B algorithm.

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
    and criterion function jointly, the user should provide a `criterion_and_derivative`
    function that exploits the synergies in the calculation of criterion and gradient.

    The lbfgsb algorithm is almost perfectly scale invariant. It is thus not necessary
    to scale the parameters.

    Below only details of the optional algorithm options are listed. For the mandatory
    arguments see :ref:`internal_optimizer_interface`. For more background on those
    options see :ref:`naming_conventions`.

    Args:
        relative_criterion_tolerance (float): Stop when the relative improvement between
            two iterations is smaller than this. More formally, this is expressed as
            ``f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= relative_criterion_tolerance``
        gradient_tolerance (float): Stop if all elements of the projected gradient are
            smaller than this.
        max_criterion_evaluations (int): If the maximum number of function evaluation is
            reached, the optimization stops but we do not count this as convergence.
        max_iterations (int): If the maximum number of iterations is reached, the
            optimization stops, but we do not count this as convergence.
        limited_memory_storage_length (int): Maximum number of saved gradients used to
            approximate the hessian matrix.

    Returns:
        dict: See :ref:`internal_optimizer_output` for details.

    """
    algo_info = DEFAULT_ALGO_INFO.copy()
    algo_info["name"] = "scipy_lbfgsb"
    func = functools.partial(
        criterion_and_derivative,
        task="criterion_and_derivative",
        algorithm_info=algo_info,
    )

    options = {
        "maxcor": limited_memory_storage_length,
        "ftol": relative_criterion_tolerance,
        "gtol": gradient_tolerance,
        "maxfun": max_criterion_evaluations,
        "maxiter": max_iterations,
        "maxls": max_line_search_steps,
    }

    res = scipy.optimize.minimize(
        fun=func,
        x0=x,
        method="L-BFGS-B",
        jac=True,
        bounds=_get_scipy_bounds(lower_bounds, upper_bounds),
        options=options,
    )

    return _process_scipy_result(res)


def scipy_slsqp(
    criterion_and_derivative,
    x,
    lower_bounds=None,
    upper_bounds=None,
    *,
    relative_criterion_tolerance=RELATIVE_CRITERION_TOLERANCE,
    max_iterations=MAX_ITERATIONS,
):
    """Minimize a scalar function of one or more variables using the SLSQP algorithm.

    SLSQP stands for Sequential Least Squares Programming.

    SLSQP is a line search algorithm. It is well suited for continuously
    differentiable scalar optimization problems with up to several hundred parameters.

    The optimizer is taken from scipy which wraps the SLSQP optimization subroutine
    originally implemented by [Dieter Kraft]_.

    Args:
        relative_criterion_tolerance (float): Precision goal for the value of f in the
            stopping criterion.
        max_iterations (int): If the maximum number of iterations is reached, the
            optimization stops, but we do not count this as convergence.

    Returns:
        dict: See :ref:`internal_optimizer_output` for details.

    .. [Dieter Kraft] A software package for sequential quadratic programming. 1988.
            Tech. Rep. DFVLR-FB 88-28, DLR German Aerospace Center –
            Institute for Flight Mechanics, Koln, Germany.
            http://degenerateconic.com/wp-content/uploads/2018/03/DFVLR_FB_88_28.pdf

    """
    algo_info = DEFAULT_ALGO_INFO.copy()
    algo_info["name"] = "scipy_slsqp"

    func = functools.partial(
        criterion_and_derivative, task="criterion", algorithm_info=algo_info,
    )

    gradient = functools.partial(
        criterion_and_derivative, task="derivative", algorithm_info=algo_info
    )

    options = {
        "maxiter": max_iterations,
        "ftol": relative_criterion_tolerance,
    }

    res = scipy.optimize.minimize(
        fun=func,
        x0=x,
        method="SLSQP",
        jac=gradient,
        bounds=_get_scipy_bounds(lower_bounds, upper_bounds),
        options=options,
    )

    return _process_scipy_result(res)


def scipy_neldermead(
    criterion_and_derivative,
    x,
    *,
    max_iterations=MAX_ITERATIONS,
    max_criterion_evaluations=MAX_CRITERION_EVALUATIONS,
    absolute_params_tolerance=ABSOLUTE_PARAMS_TOLERANCE + 1e-05,
    absolute_criterion_tolerance=ABSOLUTE_CRITERION_TOLERANCE + 1e-05,
):
    """Minimize a scalar function using the Nelder-Mead algorithm.

    The Nelder-Mead algorithm is a direct search method (based on function comparison)
    and is often applied to nonlinear optimization problems for which derivatives are
    not known.
    Unlike modern optimization methods, the Nelder–Mead heuristic can converge to a
    non-stationary point, unless the problem satisfies stronger conditions than are
    necessary for modern methods.

    Args:
        max_iterations (int): If the maximum number of iterations is reached, the
            optimization stops, but we do not count this as convergence.
        max_criterion_evaluations (int): If the maximum number of function evaluation is
            reached, the optimization stops but we do not count this as convergence.
        absolute_params_tolerance (float): Absolute difference in parameters between
            iterations that is tolerated to declare convergence.
        absolute_criterion_tolerance (float): Absolute difference in the criterion value
            between iterations that is tolerated to declare convergence.

    Returns:
        dict: See :ref:`internal_optimizer_output` for details.

    """
    algo_info = DEFAULT_ALGO_INFO.copy()
    algo_info["name"] = "scipy_neldermead"
    func = functools.partial(
        criterion_and_derivative, task="criterion", algorithm_info=algo_info,
    )
    options = {
        "maxiter": max_iterations,
        "maxfev": max_criterion_evaluations,
        "xatol": absolute_params_tolerance,
        "fatol": absolute_criterion_tolerance,
    }

    res = scipy.optimize.minimize(
        fun=func, x0=x, method="Nelder-Mead", options=options,
    )

    return _process_scipy_result(res)


def scipy_powell(
    criterion_and_derivative,
    x,
    lower_bounds=None,
    upper_bounds=None,
    *,
    relative_params_tolerance=RELATIVE_PARAMS_TOLERANCE,
    relative_criterion_tolerance=RELATIVE_CRITERION_TOLERANCE,
    max_criterion_evaluations=MAX_CRITERION_EVALUATIONS,
    max_iterations=MAX_ITERATIONS,
):
    """Minimize a scalar function of using the modified Powell method.

    The criterion function need not be differentiable.

    Powell's method is a conjugate direction method, minimising the function by a
    bi-directional search in each parameter's dimension.

    Args:
        relative_params_tolerance (float): Stop when the relative movement between
            parameter vectors is smaller than this.
        relative_criterion_tolerance (float): Stop when the relative improvement between
            two iterations is smaller than this. More formally, this is expressed as
            ``f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= relative_criterion_tolerance``
        max_criterion_evaluations (int): If the maximum number of function evaluation is
            reached, the optimization stops but we do not count this as convergence.
        max_iterations (int): If the maximum number of iterations is reached, the
            optimization stops, but we do not count this as convergence.

    Returns:
        dict: See :ref:`internal_optimizer_output` for details.

    """

    algo_info = DEFAULT_ALGO_INFO.copy()
    algo_info["name"] = "scipy_powell"
    func = functools.partial(
        criterion_and_derivative, task="criterion", algorithm_info=algo_info,
    )

    options = {
        "xtol": relative_params_tolerance,
        "ftol": relative_criterion_tolerance,
        "maxiter": max_iterations,
        "maxfev": max_criterion_evaluations,
    }

    res = scipy.optimize.minimize(
        fun=func,
        x0=x,
        method="Powell",
        bounds=_get_scipy_bounds(lower_bounds, upper_bounds),
        options=options,
    )

    return _process_scipy_result(res)


# =====================================================================================


def _process_scipy_result(scipy_results_obj):
    # using get with defaults to access dict elements is just a safety measure
    raw_res = {**scipy_results_obj}
    processed = {
        "solution_x": raw_res.get("x", None),
        "solution_criterion": raw_res.get("fun", None),
        "solution_derivative": raw_res.get("jac", None),
        "solution_hessian": raw_res.get("hess", None),
        "n_criterion_evaluations": raw_res.get("nfev", None),
        "n_derivative_evaluations": raw_res.get("njac", None),
        "n_iterations": raw_res.get("nit", None),
        "success": raw_res.get("success", None),
        "reached_convergence_criterion": None,
        "message": raw_res.get("message", None),
    }
    return processed


def _get_scipy_bounds(lower_bounds, upper_bounds):
    # Scipy works with `None` instead of infinite values for unconstrained parameters
    # and requires a list of tuples for each parameter with lower and upper bound.
    bounds = np.column_stack([lower_bounds, upper_bounds])
    mask = ~np.isfinite(bounds)
    bounds = bounds.astype("object")
    bounds[mask] = None
    return list(map(tuple, bounds))
