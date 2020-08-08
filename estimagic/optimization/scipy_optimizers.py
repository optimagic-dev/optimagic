"""Implement scipy's optimizers.

Some arguments cannot be passed as `algo_option` to estimagic's scipy wrappers:
    - disp:
        If set to True would print a convergence message.
        In estimagic it's always set to its default: False.
        Refer to estimagic's result dictionary's "success" entry for the convergence
        message.
    - return_all:
        If set to True, a list of the best solution at each iteration is returned.
        In estimagic it's always set to its default: False.
        Use estimagic's database and dashboard instead to explore your criterion and
        algorithm.
    - tol:
        This argument of minimize (not an options key) is passed as different types of
        tolerance (gradient, parameter or criterion, as well as relative or absolute)
        depending on the selected algorithm. We require the user to explicitely input
        the tolerance criteria or use our defaults instead.
    - args:
        This argument of minimize (not an options key) is partialed into the function
        for the user. Specify `criterion_kwargs` in `maximize` or `minimize` to have
        the same behavior.

Scipy's COBYLA, SLSQP and trust-constr support constraints in principle. However, for
the moment they are not supported.

estimagic does not support the following algorithms at the moment because they require
the specification of the Hessian:
    - dogleg
    - trust-ncg
    - trust-exact
    - trust-krylov

"""
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
    callback=None,
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
        callback (callable): Called after each iteration.

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
        callback=callback,
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
    callback=None,
    constraints=None,
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
        callback (callable): Called after each iteration.
        constraints: not supported at the moment. Constraints can only be passed
            directly as constraints argument to estimagic's `maximize` or `minimize`.
            See :ref:`implementation_of_constraints` for details.

    Returns:
        dict: See :ref:`internal_optimizer_output` for details.

    .. [Dieter Kraft] A software package for sequential quadratic programming. 1988.
            Tech. Rep. DFVLR-FB 88-28, DLR German Aerospace Center –
            Institute for Flight Mechanics, Koln, Germany.
            http://degenerateconic.com/wp-content/uploads/2018/03/DFVLR_FB_88_28.pdf

    """
    if constraints is not None:
        raise NotImplementedError(
            "Constraints passed to SLSQP directly are not supported yet. "
            + "Please use estimagic's constraint interface for constraints supported "
            + "by estimagic."
        )

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
        callback=callback,
    )

    return _process_scipy_result(res)


def scipy_neldermead(
    criterion_and_derivative,
    x,
    *,
    max_iterations=MAX_ITERATIONS,
    max_criterion_evaluations=MAX_CRITERION_EVALUATIONS,
    absolute_params_tolerance=ABSOLUTE_PARAMS_TOLERANCE,
    absolute_criterion_tolerance=ABSOLUTE_CRITERION_TOLERANCE,
    adaptive=False,
    callback=None,
):
    """Minimize a scalar function using the Nelder-Mead algorithm.

    The Nelder-Mead algorithm is a direct search method (based on function comparison)
    and is often applied to nonlinear optimization problems for which derivatives are
    not known.
    Unlike modern optimization methods, the Nelder–Mead heuristic can converge to a
    non-stationary point, unless the problem satisfies stronger conditions than are
    necessary for modern methods.

    The argument initial_simplex is not supported by estimagic as the internal criterion
    is passed by estimagic to `scipy_neldermead` and a user supplied initial simplex
    would in most cases not conform to the internal problem.

    Args:
        max_iterations (int): If the maximum number of iterations is reached, the
            optimization stops, but we do not count this as convergence.
        max_criterion_evaluations (int): If the maximum number of function evaluation is
            reached, the optimization stops but we do not count this as convergence.
        absolute_params_tolerance (float): Absolute difference in parameters between
            iterations that is tolerated to declare convergence.
        absolute_criterion_tolerance (float): Absolute difference in the criterion value
            between iterations that is tolerated to declare convergence.
        adaptive (bool): Adapt algorithm parameters to dimensionality of problem.
            Useful for high-dimensional minimization [1]. Default is False.
        callback (callable): Called after each iteration.

    Returns:
        dict: See :ref:`internal_optimizer_output` for details.

    """
    assert absolute_params_tolerance != 0 or absolute_criterion_tolerance != 0, (
        "Provide either absolute_params_tolerance or absolute_criterion_tolerance as "
        + "stopping criterion."
    )

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
        fun=func, x0=x, method="Nelder-Mead", options=options, callback=callback
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
    callback=None,
):
    """Minimize a scalar function using the modified Powell method.

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
        callback (callable): Called after each iteration.

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
        callback=callback,
    )

    return _process_scipy_result(res)


def scipy_bfgs(
    criterion_and_derivative,
    x,
    lower_bounds=None,
    upper_bounds=None,
    *,
    gradient_tolerance=GRADIENT_TOLERANCE,
    max_iterations=MAX_ITERATIONS,
    callback=None,
):
    """Minimize a scalar function of one or more variables using the BFGS algorithm.

    BFGS stands for Broyden-Fletcher-Goldfarb-Shanno algorithm. It is a quasi-Newton
    method that can be used for solving unconstrained nonlinear optimization problems.

    BFGS is not guaranteed to converge unless the function has a quadratic Taylor
    expansion near an optimum. However, BFGS can have acceptable performance even
    for non-smooth optimization instances.

    Args:
        gradient_tolerance (float): Stop if all elements of the projected gradient are
            smaller than this.
        max_iterations (int): If the maximum number of iterations is reached, the
            optimization stops, but we do not count this as convergence.
        callback (callable): Called after each iteration.

    Returns:
        dict: See :ref:`internal_optimizer_output` for details.

    """
    algo_info = DEFAULT_ALGO_INFO.copy()
    algo_info["name"] = "scipy_bfgs"
    func = functools.partial(
        criterion_and_derivative, task="criterion", algorithm_info=algo_info,
    )
    gradient = functools.partial(
        criterion_and_derivative, task="derivative", algorithm_info=algo_info
    )

    options = {
        "gtol": gradient_tolerance,
        "maxiter": max_iterations,
    }

    res = scipy.optimize.minimize(
        fun=func, x0=x, method="BFGS", jac=gradient, options=options, callback=callback,
    )

    return _process_scipy_result(res)


def scipy_conjugate_gradient(
    criterion_and_derivative,
    x,
    *,
    gradient_tolerance=GRADIENT_TOLERANCE,
    max_iterations=MAX_ITERATIONS,
    callback=None,
):
    """Minimize a function using a nonlinear conjugate gradient algorithm.

    The conjugate gradient method finds functions' local optima using just the gradient.

    Args:
        gradient_tolerance (float): Stop if all elements of the projected gradient are
            smaller than this.
        max_iterations (int): If the maximum number of iterations is reached, the
            optimization stops, but we do not count this as convergence.

    Returns:
        dict: See :ref:`internal_optimizer_output` for details.

    """
    algo_info = DEFAULT_ALGO_INFO.copy()
    algo_info["name"] = "scipy_conjugate_gradient"
    func = functools.partial(
        criterion_and_derivative, task="criterion", algorithm_info=algo_info,
    )

    gradient = functools.partial(
        criterion_and_derivative, task="derivative", algorithm_info=algo_info
    )

    options = {
        "gtol": gradient_tolerance,
        "maxiter": max_iterations,
    }

    res = scipy.optimize.minimize(
        fun=func, x0=x, method="CG", jac=gradient, options=options, callback=callback,
    )

    return _process_scipy_result(res)


def scipy_newton_cg(
    criterion_and_derivative,
    x,
    *,
    relative_params_tolerance=RELATIVE_PARAMS_TOLERANCE,
    max_iterations=MAX_ITERATIONS,
    callback=None,
):
    """Minimize a scalar function using Newton's conjugate gradient algorithm.

    Newton's conjugate gradient algorithm uses an approximation of the Hessian to find
    the minimum of a function. It is practical for small and large problems.

    Newton-CG methods are also called truncated Newton methods. This function differs
    scipy_truncated_newton because

    1. scipy_newton_cg's algorithm is written purely in Python using NumPy
        and scipy while scipy_truncated_newton's algorithm calls a C function.
    2. scipy_newton_cg's algorithm is only for unconstrained minimization
        while scipy_truncated_newton's algorithm supports bounds.

    Reference:
        Wright & Nocedal, 'Numerical Optimization', 1999, p. 140.

    Args:
        relative_params_tolerance (float): Stop when the relative movement between
            parameter vectors is smaller than this.
        max_iterations (int): If the maximum number of iterations is reached, the
            optimization stops, but we do not count this as convergence.
        callback (callable): Called after each iteration.

    Returns:
        dict: See :ref:`internal_optimizer_output` for details.

    """
    algo_info = DEFAULT_ALGO_INFO.copy()
    algo_info["name"] = "scipy_newton_cg"
    func = functools.partial(
        criterion_and_derivative, task="criterion", algorithm_info=algo_info,
    )
    gradient = functools.partial(
        criterion_and_derivative, task="derivative", algorithm_info=algo_info
    )

    options = {
        "xtol": relative_params_tolerance,
        "maxiter": max_iterations,
    }

    res = scipy.optimize.minimize(
        fun=func,
        x0=x,
        method="Newton-CG",
        jac=gradient,
        options=options,
        callback=callback,
    )

    return _process_scipy_result(res)


def scipy_cobyla(
    criterion_and_derivative,
    x,
    *,
    max_iterations=MAX_ITERATIONS,
    callback=None,
    constraints=None,
):
    """Minimize a scalar function of one or more variables using the COBYLA algorithm.

    COBYLA stands for Constrained Optimization By Linear Approximation.
    It is deriviative-free and supports nonlinear inequality and equality constraints.

    .. note::
        Constraints are not supported yet.

    Scipy's implementation wraps the FORTRAN implementation of the algorithm.

    References:
        Powell M.J.D. (1994), “A direct search optimization method that models the
            objective and constraint functions by linear interpolation.”, in Advances in
            Optimization and Numerical Analysis, eds. S. Gomez and J-P Hennart, Kluwer
            Academic (Dordrecht), pp. 51-67

        Powell M.J.D. (1998), “Direct search algorithms for optimization calculations”,
            Acta Numerica 7, 287-336

        Powell M.J.D. (2007),
            “A view of algorithms for optimization without derivatives”,
            Cambridge University Technical Report DAMTP 2007/NA03

    Args:
        max_iterations (int): If the maximum number of iterations is reached, the
            optimization stops, but we do not count this as convergence.
        callback (callable): Called after each iteration.
        constraints: not supported at the moment. Constraints can only be passed
            directly as constraints argument to estimagic's `maximize` or `minimize`.
            See :ref:`implementation_of_constraints` for details.

    Returns:
        dict: See :ref:`internal_optimizer_output` for details.

    """
    if constraints is not None:
        raise NotImplementedError(
            "Constraints passed to COBYLA directly are not supported yet. "
            + "Please use estimagic's constraint interface for constraints supported "
            + "by estimagic."
        )
    algo_info = DEFAULT_ALGO_INFO.copy()
    algo_info["name"] = "scipy_cobyla"

    func = functools.partial(
        criterion_and_derivative, task="criterion", algorithm_info=algo_info,
    )

    options = {"maxiter": max_iterations}

    res = scipy.optimize.minimize(
        fun=func, x0=x, method="COBYLA", options=options, callback=callback
    )

    return _process_scipy_result(res)


def scipy_truncated_newton(
    criterion_and_derivative,
    x,
    lower_bounds=None,
    upper_bounds=None,
    *,
    max_criterion_evaluations=MAX_CRITERION_EVALUATIONS,
    max_iterations=MAX_ITERATIONS,
    absolute_criterion_tolerance=ABSOLUTE_CRITERION_TOLERANCE,
    absolute_params_tolerance=ABSOLUTE_PARAMS_TOLERANCE,
    gradient_tolerance=GRADIENT_TOLERANCE,
    callback=None,
):
    """Minimize a scalar function using truncated Newton algorithm.

    This function differs from scipy_newton_cg because

    1. scipy_newton_cg's algorithm is written purely in Python using NumPy
        and scipy while scipy_truncated_newton's algorithm calls a C function.
    2. scipy_newton_cg's algorithm is only for unconstrained minimization
        while scipy_truncated_newton's algorithm supports bounds.

    Args:
        max_iterations (int): If the maximum number of iterations is reached, the
            optimization stops, but we do not count this as convergence.
        max_criterion_evaluations (int): If the maximum number of function evaluation is
            reached, the optimization stops but we do not count this as convergence.
        absolute_params_tolerance (float): Absolute difference in parameters between
            iterations after scaling that is tolerated to declare convergence.
        absolute_criterion_tolerance (float): Absolute difference in the criterion value
            between iterations after scaling that is tolerated to declare convergence.
        gradient_tolerance (float): Stop if the value of the projected gradient
            (after applying x scaling factors) is smaller than this. If
            gradient_tolerance < 0.0, gtol is set to 1e-2 * sqrt(accuracy).
        callback (callable): Called after each iteration.

    Returns:
        dict: See :ref:`internal_optimizer_output` for details.

    """
    assert absolute_params_tolerance != 0 or absolute_criterion_tolerance != 0, (
        "Provide either absolute_params_tolerance or absolute_criterion_tolerance as "
        + "stopping criterion."
    )

    algo_info = DEFAULT_ALGO_INFO.copy()
    algo_info["name"] = "scipy_truncated_newton"
    func = functools.partial(
        criterion_and_derivative, task="criterion", algorithm_info=algo_info,
    )
    gradient = functools.partial(
        criterion_and_derivative, task="derivative", algorithm_info=algo_info
    )

    options = {
        "maxiter": max_iterations,
        "ftol": absolute_criterion_tolerance,
        "xtol": absolute_params_tolerance,
        "gtol": gradient_tolerance,
        "maxfun": max_criterion_evaluations,
    }

    res = scipy.optimize.minimize(
        fun=func,
        x0=x,
        method="TNC",
        jac=gradient,
        options=options,
        callback=callback,
        bounds=_get_scipy_bounds(lower_bounds, upper_bounds),
    )

    return _process_scipy_result(res)


def scipy_trust_constr(
    criterion_and_derivative,
    x,
    lower_bounds=None,
    upper_bounds=None,
    *,
    gradient_tolerance=GRADIENT_TOLERANCE,
    max_iterations=MAX_ITERATIONS,
    relative_params_tolerance=RELATIVE_PARAMS_TOLERANCE,
    callback=None,
    constraints=None,
):
    """Minimize a scalar function of one or more variables subject to constraints.

    It swiches between two implementations depending on the problem definition.
    It is the most versatile constrained minimization algorithm
    implemented in SciPy and the most appropriate for large-scale problems.
    For equality constrained problems it is an implementation of Byrd-Omojokun
    Trust-Region SQP method described in [17]_ and in [5]_, p. 549. When
    inequality constraints  are imposed as well, it swiches to the trust-region
    interior point  method described in [16]_. This interior point algorithm,
    in turn, solves inequality constraints by introducing slack variables
    and solving a sequence of equality-constrained barrier problems
    for progressively smaller values of the barrier parameter.
    The previously described equality constrained SQP method is
    used to solve the subproblems with increasing levels of accuracy
    as the iterate gets closer to a solution.

    It approximizes the Hessian using the Broyden-Fletcher-Goldfarb-Shanno (BFGS)
    Hessian update strategy.

    References:

    .. [5] Nocedal, J, and S J Wright. 2006. Numerical Optimization.
       Springer New York.
    .. [6] Byrd, R H and P Lu and J. Nocedal. 1995. A Limited Memory
       Algorithm for Bound Constrained Optimization. SIAM Journal on
       Scientific and Statistical Computing 16 (5): 1190-1208.
    .. [16] Byrd, Richard H., Mary E. Hribar, and Jorge Nocedal. 1999.
        An interior point algorithm for large-scale nonlinear  programming.
        SIAM Journal on Optimization 9.4: 877-900.
    .. [17] Lalee, Marucha, Jorge Nocedal, and Todd Plantega. 1998. On the
        implementation of an algorithm for large-scale equality constrained
        optimization. SIAM Journal on Optimization 8.3: 682-706.

    Args:
        gradient_tolerance (float): Tolerance for termination by the norm of the
            Lagrangian gradient. The algorithm will terminate when both the infinity
            norm (i.e., max abs value) of the Lagrangian gradient and the constraint
            violation are smaller than the gradient_tolerance.
        max_iterations (int): If the maximum number of iterations is reached, the
            optimization stops, but we do not count this as convergence.
        relative_params_tolerance (float): Tolerance for termination by the change of
            the independent variable. The algorithm will terminate when the radius of
            the trust region used in the algorithm is smaller than the
            relative_params_tolerance.
        callback (callable): Called after each iteration. It must have the signature:
                ``callback(xk, OptimizeResult state) -> bool``
            where ``xk`` is the current parameter vector. and ``state``
            is an `OptimizeResult` object, with the same fields
            as the ones from the return. If callback returns True
            the algorithm execution is terminated.
            For all the other methods, the signature is:
                ``callback(xk)``
            where ``xk`` is the current parameter vector.
        constraints: not supported at the moment. Constraints can only be passed
            directly as constraints argument to estimagic's `maximize` or `minimize`.
            See :ref:`implementation_of_constraints` for details.

    Returns:
        dict: See :ref:`internal_optimizer_output` for details.

    """
    if constraints is not None:
        raise NotImplementedError(
            "Constraints passed to trust_constr directly are not supported yet. "
            + "Please use estimagic's constraint interface for constraints supported "
            + "by estimagic."
        )

    algo_info = DEFAULT_ALGO_INFO.copy()
    algo_info["name"] = "scipy_trust_constr"
    func = functools.partial(
        criterion_and_derivative, task="criterion", algorithm_info=algo_info,
    )
    gradient = functools.partial(
        criterion_and_derivative, task="derivative", algorithm_info=algo_info
    )

    options = {
        "gtol": gradient_tolerance,
        "maxiter": max_iterations,
        "xtol": relative_params_tolerance,
    }

    res = scipy.optimize.minimize(
        fun=func,
        jac=gradient,
        x0=x,
        method="trust-constr",
        bounds=_get_scipy_bounds(lower_bounds, upper_bounds),
        options=options,
        callback=callback,
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
