import functools

import numpy as np
import scipy

from estimagic.optimization.default_algo_options import ABSOLUTE_CRITERION_TOLERANCE
from estimagic.optimization.default_algo_options import ABSOLUTE_PARAMS_TOLERANCE
from estimagic.optimization.default_algo_options import GRADIENT_TOLERANCE
from estimagic.optimization.default_algo_options import INITIAL_TRUST_RADIUS
from estimagic.optimization.default_algo_options import LIMITED_MEMORY_STORAGE_LENGTH
from estimagic.optimization.default_algo_options import MAX_CRITERION_EVALUATIONS
from estimagic.optimization.default_algo_options import MAX_ITERATIONS
from estimagic.optimization.default_algo_options import MAX_LINE_SEARCH_STEPS
from estimagic.optimization.default_algo_options import RELATIVE_CRITERION_TOLERANCE
from estimagic.optimization.default_algo_options import RELATIVE_PARAMS_TOLERANCE
from estimagic.optimization.default_algo_options import (
    SECOND_BEST_ABSOLUTE_CRITERION_TOLERANCE,
)
from estimagic.optimization.default_algo_options import (
    SECOND_BEST_ABSOLUTE_PARAMS_TOLERANCE,
)


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
    and criterion function jointly, the user should provide a
    ``criterion_and_derivative`` function that exploits the synergies in the
    calculation of criterion and gradient.

    The lbfgsb algorithm is almost perfectly scale invariant. Thus, it is not necessary
    to scale the parameters.

    Below, only details of the optional algorithm options are listed. For the mandatory
    arguments see :ref:`internal_optimizer_interface`. For more background on those
    options, see :ref:`naming_conventions`.

    Args:
        relative_criterion_tolerance (float): Stop when the relative improvement between
            two iterations is smaller than this. More formally, this is expressed as

            .. math::

                \\frac{(f^k - f^{k+1})}{\\max{{|f^k|, |f^{k+1}|, 1}}} \\leq
                \\text{relative_criterion_tolerance}

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
    absolute_criterion_tolerance=SECOND_BEST_ABSOLUTE_CRITERION_TOLERANCE,
    max_iterations=MAX_ITERATIONS,
):
    """Minimize a scalar function of one or more variables using the SLSQP algorithm.

    SLSQP stands for Sequential Least Squares Programming.

    SLSQP is a line search algorithm. It is well suited for continuously
    differentiable scalar optimization problems with up to several hundred parameters.

    The optimizer is taken from scipy which wraps the SLSQP optimization subroutine
    originally implemented by :cite:`Kraft1988`.

    .. note::
        SLSQP's general nonlinear constraints are not supported yet by estimagic.

    Below, only details of the optional algorithm options are listed. For the mandatory
    arguments see :ref:`internal_optimizer_interface`. For more background on those
    options, see :ref:`naming_conventions`.

    Args:
        absolute_criterion_tolerance (float): Precision goal for the value of f in the
            stopping criterion.
        max_iterations (int): If the maximum number of iterations is reached, the
            optimization stops, but we do not count this as convergence.

    Returns:
        dict: See :ref:`internal_optimizer_output` for details.

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
        # this is the absolute criterion tolerance according to
        # scipy/optimize/slsqp/slsqp_optmz.f:495
        "ftol": absolute_criterion_tolerance,
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
    absolute_criterion_tolerance=SECOND_BEST_ABSOLUTE_CRITERION_TOLERANCE,
    absolute_params_tolerance=SECOND_BEST_ABSOLUTE_PARAMS_TOLERANCE,
    adaptive=False,
):
    """Minimize a scalar function using the Nelder-Mead algorithm.

    The Nelder-Mead algorithm is a direct search method (based on function comparison)
    and is often applied to nonlinear optimization problems for which derivatives are
    not known.
    Unlike most modern optimization methods, the Nelder–Mead heuristic can converge to
    a non-stationary point, unless the problem satisfies stronger conditions than are
    necessary for modern methods.

    Nelder-Mead is never the best algorithm to solve a problem but rarely the worst.
    Its popularity is likely due to historic reasons and much larger than its
    properties warrant.

    The argument `initial_simplex` is not supported by estimagic as it is not
    compatible with estimagic's handling of constraints.

    Below, only details of the optional algorithm options are listed. For the mandatory
    arguments see :ref:`internal_optimizer_interface`. For more background on those
    options, see :ref:`naming_conventions`.

    Args:
        max_iterations (int): If the maximum number of iterations is reached, the
            optimization stops, but we do not count this as convergence.
        max_criterion_evaluations (int): If the maximum number of function evaluation
            is reached, the optimization stops but we do not count this as convergence.
        absolute_params_tolerance (float): Absolute difference in parameters between
            iterations that is tolerated to declare convergence.
            As no relative tolerances can be passed to Nelder-Mead, estimagic sets a
            non zero default for this.
        absolute_criterion_tolerance (float): Absolute difference in the criterion value
            between iterations that is tolerated to declare convergence.
            As no relative tolerances can be passed to Nelder-Mead, estimagic sets a
            non zero default for this.
        adaptive (bool): Adapt algorithm parameters to dimensionality of problem.
            Useful for high-dimensional minimization (:cite:`Gao2012`, p. 259-277).
            scipy's default is False.

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
        # both tolerances seem to have to be fulfilled for Nelder-Mead to converge.
        # if not both are specified it does not converge in our tests.
        "xatol": absolute_params_tolerance,
        "fatol": absolute_criterion_tolerance,
    }
    if adaptive is not None:
        options["adaptive"] = adaptive

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
    """Minimize a scalar function using the modified Powell method.

    .. warning::
        In our benchmark using a quadratic objective function, the Powell algorithm
        did not find the optimum very precisely (less than 4 decimal places).
        If you require high precision, you should refine an optimum found with Powell
        with another local optimizer.

    The criterion function need not be differentiable.

    Powell's method is a conjugate direction method, minimising the function by a
    bi-directional search in each parameter's dimension.

    The argument ``direc``, which is the initial set of direction vectors and which
    is part of the scipy interface is not supported by estimagic because it is
    incompatible with how estimagic handles constraints.

    Below, only details of the optional algorithm options are listed. For the mandatory
    arguments see :ref:`internal_optimizer_interface`. For more background on those
    options, see :ref:`naming_conventions`.

    Args:
        relative_params_tolerance (float): Stop when the relative movement between
            parameter vectors is smaller than this.
        relative_criterion_tolerance (float): Stop when the relative improvement between
            two iterations is smaller than this. More formally, this is expressed as

            .. math::

                \\frac{(f^k - f^{k+1})}{\\max{{|f^k|, |f^{k+1}|, 1}}} \\leq
                \\text{relative_criterion_tolerance}

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
        "maxfev": max_criterion_evaluations,
        "maxiter": max_iterations,
    }

    res = scipy.optimize.minimize(
        fun=func,
        x0=x,
        method="Powell",
        bounds=_get_scipy_bounds(lower_bounds, upper_bounds),
        options=options,
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
    norm=np.inf,
):
    """Minimize a scalar function of one or more variables using the BFGS algorithm.

    BFGS stands for Broyden-Fletcher-Goldfarb-Shanno algorithm. It is a quasi-Newton
    method that can be used for solving unconstrained nonlinear optimization problems.

    BFGS is not guaranteed to converge unless the function has a quadratic Taylor
    expansion near an optimum. However, BFGS can have acceptable performance even
    for non-smooth optimization instances.

    Below, only details of the optional algorithm options are listed. For the mandatory
    arguments see :ref:`internal_optimizer_interface`. For more background on those
    options, see :ref:`naming_conventions`.

    Args:
        gradient_tolerance (float): Stop if all elements of the projected gradient are
            smaller than this.
        max_iterations (int): If the maximum number of iterations is reached, the
            optimization stops, but we do not count this as convergence.
        norm (float): Order of the vector norm that is used to calculate the gradient's
            "score" that is compared to the gradient tolerance to determine convergence.
            Defaut is infinite which means that the largest entry of the gradient vector
            is compared to the gradient tolerance.

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
        "norm": norm,
    }

    res = scipy.optimize.minimize(
        fun=func, x0=x, method="BFGS", jac=gradient, options=options,
    )

    return _process_scipy_result(res)


def scipy_conjugate_gradient(
    criterion_and_derivative,
    x,
    *,
    gradient_tolerance=GRADIENT_TOLERANCE,
    max_iterations=MAX_ITERATIONS,
    norm=np.inf,
):
    """Minimize a function using a nonlinear conjugate gradient algorithm.

    The conjugate gradient method finds functions' local optima using just the gradient.

    This conjugate gradient algorithm is based on that of Polak and Ribiere, detailed
    in :cite:`Nocedal2006`, pp. 120-122.

    Conjugate gradient methods tend to work better when:

    * the criterion has a unique global minimizing point, and no local minima or
      other stationary points.
    * the criterion is, at least locally, reasonably well approximated by a
      quadratic function.
    * the criterion is continuous and has a continuous gradient.
    * the gradient is not too large, e.g., has a norm less than 1000.
    * The initial guess is reasonably close to the criterion's global minimizer.

    Below, only details of the optional algorithm options are listed. For the mandatory
    arguments see :ref:`internal_optimizer_interface`. For more background on those
    options, see :ref:`naming_conventions`.

    Args:
        gradient_tolerance (float): Stop if all elements of the projected gradient are
            smaller than this.
        max_iterations (int): If the maximum number of iterations is reached, the
            optimization stops, but we do not count this as convergence.
        norm (float): Order of the vector norm that is used to calculate the gradient's
            "score" that is compared to the gradient tolerance to determine convergence.
            Defaut is infinite which means that the largest entry of the gradient vector
            is compared to the gradient tolerance.

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
        "norm": norm,
    }

    res = scipy.optimize.minimize(
        fun=func, x0=x, method="CG", jac=gradient, options=options,
    )

    return _process_scipy_result(res)


def scipy_newton_cg(
    criterion_and_derivative,
    x,
    *,
    relative_params_tolerance=RELATIVE_PARAMS_TOLERANCE,
    max_iterations=MAX_ITERATIONS,
):
    """Minimize a scalar function using Newton's conjugate gradient algorithm.

    .. warning::
        In our benchmark using a quadratic objective function, the truncated newton
        algorithm did not find the optimum very precisely (less than 4 decimal places).
        If you require high precision, you should refine an optimum found with Powell
        with another local optimizer.

    Newton's conjugate gradient algorithm uses an approximation of the Hessian to find
    the minimum of a function. It is practical for small and large problems
    (see :cite:`Nocedal2006`, p. 140).

    Newton-CG methods are also called truncated Newton methods. This function differs
    scipy_truncated_newton because

    * ``scipy_newton_cg``'s algorithm is written purely in Python using NumPy
      and scipy while ``scipy_truncated_newton``'s algorithm calls a C function.

    * ``scipy_newton_cg``'s algorithm is only for unconstrained minimization
      while ``scipy_truncated_newton``'s algorithm supports bounds.

    Conjugate gradient methods tend to work better when:

    * the criterion has a unique global minimizing point, and no local minima or
      other stationary points.
    * the criterion is, at least locally, reasonably well approximated by a
      quadratic function.
    * the criterion is continuous and has a continuous gradient.
    * the gradient is not too large, e.g., has a norm less than 1000.
    * The initial guess is reasonably close to the criterion's global minimizer.

    Below, only details of the optional algorithm options are listed. For the mandatory
    arguments see :ref:`internal_optimizer_interface`. For more background on those
    options, see :ref:`naming_conventions`.

    Args:
        relative_params_tolerance (float): Stop when the relative movement between
            parameter vectors is smaller than this. Newton CG uses the average relative
            change in the parameters for determining the convergence.
        max_iterations (int): If the maximum number of iterations is reached, the
            optimization stops, but we do not count this as convergence.

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
        fun=func, x0=x, method="Newton-CG", jac=gradient, options=options,
    )

    return _process_scipy_result(res)


def scipy_cobyla(
    criterion_and_derivative,
    x,
    *,
    max_iterations=MAX_ITERATIONS,
    relative_params_tolerance=RELATIVE_PARAMS_TOLERANCE,
    initial_trust_radius=INITIAL_TRUST_RADIUS,
):
    """Minimize a scalar function of one or more variables using the COBYLA algorithm.

    COBYLA stands for Constrained Optimization By Linear Approximation.
    It is deriviative-free and supports nonlinear inequality and equality constraints.

    .. note::
        Cobyla's general nonlinear constraints is not supported yet by estimagic.

    Scipy's implementation wraps the FORTRAN implementation of the algorithm.

    For more information on COBYLA see :cite:`Powell1994`, :cite:`Powell1998` and
    :cite:`Powell2007`.

    Below, only details of the optional algorithm options are listed. For the mandatory
    arguments see :ref:`internal_optimizer_interface`. For more background on those
    options, see :ref:`naming_conventions`.

    Args:
        max_iterations (int): If the maximum number of iterations is reached, the
            optimization stops, but we do not count this as convergence.
        relative_params_tolerance (float): Stop when the relative movement between
            parameter vectors is smaller than this. In case of COBYLA this is a lower
            bound on the size of the trust region and can be seen as the required
            accuracy in the variables but this accuracy is not guaranteed.
        initial_trust_radius (float): Initial trust radius.
            Since a linear approximation is likely only good near the current simplex,
            the linear program is given the further requirement that the solution,
            which will become the next evaluation point must be within a radius
            RHO_j from x_j. RHO_j only decreases, never increases. The initial RHO_j is
            the initial_trust_radius. In this way COBYLA's iterations behave like a
            trust region algorithm.

    Returns:
        dict: See :ref:`internal_optimizer_output` for details.

    """
    algo_info = DEFAULT_ALGO_INFO.copy()
    algo_info["name"] = "scipy_cobyla"

    func = functools.partial(
        criterion_and_derivative, task="criterion", algorithm_info=algo_info,
    )

    options = {"maxiter": max_iterations, "rhobeg": initial_trust_radius}

    res = scipy.optimize.minimize(
        fun=func, x0=x, method="COBYLA", options=options, tol=relative_params_tolerance,
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
    func_min_estimate=None,
    max_hess_evaluations_per_iteration=None,
    max_step_for_line_search=None,
    func_scaling_factor=None,
    line_search_severity=None,
    finitie_difference_precision=None,
    criterion_rescale_factor=None,
):
    """Minimize a scalar function using truncated Newton algorithm.

    This function differs from scipy_newton_cg because

    * ``scipy_newton_cg``'s algorithm is written purely in Python using NumPy
      and scipy while ``scipy_truncated_newton``'s algorithm calls a C function.

    * ``scipy_newton_cg``'s algorithm is only for unconstrained minimization
      while ``scipy_truncated_newton``'s algorithm supports bounds.

    Conjugate gradient methods tend to work better when:

    * the criterion has a unique global minimizing point, and no local minima or
      other stationary points.
    * the criterion is, at least locally, reasonably well approximated by a
      quadratic function.
    * the criterion is continuous and has a continuous gradient.
    * the gradient is not too large, e.g., has a norm less than 1000.
    * The initial guess is reasonably close to the criterion's global minimizer.

    estimagic does not support the ``scale``  nor ``offset`` argument as they are not
    compatible with the way estimagic handles constraints. It also does not support
    ``messg_num`` which is an additional way to control the verbosity of the optimizer.

    Below, only details of the optional algorithm options are listed. For the mandatory
    arguments see :ref:`internal_optimizer_interface`. For more background on those
    options, see :ref:`naming_conventions`.

    Args:
        func_min_estimate (float): Minimum function value estimate. Defaults to 0.
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
            gradient_tolerance < 0.0, gradient_tolerance is set to
            1e-2 * sqrt(accuracy).
        max_hess_evaluations_per_iteration (int): Maximum number of hessian*vector
            evaluations per main iteration. If ``max_hess_evaluations == 0``, the
            direction chosen is ``- gradient``. If ``max_hess_evaluations < 0``,
            ``max_hess_evaluations`` is set to ``max(1,min(50,n/2))`` where n is the
            length of the parameter vector. This is also the default.
        max_step_for_line_search (float): Maximum step for the line search.
            It may be increased during the optimization. If too small, it will be set
            to 10.0. By default we use scipy's default.
        func_scaling_factor (float): Scaling factor (in log10) used to control the
            rescaling of the function evaluations. If the scaling factor is 0, rescale
            at each iteration. This is the default. On the other hand, if it is large
            no rescaling is done. If it is < 0, the scaling factor is set to 1.3.
        line_search_severity (float): Severity of the line search. If < 0 or > 1,
            set to 0.25. Estimagic defaults to scipy's default.
        finitie_difference_precision (float): Relative precision for finite difference
            calculations. If <= machine_precision, set to sqrt(machine_precision).
            Estimagic defaults to scipy's default.
        criterion_rescale_factor (float): Scaling factor (in log10) used to trigger
            criterion rescaling. If 0, rescale at each iteration. If a large value,
            never rescale. If < 0, rescale is set to 1.3. Estimagic defaults to scipy's
            default.

    Returns:
        dict: See :ref:`internal_optimizer_output` for details.

    """
    algo_info = DEFAULT_ALGO_INFO.copy()
    algo_info["name"] = "scipy_truncated_newton"
    func = functools.partial(
        criterion_and_derivative, task="criterion", algorithm_info=algo_info,
    )
    gradient = functools.partial(
        criterion_and_derivative, task="derivative", algorithm_info=algo_info
    )

    options = {
        # scipy/optimize/tnc/tnc.c::809 and 844 show that ftol is the
        # absolute criterion tolerance
        "ftol": absolute_criterion_tolerance,
        # scipy/optimize/tnc/tnc.c::856 show sthat xtol is the absolute parameter
        # tolerance
        "xtol": absolute_params_tolerance,
        "gtol": gradient_tolerance,
        "maxfun": max_criterion_evaluations,
        "maxiter": max_iterations,
    }
    if max_hess_evaluations_per_iteration is not None:
        options["maxCGit"] = max_hess_evaluations_per_iteration
    if max_step_for_line_search is not None:
        options["stepmx"] = max_step_for_line_search
    if func_scaling_factor is not None:
        options["rescale"] = func_scaling_factor
    if func_min_estimate is not None:
        options["minfev"] = func_min_estimate
    if line_search_severity is not None:
        options["eta"] = line_search_severity
    if finitie_difference_precision is not None:
        options["accuracy"] = finitie_difference_precision
    if criterion_rescale_factor is not None:
        options["rescale"] = criterion_rescale_factor

    res = scipy.optimize.minimize(
        fun=func,
        x0=x,
        method="TNC",
        jac=gradient,
        options=options,
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
    initial_trust_radius=INITIAL_TRUST_RADIUS,
):
    """Minimize a scalar function of one or more variables subject to constraints.

    .. warning::
        In our benchmark using a quadratic objective function, the trust_constr
        algorithm did not find the optimum very precisely (less than 4 decimal places).
        If you require high precision, you should refine an optimum found with Powell
        with another local optimizer.

    .. note::
        Its general nonlinear constraints' handling is not supported yet by estimagic.

    It swiches between two implementations depending on the problem definition.
    It is the most versatile constrained minimization algorithm
    implemented in SciPy and the most appropriate for large-scale problems.
    For equality constrained problems it is an implementation of Byrd-Omojokun
    Trust-Region SQP method described in :cite:`Lalee1998` and in :cite:`Conn2000`,
    p. 549. When inequality constraints  are imposed as well, it swiches to the
    trust-region interior point method described in :cite:`Byrd1999`.
    This interior point algorithm in turn, solves inequality constraints by
    introducing slack variables and solving a sequence of equality-constrained
    barrier problems for progressively smaller values of the barrier parameter.
    The previously described equality constrained SQP method is
    used to solve the subproblems with increasing levels of accuracy
    as the iterate gets closer to a solution.

    It approximizes the Hessian using the Broyden-Fletcher-Goldfarb-Shanno (BFGS)
    Hessian update strategy.

    Below, only details of the optional algorithm options are listed. For the mandatory
    arguments see :ref:`internal_optimizer_interface`. For more background on those
    options, see :ref:`naming_conventions`.

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
        initial_trust_radius (float): Initial trust radius. The trust radius gives the
            maximum distance between solution points in consecutive iterations.
            It reflects the trust the algorithm puts in the local approximation of the
            optimization problem. For an accurate local approximation the trust-region
            should be large and for an approximation valid only close to the current
            point it should be a small one. The trust radius is automatically updated
            throughout the optimization process, with ``initial_trust_radius`` being
            its initial value. Default is 1 (recommended in :cite:`Conn2000`, p. 19).

    Returns:
        dict: See :ref:`internal_optimizer_output` for details.

    """
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
        "initial_tr_radius": initial_trust_radius,
        # don't have "grad" here as we already supply the gradient via the "jac"
        # argument supplied directly to scipy.optimize.minimize.
    }

    res = scipy.optimize.minimize(
        fun=func,
        jac=gradient,
        x0=x,
        method="trust-constr",
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
