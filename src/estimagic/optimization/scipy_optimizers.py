"""Implement scipy algorithms.

The following ``scipy`` algorithms are not supported because they
require the specification of the Hessian:

- dogleg
- trust-ncg
- trust-exact
- trust-krylov

The following arguments are not supported as part of ``algo_options``:

- ``disp``
    If set to True would print a convergence message.
    In estimagic it's always set to its default False.
    Refer to estimagic's result dictionary's "success" entry for the convergence
    message.
- ``return_all``
    If set to True, a list of the best solution at each iteration is returned.
    In estimagic it's always set to its default False.
    Use estimagic's database and dashboard instead to explore your criterion and
    algorithm.
- ``tol``
    This argument of minimize (not an options key) is passed as different types of
    tolerance (gradient, parameter or criterion, as well as relative or absolute)
    depending on the selected algorithm. We require the user to explicitely input
    the tolerance criteria or use our defaults instead.
- ``args``
    This argument of minimize (not an options key) is partialed into the function
    for the user. Specify ``criterion_kwargs`` in ``maximize`` or ``minimize`` to
    achieve the same behavior.
- ``callback``
    This argument would be called after each iteration and the algorithm would
    terminate if it returned True.

.. note::
    `scipy`'s COBYLA, SLSQP and trust-constr support general non linear constraints
    in principle. However, for the moment they are not supported.

"""
import functools

import numpy as np
import scipy
from estimagic.optimization.algo_options import CONVERGENCE_ABSOLUTE_CRITERION_TOLERANCE
from estimagic.optimization.algo_options import CONVERGENCE_ABSOLUTE_GRADIENT_TOLERANCE
from estimagic.optimization.algo_options import CONVERGENCE_ABSOLUTE_PARAMS_TOLERANCE
from estimagic.optimization.algo_options import CONVERGENCE_RELATIVE_CRITERION_TOLERANCE
from estimagic.optimization.algo_options import CONVERGENCE_RELATIVE_GRADIENT_TOLERANCE
from estimagic.optimization.algo_options import CONVERGENCE_RELATIVE_PARAMS_TOLERANCE
from estimagic.optimization.algo_options import (
    CONVERGENCE_SECOND_BEST_ABSOLUTE_CRITERION_TOLERANCE,
)
from estimagic.optimization.algo_options import (
    CONVERGENCE_SECOND_BEST_ABSOLUTE_PARAMS_TOLERANCE,
)
from estimagic.optimization.algo_options import LIMITED_MEMORY_STORAGE_LENGTH
from estimagic.optimization.algo_options import MAX_LINE_SEARCH_STEPS
from estimagic.optimization.algo_options import STOPPING_MAX_CRITERION_EVALUATIONS
from estimagic.optimization.algo_options import STOPPING_MAX_ITERATIONS
from estimagic.utilities import calculate_trustregion_initial_radius

DEFAULT_ALGO_INFO = {
    "primary_criterion_entry": "value",
    "parallelizes": False,
    "needs_scaling": False,
}


def scipy_lbfgsb(
    criterion_and_derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    convergence_relative_criterion_tolerance=CONVERGENCE_RELATIVE_CRITERION_TOLERANCE,
    convergence_absolute_gradient_tolerance=CONVERGENCE_ABSOLUTE_GRADIENT_TOLERANCE,
    stopping_max_criterion_evaluations=STOPPING_MAX_CRITERION_EVALUATIONS,
    stopping_max_iterations=STOPPING_MAX_ITERATIONS,
    limited_memory_storage_length=LIMITED_MEMORY_STORAGE_LENGTH,
    max_line_search_steps=MAX_LINE_SEARCH_STEPS,
):
    """Minimize a scalar function of one or more variables using the L-BFGS-B algorithm.

    For details see :ref:`list_of_scipy_algorithms`.

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
        "ftol": convergence_relative_criterion_tolerance,
        "gtol": convergence_absolute_gradient_tolerance,
        "maxfun": stopping_max_criterion_evaluations,
        "maxiter": stopping_max_iterations,
        "maxls": max_line_search_steps,
    }

    res = scipy.optimize.minimize(
        fun=func,
        x0=x,
        method="L-BFGS-B",
        jac=True,
        bounds=get_scipy_bounds(lower_bounds, upper_bounds),
        options=options,
    )

    return process_scipy_result(res)


def scipy_slsqp(
    criterion_and_derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    convergence_absolute_criterion_tolerance=CONVERGENCE_SECOND_BEST_ABSOLUTE_CRITERION_TOLERANCE,  # noqa: E501
    stopping_max_iterations=STOPPING_MAX_ITERATIONS,
):
    """Minimize a scalar function of one or more variables using the SLSQP algorithm.

    For details see :ref:`list_of_scipy_algorithms`.

    """
    algo_info = DEFAULT_ALGO_INFO.copy()
    algo_info["name"] = "scipy_slsqp"

    func = functools.partial(
        criterion_and_derivative,
        task="criterion",
        algorithm_info=algo_info,
    )

    gradient = functools.partial(
        criterion_and_derivative, task="derivative", algorithm_info=algo_info
    )

    options = {
        "maxiter": stopping_max_iterations,
        # this is the absolute criterion tolerance according to
        # scipy/optimize/slsqp/slsqp_optmz.f:495
        "ftol": convergence_absolute_criterion_tolerance,
    }

    res = scipy.optimize.minimize(
        fun=func,
        x0=x,
        method="SLSQP",
        jac=gradient,
        bounds=get_scipy_bounds(lower_bounds, upper_bounds),
        options=options,
    )

    return process_scipy_result(res)


def scipy_neldermead(
    criterion_and_derivative,
    x,
    *,
    stopping_max_iterations=STOPPING_MAX_ITERATIONS,
    stopping_max_criterion_evaluations=STOPPING_MAX_CRITERION_EVALUATIONS,
    convergence_absolute_criterion_tolerance=CONVERGENCE_SECOND_BEST_ABSOLUTE_CRITERION_TOLERANCE,  # noqa: E501
    convergence_absolute_params_tolerance=CONVERGENCE_SECOND_BEST_ABSOLUTE_PARAMS_TOLERANCE,  # noqa: E501
    adaptive=False,
):
    """Minimize a scalar function using the Nelder-Mead algorithm.

    For details see :ref:`list_of_scipy_algorithms`.

    """
    algo_info = DEFAULT_ALGO_INFO.copy()
    algo_info["name"] = "scipy_neldermead"
    func = functools.partial(
        criterion_and_derivative,
        task="criterion",
        algorithm_info=algo_info,
    )
    options = {
        "maxiter": stopping_max_iterations,
        "maxfev": stopping_max_criterion_evaluations,
        # both tolerances seem to have to be fulfilled for Nelder-Mead to converge.
        # if not both are specified it does not converge in our tests.
        "xatol": convergence_absolute_params_tolerance,
        "fatol": convergence_absolute_criterion_tolerance,
        "adaptive": adaptive,
    }

    res = scipy.optimize.minimize(
        fun=func,
        x0=x,
        method="Nelder-Mead",
        options=options,
    )

    return process_scipy_result(res)


def scipy_powell(
    criterion_and_derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    convergence_relative_params_tolerance=CONVERGENCE_RELATIVE_PARAMS_TOLERANCE,
    convergence_relative_criterion_tolerance=CONVERGENCE_RELATIVE_CRITERION_TOLERANCE,
    stopping_max_criterion_evaluations=STOPPING_MAX_CRITERION_EVALUATIONS,
    stopping_max_iterations=STOPPING_MAX_ITERATIONS,
):
    """Minimize a scalar function using the modified Powell method.

    For details see :ref:`list_of_scipy_algorithms`.

    """
    algo_info = DEFAULT_ALGO_INFO.copy()
    algo_info["name"] = "scipy_powell"
    func = functools.partial(
        criterion_and_derivative,
        task="criterion",
        algorithm_info=algo_info,
    )

    options = {
        "xtol": convergence_relative_params_tolerance,
        "ftol": convergence_relative_criterion_tolerance,
        "maxfev": stopping_max_criterion_evaluations,
        "maxiter": stopping_max_iterations,
    }

    res = scipy.optimize.minimize(
        fun=func,
        x0=x,
        method="Powell",
        bounds=get_scipy_bounds(lower_bounds, upper_bounds),
        options=options,
    )

    return process_scipy_result(res)


def scipy_bfgs(
    criterion_and_derivative,
    x,
    *,
    convergence_absolute_gradient_tolerance=CONVERGENCE_ABSOLUTE_GRADIENT_TOLERANCE,
    stopping_max_iterations=STOPPING_MAX_ITERATIONS,
    norm=np.inf,
):
    """Minimize a scalar function of one or more variables using the BFGS algorithm.

    For details see :ref:`list_of_scipy_algorithms`.

    """
    algo_info = DEFAULT_ALGO_INFO.copy()
    algo_info["name"] = "scipy_bfgs"
    func = functools.partial(
        criterion_and_derivative,
        task="criterion",
        algorithm_info=algo_info,
    )
    gradient = functools.partial(
        criterion_and_derivative, task="derivative", algorithm_info=algo_info
    )

    options = {
        "gtol": convergence_absolute_gradient_tolerance,
        "maxiter": stopping_max_iterations,
        "norm": norm,
    }

    res = scipy.optimize.minimize(
        fun=func,
        x0=x,
        method="BFGS",
        jac=gradient,
        options=options,
    )

    return process_scipy_result(res)


def scipy_conjugate_gradient(
    criterion_and_derivative,
    x,
    *,
    convergence_absolute_gradient_tolerance=CONVERGENCE_ABSOLUTE_GRADIENT_TOLERANCE,
    stopping_max_iterations=STOPPING_MAX_ITERATIONS,
    norm=np.inf,
):
    """Minimize a function using a nonlinear conjugate gradient algorithm.

    For details see :ref:`list_of_scipy_algorithms`.

    """
    algo_info = DEFAULT_ALGO_INFO.copy()
    algo_info["name"] = "scipy_conjugate_gradient"
    func = functools.partial(
        criterion_and_derivative,
        task="criterion",
        algorithm_info=algo_info,
    )

    gradient = functools.partial(
        criterion_and_derivative, task="derivative", algorithm_info=algo_info
    )

    options = {
        "gtol": convergence_absolute_gradient_tolerance,
        "maxiter": stopping_max_iterations,
        "norm": norm,
    }

    res = scipy.optimize.minimize(
        fun=func,
        x0=x,
        method="CG",
        jac=gradient,
        options=options,
    )

    return process_scipy_result(res)


def scipy_newton_cg(
    criterion_and_derivative,
    x,
    *,
    convergence_relative_params_tolerance=CONVERGENCE_RELATIVE_PARAMS_TOLERANCE,
    stopping_max_iterations=STOPPING_MAX_ITERATIONS,
):
    """Minimize a scalar function using Newton's conjugate gradient algorithm.

    For details see :ref:`list_of_scipy_algorithms`.

    """
    algo_info = DEFAULT_ALGO_INFO.copy()
    algo_info["name"] = "scipy_newton_cg"
    func = functools.partial(
        criterion_and_derivative,
        task="criterion",
        algorithm_info=algo_info,
    )
    gradient = functools.partial(
        criterion_and_derivative, task="derivative", algorithm_info=algo_info
    )

    options = {
        "xtol": convergence_relative_params_tolerance,
        "maxiter": stopping_max_iterations,
    }

    res = scipy.optimize.minimize(
        fun=func,
        x0=x,
        method="Newton-CG",
        jac=gradient,
        options=options,
    )

    return process_scipy_result(res)


def scipy_cobyla(
    criterion_and_derivative,
    x,
    *,
    stopping_max_iterations=STOPPING_MAX_ITERATIONS,
    convergence_relative_params_tolerance=CONVERGENCE_RELATIVE_PARAMS_TOLERANCE,
    trustregion_initial_radius=None,
):
    """Minimize a scalar function of one or more variables using the COBYLA algorithm.

    For details see :ref:`list_of_scipy_algorithms`.

    """
    algo_info = DEFAULT_ALGO_INFO.copy()
    algo_info["name"] = "scipy_cobyla"

    func = functools.partial(
        criterion_and_derivative,
        task="criterion",
        algorithm_info=algo_info,
    )

    if trustregion_initial_radius is None:
        trustregion_initial_radius = calculate_trustregion_initial_radius(x)

    options = {"maxiter": stopping_max_iterations, "rhobeg": trustregion_initial_radius}

    res = scipy.optimize.minimize(
        fun=func,
        x0=x,
        method="COBYLA",
        options=options,
        tol=convergence_relative_params_tolerance,
    )

    return process_scipy_result(res)


def scipy_truncated_newton(
    criterion_and_derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    stopping_max_criterion_evaluations=STOPPING_MAX_CRITERION_EVALUATIONS,
    stopping_max_iterations=STOPPING_MAX_ITERATIONS,
    convergence_absolute_criterion_tolerance=CONVERGENCE_ABSOLUTE_CRITERION_TOLERANCE,
    convergence_absolute_params_tolerance=CONVERGENCE_ABSOLUTE_PARAMS_TOLERANCE,
    convergence_absolute_gradient_tolerance=CONVERGENCE_ABSOLUTE_GRADIENT_TOLERANCE,
    func_min_estimate=0,
    max_hess_evaluations_per_iteration=-1,
    max_step_for_line_search=0,
    line_search_severity=-1,
    finitie_difference_precision=0,
    criterion_rescale_factor=-1,
):
    """Minimize a scalar function using truncated Newton algorithm.

    For details see :ref:`list_of_scipy_algorithms`.

    """
    algo_info = DEFAULT_ALGO_INFO.copy()
    algo_info["name"] = "scipy_truncated_newton"
    func = functools.partial(
        criterion_and_derivative,
        task="criterion",
        algorithm_info=algo_info,
    )
    gradient = functools.partial(
        criterion_and_derivative, task="derivative", algorithm_info=algo_info
    )

    options = {
        # scipy/optimize/tnc/tnc.c::809 and 844 show that ftol is the
        # absolute criterion tolerance
        "ftol": convergence_absolute_criterion_tolerance,
        # scipy/optimize/tnc/tnc.c::856 show sthat xtol is the absolute parameter
        # tolerance
        "xtol": convergence_absolute_params_tolerance,
        "gtol": convergence_absolute_gradient_tolerance,
        "maxfun": stopping_max_criterion_evaluations,
        "maxiter": stopping_max_iterations,
        "maxCGit": max_hess_evaluations_per_iteration,
        "stepmx": max_step_for_line_search,
        "minfev": func_min_estimate,
        "eta": line_search_severity,
        "accuracy": finitie_difference_precision,
        "rescale": criterion_rescale_factor,
    }

    res = scipy.optimize.minimize(
        fun=func,
        x0=x,
        method="TNC",
        jac=gradient,
        options=options,
        bounds=get_scipy_bounds(lower_bounds, upper_bounds),
    )

    return process_scipy_result(res)


def scipy_trust_constr(
    criterion_and_derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    convergence_absolute_gradient_tolerance=1e-08,
    convergence_relative_params_tolerance=CONVERGENCE_RELATIVE_PARAMS_TOLERANCE,
    stopping_max_iterations=STOPPING_MAX_ITERATIONS,
    trustregion_initial_radius=None,
):
    """Minimize a scalar function of one or more variables subject to constraints.

    For details see :ref:`list_of_scipy_algorithms`.

    """
    algo_info = DEFAULT_ALGO_INFO.copy()
    algo_info["name"] = "scipy_trust_constr"
    func = functools.partial(
        criterion_and_derivative,
        task="criterion",
        algorithm_info=algo_info,
    )
    gradient = functools.partial(
        criterion_and_derivative, task="derivative", algorithm_info=algo_info
    )

    if trustregion_initial_radius is None:
        trustregion_initial_radius = calculate_trustregion_initial_radius(x)

    options = {
        "gtol": convergence_absolute_gradient_tolerance,
        "maxiter": stopping_max_iterations,
        "xtol": convergence_relative_params_tolerance,
        "initial_tr_radius": trustregion_initial_radius,
        # don't have "grad" here as we already supply the gradient via the "jac"
        # argument supplied directly to scipy.optimize.minimize.
    }

    res = scipy.optimize.minimize(
        fun=func,
        jac=gradient,
        x0=x,
        method="trust-constr",
        bounds=get_scipy_bounds(lower_bounds, upper_bounds),
        options=options,
    )

    return process_scipy_result(res)


def scipy_ls_trf(
    criterion_and_derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    convergence_relative_criterion_tol=CONVERGENCE_RELATIVE_CRITERION_TOLERANCE,
    convergence_relative_gradient_tol=CONVERGENCE_RELATIVE_GRADIENT_TOLERANCE,
    stopping_max_criterion_evaluations=STOPPING_MAX_CRITERION_EVALUATIONS,
    relative_step_size_diff_approx=None,
    tr_solver=None,
    tr_solver_options=None,
):
    """
    Minimize a scalar function using a trust region reflective method.

    For details see :ref:`list_of_scipy_algorithms`.

    """
    return _scipy_least_squares(
        criterion_and_derivative,
        x,
        lower_bounds,
        upper_bounds,
        convergence_relative_criterion_tol=convergence_relative_criterion_tol,
        convergence_relative_gradient_tol=convergence_relative_gradient_tol,
        stopping_max_criterion_evaluations=stopping_max_criterion_evaluations,
        relative_step_size_diff_approx=relative_step_size_diff_approx,
        tr_solver=tr_solver,
        tr_solver_options=tr_solver_options,
        method="trf",
    )


def scipy_ls_dogbox(
    criterion_and_derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    convergence_relative_criterion_tol=CONVERGENCE_RELATIVE_CRITERION_TOLERANCE,
    convergence_relative_gradient_tol=CONVERGENCE_RELATIVE_GRADIENT_TOLERANCE,
    stopping_max_criterion_evaluations=STOPPING_MAX_CRITERION_EVALUATIONS,
    relative_step_size_diff_approx=None,
    tr_solver=None,
    tr_solver_options=None,
):
    """
    Minimize a scalar function using a rectangular trust region method.

    For details see :ref:`list_of_scipy_algorithms`.

    """
    return _scipy_least_squares(
        criterion_and_derivative,
        x,
        lower_bounds,
        upper_bounds,
        convergence_relative_criterion_tol=convergence_relative_criterion_tol,
        convergence_relative_gradient_tol=convergence_relative_gradient_tol,
        stopping_max_criterion_evaluations=stopping_max_criterion_evaluations,
        relative_step_size_diff_approx=relative_step_size_diff_approx,
        tr_solver=tr_solver,
        tr_solver_options=tr_solver_options,
        method="dogbox",
    )


def process_scipy_result(scipy_results_obj):
    # using get with defaults to access dict elements is just a safety measure
    raw_res = {**scipy_results_obj}
    processed = {
        "solution_x": raw_res.get("x"),
        "solution_criterion": raw_res.get("fun"),
        "solution_derivative": raw_res.get("jac"),
        "solution_hessian": raw_res.get("hess"),
        "n_criterion_evaluations": raw_res.get("nfev"),
        "n_derivative_evaluations": raw_res.get("njac") or raw_res.get("njev"),
        "n_iterations": raw_res.get("nit"),
        "success": raw_res.get("success"),
        "reached_convergence_criterion": None,
        "message": raw_res.get("message"),
    }
    return processed


def get_scipy_bounds(lower_bounds, upper_bounds):
    # Scipy works with `None` instead of infinite values for unconstrained parameters
    # and requires a list of tuples for each parameter with lower and upper bound.
    bounds = np.column_stack([lower_bounds, upper_bounds])
    mask = ~np.isfinite(bounds)
    bounds = bounds.astype("object")
    bounds[mask] = None
    return list(map(tuple, bounds))


def _scipy_least_squares(
    criterion_and_derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    convergence_relative_criterion_tol=CONVERGENCE_RELATIVE_CRITERION_TOLERANCE,
    convergence_relative_gradient_tol=CONVERGENCE_RELATIVE_GRADIENT_TOLERANCE,
    stopping_max_criterion_evaluations=STOPPING_MAX_CRITERION_EVALUATIONS,
    relative_step_size_diff_approx=None,
    tr_solver=None,
    tr_solver_options=None,
    method="trf",
):
    """Internal function used by the scipy_ls_trf and scipy_ls_dogbox functions."""

    if method not in ["trf", "dogbox", "lm"]:
        raise ValueError(
            f"Method {method} is not supported within scipy_least_squares."
        )

    if tr_solver_options is None:
        tr_solver_options = {}

    algo_info = DEFAULT_ALGO_INFO.copy()
    algo_info["name"] = f"scipy_ls_{method}"
    algo_info["primary_criterion_entry"] = "root_contributions"
    func = functools.partial(
        criterion_and_derivative,
        task="criterion",
        algorithm_info=algo_info,
    )

    gradient = functools.partial(
        criterion_and_derivative, task="derivative", algorithm_info=algo_info
    )

    res = scipy.optimize.least_squares(
        fun=func,
        x0=x,
        jac=gradient,
        # Don't use get_scipy_bounds, b.c. least_squares uses np.inf
        bounds=(lower_bounds, upper_bounds),
        max_nfev=stopping_max_criterion_evaluations,
        ftol=convergence_relative_criterion_tol,
        gtol=convergence_relative_gradient_tol,
        method=method,
        diff_step=relative_step_size_diff_approx,
        tr_solver=tr_solver,
        tr_options=tr_solver_options,
    )

    return process_scipy_result(res)
