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

"""
import functools

import numpy as np
import scipy
from scipy.optimize import Bounds, NonlinearConstraint

from estimagic.batch_evaluators import process_batch_evaluator
from estimagic.decorators import mark_minimizer
from estimagic.optimization.algo_options import (
    CONVERGENCE_ABSOLUTE_CRITERION_TOLERANCE,
    CONVERGENCE_ABSOLUTE_GRADIENT_TOLERANCE,
    CONVERGENCE_ABSOLUTE_PARAMS_TOLERANCE,
    CONVERGENCE_RELATIVE_CRITERION_TOLERANCE,
    CONVERGENCE_RELATIVE_GRADIENT_TOLERANCE,
    CONVERGENCE_RELATIVE_PARAMS_TOLERANCE,
    CONVERGENCE_SECOND_BEST_ABSOLUTE_CRITERION_TOLERANCE,
    CONVERGENCE_SECOND_BEST_ABSOLUTE_PARAMS_TOLERANCE,
    LIMITED_MEMORY_STORAGE_LENGTH,
    MAX_LINE_SEARCH_STEPS,
    STOPPING_MAX_CRITERION_EVALUATIONS,
    STOPPING_MAX_CRITERION_EVALUATIONS_GLOBAL,
    STOPPING_MAX_ITERATIONS,
)
from estimagic.parameters.nonlinear_constraints import (
    equality_as_inequality_constraints,
    vector_as_list_of_scalar_constraints,
)
from estimagic.utilities import calculate_trustregion_initial_radius


@mark_minimizer(name="scipy_lbfgsb")
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
    options = {
        "maxcor": limited_memory_storage_length,
        "ftol": convergence_relative_criterion_tolerance,
        "gtol": convergence_absolute_gradient_tolerance,
        "maxfun": stopping_max_criterion_evaluations,
        "maxiter": stopping_max_iterations,
        "maxls": max_line_search_steps,
    }
    res = scipy.optimize.minimize(
        fun=criterion_and_derivative,
        x0=x,
        method="L-BFGS-B",
        jac=True,
        bounds=_get_scipy_bounds(lower_bounds, upper_bounds),
        options=options,
    )

    return process_scipy_result(res)


@mark_minimizer(name="scipy_slsqp")
def scipy_slsqp(
    criterion,
    derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    nonlinear_constraints=(),
    convergence_absolute_criterion_tolerance=CONVERGENCE_SECOND_BEST_ABSOLUTE_CRITERION_TOLERANCE,  # noqa: E501
    stopping_max_iterations=STOPPING_MAX_ITERATIONS,
):
    """Minimize a scalar function of one or more variables using the SLSQP algorithm.

    For details see :ref:`list_of_scipy_algorithms`.

    """
    options = {
        "maxiter": stopping_max_iterations,
        "ftol": convergence_absolute_criterion_tolerance,
    }

    res = scipy.optimize.minimize(
        fun=criterion,
        x0=x,
        method="SLSQP",
        jac=derivative,
        bounds=_get_scipy_bounds(lower_bounds, upper_bounds),
        constraints=nonlinear_constraints,
        options=options,
    )

    return process_scipy_result(res)


@mark_minimizer(name="scipy_neldermead", needs_scaling=True)
def scipy_neldermead(
    criterion,
    x,
    lower_bounds,
    upper_bounds,
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
        fun=criterion,
        x0=x,
        bounds=_get_scipy_bounds(lower_bounds, upper_bounds),
        method="Nelder-Mead",
        options=options,
    )

    return process_scipy_result(res)


@mark_minimizer(name="scipy_powell", needs_scaling=True)
def scipy_powell(
    criterion,
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
    options = {
        "xtol": convergence_relative_params_tolerance,
        "ftol": convergence_relative_criterion_tolerance,
        "maxfev": stopping_max_criterion_evaluations,
        "maxiter": stopping_max_iterations,
    }

    res = scipy.optimize.minimize(
        fun=criterion,
        x0=x,
        method="Powell",
        bounds=_get_scipy_bounds(lower_bounds, upper_bounds),
        options=options,
    )

    return process_scipy_result(res)


@mark_minimizer(name="scipy_bfgs")
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
    options = {
        "gtol": convergence_absolute_gradient_tolerance,
        "maxiter": stopping_max_iterations,
        "norm": norm,
    }

    res = scipy.optimize.minimize(
        fun=criterion_and_derivative,
        x0=x,
        method="BFGS",
        jac=True,
        options=options,
    )

    return process_scipy_result(res)


@mark_minimizer(name="scipy_conjugate_gradient")
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
    options = {
        "gtol": convergence_absolute_gradient_tolerance,
        "maxiter": stopping_max_iterations,
        "norm": norm,
    }

    res = scipy.optimize.minimize(
        fun=criterion_and_derivative,
        x0=x,
        method="CG",
        jac=True,
        options=options,
    )

    return process_scipy_result(res)


@mark_minimizer(name="scipy_newton_cg")
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
    options = {
        "xtol": convergence_relative_params_tolerance,
        "maxiter": stopping_max_iterations,
    }

    res = scipy.optimize.minimize(
        fun=criterion_and_derivative,
        x0=x,
        method="Newton-CG",
        jac=True,
        options=options,
    )

    return process_scipy_result(res)


@mark_minimizer(name="scipy_cobyla", needs_scaling=True)
def scipy_cobyla(
    criterion,
    x,
    *,
    nonlinear_constraints=(),
    stopping_max_iterations=STOPPING_MAX_ITERATIONS,
    convergence_relative_params_tolerance=CONVERGENCE_RELATIVE_PARAMS_TOLERANCE,
    trustregion_initial_radius=None,
):
    """Minimize a scalar function of one or more variables using the COBYLA algorithm.

    For details see :ref:`list_of_scipy_algorithms`.

    """
    if trustregion_initial_radius is None:
        trustregion_initial_radius = calculate_trustregion_initial_radius(x)

    options = {"maxiter": stopping_max_iterations, "rhobeg": trustregion_initial_radius}

    # cannot handle equality constraints
    nonlinear_constraints = equality_as_inequality_constraints(nonlinear_constraints)

    res = scipy.optimize.minimize(
        fun=criterion,
        x0=x,
        method="COBYLA",
        constraints=nonlinear_constraints,
        options=options,
        tol=convergence_relative_params_tolerance,
    )

    return process_scipy_result(res)


@mark_minimizer(name="scipy_truncated_newton")
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
        fun=criterion_and_derivative,
        x0=x,
        method="TNC",
        jac=True,
        options=options,
        bounds=_get_scipy_bounds(lower_bounds, upper_bounds),
    )

    return process_scipy_result(res)


@mark_minimizer(name="scipy_trust_constr")
def scipy_trust_constr(
    criterion_and_derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    nonlinear_constraints=(),
    convergence_absolute_gradient_tolerance=1e-08,
    convergence_relative_params_tolerance=CONVERGENCE_RELATIVE_PARAMS_TOLERANCE,
    stopping_max_iterations=STOPPING_MAX_ITERATIONS,
    trustregion_initial_radius=None,
):
    """Minimize a scalar function of one or more variables subject to constraints.

    For details see :ref:`list_of_scipy_algorithms`.

    """
    if trustregion_initial_radius is None:
        trustregion_initial_radius = calculate_trustregion_initial_radius(x)

    options = {
        "gtol": convergence_absolute_gradient_tolerance,
        "maxiter": stopping_max_iterations,
        "xtol": convergence_relative_params_tolerance,
        "initial_tr_radius": trustregion_initial_radius,
    }

    # cannot handle equality constraints
    nonlinear_constraints = equality_as_inequality_constraints(nonlinear_constraints)

    res = scipy.optimize.minimize(
        fun=criterion_and_derivative,
        jac=True,
        x0=x,
        method="trust-constr",
        bounds=_get_scipy_bounds(lower_bounds, upper_bounds),
        constraints=_get_scipy_constraints(nonlinear_constraints),
        options=options,
    )

    return process_scipy_result(res)


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


def _get_scipy_bounds(lower_bounds, upper_bounds):
    return Bounds(lb=lower_bounds, ub=upper_bounds)


def _get_scipy_constraints(constraints):
    """Transform internal nonlinear constraints to scipy readable format.

    This format is currently only used by scipy_trust_constr.

    """
    scipy_constraints = [_internal_to_scipy_constraint(c) for c in constraints]
    return scipy_constraints


def _internal_to_scipy_constraint(c):
    new_constr = NonlinearConstraint(
        fun=c["fun"],
        lb=np.zeros(c["n_constr"]),
        ub=np.tile(np.inf, c["n_constr"]),
        jac=c["jac"],
    )
    return new_constr


def _scipy_least_squares(
    criterion,
    derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    convergence_relative_criterion_tolerance=CONVERGENCE_RELATIVE_CRITERION_TOLERANCE,
    convergence_relative_gradient_tolerance=CONVERGENCE_RELATIVE_GRADIENT_TOLERANCE,
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

    res = scipy.optimize.least_squares(
        fun=criterion,
        x0=x,
        jac=derivative,
        # Don't use get_scipy_bounds, b.c. least_squares uses np.inf
        bounds=(lower_bounds, upper_bounds),
        max_nfev=stopping_max_criterion_evaluations,
        ftol=convergence_relative_criterion_tolerance,
        gtol=convergence_relative_gradient_tolerance,
        method=method,
        diff_step=relative_step_size_diff_approx,
        tr_solver=tr_solver,
        tr_options=tr_solver_options,
    )

    return process_scipy_result(res)


_scipy_ls_trf = functools.partial(_scipy_least_squares, method="trf")
scipy_ls_trf = mark_minimizer(
    _scipy_ls_trf, name="scipy_ls_trf", primary_criterion_entry="root_contributions"
)

_scipy_ls_dogbox = functools.partial(_scipy_least_squares, method="dogbox")
scipy_ls_dogbox = mark_minimizer(
    _scipy_ls_dogbox,
    name="scipy_ls_dogbox",
    primary_criterion_entry="root_contributions",
)


@mark_minimizer(name="scipy_ls_lm", primary_criterion_entry="root_contributions")
def scipy_ls_lm(
    criterion,
    derivative,
    x,
    *,
    convergence_relative_criterion_tolerance=CONVERGENCE_RELATIVE_CRITERION_TOLERANCE,
    convergence_relative_gradient_tolerance=CONVERGENCE_RELATIVE_GRADIENT_TOLERANCE,
    stopping_max_criterion_evaluations=STOPPING_MAX_CRITERION_EVALUATIONS,
    relative_step_size_diff_approx=None,
    tr_solver=None,
    tr_solver_options=None,
):
    """Internal function used by the scipy_ls_trf and scipy_ls_dogbox functions."""

    if tr_solver_options is None:
        tr_solver_options = {}

    res = scipy.optimize.least_squares(
        fun=criterion,
        x0=x,
        jac=derivative,
        max_nfev=stopping_max_criterion_evaluations,
        ftol=convergence_relative_criterion_tolerance,
        gtol=convergence_relative_gradient_tolerance,
        method="lm",
        diff_step=relative_step_size_diff_approx,
        tr_solver=tr_solver,
        tr_options=tr_solver_options,
    )

    return process_scipy_result(res)


@mark_minimizer(name="scipy_basinhopping", is_global=True)
def scipy_basinhopping(
    criterion,
    derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    local_algorithm="L-BFGS-B",
    n_local_optimizations=100,
    temperature=1.0,
    stepsize=0.5,
    local_algo_options=None,
    take_step=None,
    accept_test=None,
    interval=50,
    convergence_n_unchanged_iterations=None,
    seed=None,
    target_accept_rate=0.5,
    stepwise_factor=0.9,
):
    """Find the global minimum of a function using the basin-hopping algorithm.

    For details see :ref:`list_of_scipy_algorithms`.

    """
    n_local_optimizations = max(1, n_local_optimizations - 1)
    local_algo_options = {} if local_algo_options is None else local_algo_options
    default_minimizer_kwargs = {
        "method": local_algorithm,
        "bounds": _get_scipy_bounds(lower_bounds, upper_bounds),
        "jac": derivative,
    }

    minimizer_kwargs = {**default_minimizer_kwargs, **local_algo_options}

    res = scipy.optimize.basinhopping(
        func=criterion,
        x0=x,
        minimizer_kwargs=minimizer_kwargs,
        niter=n_local_optimizations,
        T=temperature,
        stepsize=stepsize,
        take_step=take_step,
        accept_test=accept_test,
        interval=interval,
        niter_success=convergence_n_unchanged_iterations,
        seed=seed,
        target_accept_rate=target_accept_rate,
        stepwise_factor=stepwise_factor,
    )

    return process_scipy_result(res)


@mark_minimizer(name="scipy_brute", is_global=True, disable_history=True)
def scipy_brute(
    criterion,
    lower_bounds,
    upper_bounds,
    x,  # noqa: ARG001
    *,
    n_grid_points=20,
    polishing_function=None,
    n_cores=1,
    batch_evaluator="joblib",
):
    """Minimize a function over a given range by brute force.

    For details see :ref:`list_of_scipy_algorithms`.

    Lower and upper bounds are passed to range where each component
    of the ranges tuple must be either a “slice object” or a range tuple
    of the form (low, high). The program uses these to create the
    grid of points on which the objective function will be computed.

    full_output always true.
    polishing func none to stay inside bounds, bounds - ranges difference.

    """
    workers = _get_workers(n_cores, batch_evaluator)

    res = scipy.optimize.brute(
        func=criterion,
        ranges=tuple(map(tuple, np.column_stack([lower_bounds, upper_bounds]))),
        Ns=n_grid_points,
        full_output=True,
        finish=polishing_function,
        workers=workers,
    )
    out = {
        "solution_x": res[0],
        "solution_criterion": res[1],
        "n_criterion_evaluations": res[2].size,
        "n_iterations": res[2].size,
        "success": True,
        "message": "brute force optimization terminated successfully",
    }

    return out


@mark_minimizer(
    name="scipy_differential_evolution", is_global=True, disable_history=True
)
def scipy_differential_evolution(
    criterion,
    lower_bounds,
    upper_bounds,
    x,  # noqa: ARG001
    nonlinear_constraints,
    *,
    strategy="best1bin",
    stopping_max_iterations=STOPPING_MAX_CRITERION_EVALUATIONS_GLOBAL,
    population_size_multiplier=15,
    convergence_relative_criterion_tolerance=0.01,
    mutation_constant=(0.5, 1),
    recombination_constant=0.7,
    seed=None,
    polish=True,
    sampling_method="latinhypercube",
    convergence_absolute_criterion_tolerance=CONVERGENCE_SECOND_BEST_ABSOLUTE_CRITERION_TOLERANCE,  # noqa: E501
    n_cores=1,
    batch_evaluator="joblib",
):
    """Finds the global minimum of a multivariate function.

    For details see :ref:`list_of_scipy_algorithms`.

    """
    workers = _get_workers(n_cores, batch_evaluator)
    res = scipy.optimize.differential_evolution(
        func=criterion,
        bounds=_get_scipy_bounds(lower_bounds, upper_bounds),
        strategy=strategy,
        maxiter=stopping_max_iterations,
        popsize=population_size_multiplier,
        tol=convergence_relative_criterion_tolerance,
        mutation=mutation_constant,
        recombination=recombination_constant,
        seed=seed,
        polish=polish,
        init=sampling_method,
        atol=convergence_absolute_criterion_tolerance,
        updating="deferred",
        workers=workers,
        constraints=_get_scipy_constraints(nonlinear_constraints),
    )

    return process_scipy_result(res)


@mark_minimizer(name="scipy_shgo", is_global=True)
def scipy_shgo(
    criterion,
    lower_bounds,
    upper_bounds,
    x,  # noqa: ARG001
    derivative,
    nonlinear_constraints,
    *,
    local_algorithm="L-BFGS-B",
    local_algo_options=None,
    n_sampling_points=128,
    n_simplex_iterations=1,
    sampling_method="simplicial",
    max_sampling_evaluations=None,
    convergence_minimum_criterion_value=None,
    convergence_minimum_criterion_tolerance=1e-4,
    stopping_max_iterations=None,
    stopping_max_criterion_evaluations=STOPPING_MAX_CRITERION_EVALUATIONS_GLOBAL,
    stopping_max_processing_time=None,
    minimum_homology_group_rank_differential=None,
    symmetry=False,
    minimize_every_iteration=True,
    max_local_minimizations_per_iteration=False,
    infinity_constraints=True,
):
    """Finds the global minimum of a function using SHG optimization.

    SHGO stands for “simplicial homology global optimization”.

    For details see :ref:`list_of_scipy_algorithms`.

    """
    if local_algorithm == "COBYLA":
        nonlinear_constraints = equality_as_inequality_constraints(
            nonlinear_constraints
        )

    nonlinear_constraints = vector_as_list_of_scalar_constraints(nonlinear_constraints)

    local_algo_options = {} if local_algo_options is None else local_algo_options
    default_minimizer_kwargs = {
        "method": local_algorithm,
        "bounds": _get_scipy_bounds(lower_bounds, upper_bounds),
        "jac": derivative,
    }

    minimizer_kwargs = {**default_minimizer_kwargs, **local_algo_options}
    options = {
        "maxfev": max_sampling_evaluations,
        "f_min": convergence_minimum_criterion_value,
        "f_tol": convergence_minimum_criterion_tolerance,
        "maxiter": stopping_max_iterations,
        "maxev": stopping_max_criterion_evaluations,
        "maxtime": stopping_max_processing_time,
        "minhgrd": minimum_homology_group_rank_differential,
        "symmetry": symmetry,
        "jac": derivative,
        "minimize_every_iter": minimize_every_iteration,
        "local_iter": max_local_minimizations_per_iteration,
        "infty_constraints": infinity_constraints,
    }

    if any(options.values()) is False:
        options = None

    res = scipy.optimize.shgo(
        func=criterion,
        bounds=_get_scipy_bounds(lower_bounds, upper_bounds),
        constraints=nonlinear_constraints,
        minimizer_kwargs=minimizer_kwargs,
        n=n_sampling_points,
        iters=n_simplex_iterations,
        sampling_method=sampling_method,
        options=options,
    )

    return process_scipy_result(res)


@mark_minimizer(name="scipy_dual_annealing", is_global=True)
def scipy_dual_annealing(
    criterion,
    derivative,
    lower_bounds,
    upper_bounds,
    x,
    *,
    stopping_max_iterations=STOPPING_MAX_CRITERION_EVALUATIONS_GLOBAL,
    local_algorithm="L-BFGS-B",
    local_algo_options=None,
    initial_temperature=5230.0,
    restart_temperature_ratio=2e-05,
    visit=2.62,
    accept=-5.0,
    stopping_max_criterion_evaluations=STOPPING_MAX_CRITERION_EVALUATIONS,
    seed=None,
    no_local_search=False,
):
    """Find the global minimum of a function using Dual Annealing.

    For details see :ref:`list_of_scipy_algorithms`.

    """
    local_algo_options = {} if local_algo_options is None else local_algo_options
    default_minimizer_kwargs = {
        "method": local_algorithm,
        "bounds": _get_scipy_bounds(lower_bounds, upper_bounds),
        "jac": derivative,
    }

    minimizer_kwargs = {**default_minimizer_kwargs, **local_algo_options}

    res = scipy.optimize.dual_annealing(
        func=criterion,
        bounds=_get_scipy_bounds(lower_bounds, upper_bounds),
        maxiter=stopping_max_iterations,
        minimizer_kwargs=minimizer_kwargs,
        initial_temp=initial_temperature,
        restart_temp_ratio=restart_temperature_ratio,
        visit=visit,
        accept=accept,
        maxfun=stopping_max_criterion_evaluations,
        seed=seed,
        no_local_search=no_local_search,
        x0=x,
    )

    return process_scipy_result(res)


@mark_minimizer(name="scipy_direct", is_global=True)
def scipy_direct(
    criterion,
    lower_bounds,
    upper_bounds,
    x,  # noqa: ARG001
    *,
    convergence_relative_criterion_tolerance=CONVERGENCE_RELATIVE_CRITERION_TOLERANCE,
    stopping_max_criterion_evaluations=STOPPING_MAX_CRITERION_EVALUATIONS,
    stopping_max_iterations=STOPPING_MAX_CRITERION_EVALUATIONS_GLOBAL,
    locally_biased=True,
    convergence_minimum_criterion_value=-np.inf,
    convergence_minimum_criterion_tolerance=1e-4,
    volume_hyperrectangle_tolerance=1e-16,
    length_hyperrectangle_tolerance=1e-6,
):
    """Finds the global minimum of a function using the DIRECT algorithm.

    For details see :ref:`list_of_scipy_algorithms`.

    """

    res = scipy.optimize.direct(
        func=criterion,
        bounds=_get_scipy_bounds(lower_bounds, upper_bounds),
        eps=convergence_relative_criterion_tolerance,
        maxfun=stopping_max_criterion_evaluations,
        maxiter=stopping_max_iterations,
        locally_biased=locally_biased,
        f_min=convergence_minimum_criterion_value,
        f_min_rtol=convergence_minimum_criterion_tolerance,
        vol_tol=volume_hyperrectangle_tolerance,
        len_tol=length_hyperrectangle_tolerance,
    )

    return process_scipy_result(res)


def _get_workers(n_cores, batch_evaluator):
    batch_evaluator = process_batch_evaluator(batch_evaluator)
    out = functools.partial(
        batch_evaluator,
        n_cores=n_cores,
        error_handling="raise",
    )
    return out
