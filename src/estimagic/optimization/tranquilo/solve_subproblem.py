import inspect
import warnings
from functools import partial
from typing import NamedTuple

import numpy as np

from estimagic.optimization.subsolvers.bntr import (
    bntr,
)
from estimagic.optimization.subsolvers.bntr_fast import (
    bntr_fast,
)
from estimagic.optimization.subsolvers.gqtpar import (
    gqtpar,
)
from estimagic.optimization.subsolvers.gqtpar_fast import gqtpar_fast
from estimagic.optimization.tranquilo.wrapped_subsolvers import (
    slsqp_sphere,
    solve_multistart,
)
from estimagic.optimization.tranquilo.options import SubsolverOptions


def get_subsolver(solver, user_options=None, bounds=None):
    """Get an algorithm-function with partialled options.

    Args:
        solver (str or callable): Name of a subproblem solver or subproblem solver. The
            first argument of any subsolver needs to be ``model``. If the solver
            supports bounds, the next arguments have to be ``lower_bounds`` and
            ``upper_bounds``. Moreover, subsolvers can have any number of additional
            keyword arguments.
        user_options (dict):
            Options for the subproblem solver. The following are supported:
            - maxiter (int): Maximum number of iterations to perform when solving the
                trust-region subproblem ("bntr" and "gqtpar").
            - maxiter_gradient_descent (int): Maximum number of gradient descent
                iterations to perform ("bntr").
            - conjugate_gradient_method (str): Method for computing the conjugate
                gradient step ("bntr").
                Available conjugate gradient methods are:
                - "cg"
                - "steihaug_toint"
                - "trsbox" (default)
            - gtol_abs (float): Convergence tolerance for the absolute gradient norm
                in the trust-region subproblem ("bntr").
            - gtol_rel (float): Convergence tolerance for the relative gradient norm
                in the trust-region subproblem ("bntr").
            - gtol_scaled (float): Convergence tolerance for the scaled gradient norm
                in the trust-region subproblem ("bntr").
            - gtol_abs_conjugate_gradient (float): Convergence tolerance for the
                absolute gradient norm in the conjugate gradient step ("bntr").
            - gtol_rel_conjugate_gradient (float): Convergence tolerance for the
                relative gradient norm in the conjugate gradient step ("bntr").
            - k_easy (float): topping criterion for the "easy" case in the trust-region
                subproblem ("gqtpar").
            - k_hard (float): Stopping criterion for the "hard" case in the trust-region
                subproblem ("gqtpar").
        bounds (Bounds):

    Returns:
        callable: The subsolver.

    """
    user_options = {} if user_options is None else user_options

    built_in_solvers = {
        "bntr": bntr,
        "bntr_fast": bntr_fast,
        "gqtpar": gqtpar,
        "gqtpar_fast": gqtpar_fast,
        "multistart": solve_multistart,
        "slsqp_sphere": slsqp_sphere,
    }

    if isinstance(solver, str) and solver in built_in_solvers:
        _solver = built_in_solvers[solver]
        _solver_name = solver
    elif callable(solver):
        _solver = solver
        _solver_name = getattr(solver, "__name__", "your solver")
    else:
        raise ValueError(
            "Invalid solver: {solver}. Must be one of {list(built_in_solvers)} "
            "or a callable."
        )

    all_options = SubsolverOptions()._replace(**user_options)._asdict()

    args = set(inspect.signature(_solver).parameters)

    if "model" not in args:
        raise ValueError("subproblem solvers need to take 'model' as first argument.")

    valid_bounds = {"lower_bounds", "upper_bounds"}.intersection(args)

    bounds_dict = {"lower_bounds": None, "upper_bounds": None}
    if bounds is not None and bounds.has_any:
        for type_ in ["lower", "upper"]:
            candidate = getattr(bounds, type_)
            if candidate is not None and np.isfinite(candidate).any():
                bounds_dict[f"{type_}_bounds"] = candidate

    for name, value in bounds_dict.items():
        if name not in valid_bounds and value is not None:
            raise ValueError(
                f"You have {name} but requested a subproblem solver that does not "
                "support them. Use bntr or another bounded subproblem solver instead."
            )

    bounds_dict = {k: v for k, v in bounds_dict.items() if k in valid_bounds}

    not_options = {"model"} | valid_bounds
    if isinstance(_solver, partial):
        partialed_in = set(_solver.args).union(set(_solver.keywords))
        not_options = not_options | partialed_in

    valid_options = args - not_options

    reduced = {key: val for key, val in all_options.items() if key in valid_options}

    ignored = {
        key: val for key, val in user_options.items() if key not in valid_options
    }

    if ignored:
        warnings.warn(
            "The following options were ignored because they are not compatible "
            f"with {_solver_name}:\n\n {ignored}"
        )

    out = partial(
        _solve_subproblem_template, solver=_solver, bounds=bounds_dict, options=reduced
    )

    return out


def _solve_subproblem_template(
    model,
    trustregion,
    solver,
    bounds,
    options,
):
    """Solve the quadratic subproblem.

    Args:
        model (NamedTuple): NamedTuple containing the parameters of the fitted surrogate
            model, i.e. ``linear_terms`` and ``square_terms``. The model is assumed to
            be defined in terms of centered and scaled parameter vectors.
        trustregion (NamedTuple): Contains ``center`` (np.ndarray) and ``radius``
            (float). Used to center bounds.
        solver (callable): Trust-region subsolver to use. All options must already be
            partialled in such that the subsolver only depends on ``model``,
            ``lower_bounds`` and ``upper_bounds``
        bounds (dict): Dict containing the entries "lower_bounds" and "upper_bounds"
            Bounds are assumed to be in terms of the original parameter space, i.e. not
            centered yet.
        options (dict): Solver specific options.


    Returns:
        (dict): Result dictionary containing the following entries:
            - "x" (np.ndarray): The optimal x in terms of the original parameter space.
            - "expected_improvement" (float): The expected improvement at the solution.
              The sign has already been flipped, i.e. large means more improvement.
            - "n_iterations" (int): Number of iterations performed before termination.
            - "success" (bool): Boolean indicating whether a solution has been found
              before reaching maxiter.

    """

    _bounds = _get_centered_and_scaled_bounds(bounds, trustregion)

    _center = np.zeros_like(trustregion.center)

    raw_result = solver(model, **_bounds, **options, x_candidate=_center)

    x = trustregion.map_from_unit(raw_result["x"])

    if "lower_bounds" in bounds:
        x = np.clip(x, bounds["lower_bounds"], np.inf)

    if "upper_bounds" in bounds:
        x = np.clip(x, -np.inf, bounds["upper_bounds"])

    # make sure expected improvement is calculated accurately in case of clipping and
    # does not depend on whether the subsolver ignores intercepts or not.
    fval_old = model.predict(trustregion.map_to_unit(trustregion.center))
    fval_candidate = model.predict(raw_result["x"])

    expected_improvement = -(fval_candidate - fval_old)

    result = SubproblemResult(
        x=x,
        expected_improvement=expected_improvement,
        n_iterations=raw_result["n_iterations"],
        success=raw_result["success"],
        x_unit=raw_result["x"],
        shape=trustregion.shape,
    )

    return result


def _get_centered_and_scaled_bounds(bounds, trustregion):
    out = {}
    n_params = len(trustregion.center)
    if "lower_bounds" in bounds:
        if bounds["lower_bounds"] is None:
            lower_bounds = np.full(n_params, -1)
        else:
            lower_bounds = trustregion.map_to_unit(bounds["lower_bounds"])
            lower_bounds = np.nan_to_num(lower_bounds, nan=-1, neginf=-1)
        out["lower_bounds"] = lower_bounds

    if "upper_bounds" in bounds:
        if bounds["upper_bounds"] is None:
            upper_bounds = np.ones(n_params)
        else:
            upper_bounds = trustregion.map_to_unit(bounds["upper_bounds"])
            upper_bounds = np.nan_to_num(upper_bounds, nan=1, posinf=1)
        out["upper_bounds"] = upper_bounds
    return out


class SubproblemResult(NamedTuple):
    """Result of the subproblem solver."""

    x: np.ndarray
    expected_improvement: float
    n_iterations: int
    success: bool
    x_unit: np.ndarray
    shape: str
