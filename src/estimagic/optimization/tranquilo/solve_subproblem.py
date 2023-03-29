from functools import partial
from typing import NamedTuple

import numpy as np

from estimagic.optimization.tranquilo.get_component import get_component
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


def get_subsolver(sphere_solver, cube_solver, user_options=None):
    """Get an algorithm-function with partialled options.

    Args:
        sphere_solver (str or callable): Name of a subproblem solver or subproblem
            solver. The first argument of any subsolver needs to be ``model``. If the
            solver supports bounds, the next arguments have to be ``lower_bounds`` and
            ``upper_bounds``. Moreover, subsolvers can have any number of additional
            keyword arguments.
        cube_solver (str or callable): Name of a subproblem solver or subproblem solver.
            The first argument of any subsolver needs to be ``model``. If the solver
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

    Returns:
        callable: The subsolver.

    """
    built_in_sphere_solvers = {
        "gqtpar": gqtpar,
        "gqtpar_fast": gqtpar_fast,
        "multistart": solve_multistart,
        "slsqp_sphere": slsqp_sphere,
    }

    built_in_cube_solvers = {
        "bntr": bntr,
        "bntr_fast": bntr_fast,
        "multistart": solve_multistart,
    }

    _sphere_subsolver = get_component(
        name_or_func=sphere_solver,
        component_name="sphere_solver",
        func_dict=built_in_sphere_solvers,
        default_options=SubsolverOptions(),
        user_options=user_options,
        mandatory_signature=["model", "x_candidate"],
    )

    _cube_subsolver = get_component(
        name_or_func=cube_solver,
        component_name="cube_solver",
        func_dict=built_in_cube_solvers,
        default_options=SubsolverOptions(),
        user_options=user_options,
        mandatory_signature=["model", "x_candidate", "lower_bounds", "upper_bounds"],
    )

    solver = partial(
        _solve_subproblem_template,
        sphere_solver=_sphere_subsolver,
        cube_solver=_cube_subsolver,
    )

    return solver


def _solve_subproblem_template(
    model,
    trustregion,
    sphere_solver,
    cube_solver,
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


    Returns:
        (dict): Result dictionary containing the following entries:
            - "x" (np.ndarray): The optimal x in terms of the original parameter space.
            - "expected_improvement" (float): The expected improvement at the solution.
              The sign has already been flipped, i.e. large means more improvement.
            - "n_iterations" (int): Number of iterations performed before termination.
            - "success" (bool): Boolean indicating whether a solution has been found
              before reaching maxiter.

    """
    old_x_unit = trustregion.map_to_unit(trustregion.center)

    if trustregion.shape == "sphere":
        raw_result = sphere_solver(model, x_candidate=old_x_unit)
    else:
        raw_result = cube_solver(
            model=model,
            x_candidate=old_x_unit,
            lower_bounds=-np.ones_like(old_x_unit),
            upper_bounds=np.ones_like(old_x_unit),
        )

    x = trustregion.map_from_unit(raw_result["x"])

    if trustregion.bounds is not None and trustregion.bounds.has_any:
        x = np.clip(x, trustregion.bounds.lower, np.inf)
        x = np.clip(x, -np.inf, trustregion.bounds.upper)

    # make sure expected improvement is calculated accurately in case of clipping and
    # does not depend on whether the subsolver ignores intercepts or not.
    fval_old = model.predict(old_x_unit)
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


class SubproblemResult(NamedTuple):
    """Result of the subproblem solver."""

    x: np.ndarray
    expected_improvement: float
    n_iterations: int
    success: bool
    x_unit: np.ndarray
    shape: str
