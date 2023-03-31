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
        sphere_solver (str or callable): Name of a subproblem solver or a subproblem
            solver, designed to solve the problem in the unit sphere. The first argument
            of any subsolver needs to be ``model``. The second argument needs to be
            ``x_candidate``, an initial guess for the solution in the unit space.
            Moreover, subsolvers can have any number of additional keyword arguments.
        cube_solver (str or callable): Name of a subproblem solver or a subproblem
            solver, designed to solve the problem in the unit box. The first argument
            of any subsolver needs to be ``model``. The second and third arguments have
            to be ``lower_bounds`` and ``upper_bounds``. The fourth argument needs to be
            ``x_candidate``, an initial guess for the solution in the unit space.
            Moreover, subsolvers can have any number of additional keyword arguments.
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
        model (ScalarModel): The fitted model of which we want to find the minimum.
        trustregion (Region): The trustregion on which the model was fitted.
        sphere_solver (callable): Spherical subproblem solver, designed to solve the
            problem in the unit sphere. The first argument of any subsolver needs to be
            ``model``. The second argument needs to be ``x_candidate``, an initial guess
            for the solution in the unit space. Moreover, subsolvers can have any number
            of additional keyword arguments.
        cube_solver (callable): Cubical subproblem solver, designed to solve the problem
            in the unit box. The first argument of any subsolver needs to be ``model``.
            The second and third arguments have to be ``lower_bounds`` and
            ``upper_bounds``. The fourth argument needs to be ``x_candidate``, an
            initial guess for the solution in the unit space. Moreover, subsolvers can
            have any number of additional keyword arguments.


    Returns:
        SubproblemResult: Namedtuple with the following entries:
            - "x" (np.ndarray): The optimal x in terms of the original parameter space.
            - "expected_improvement" (float): The expected improvement at the solution.
              The sign has already been flipped, i.e. large means more improvement.
            - "n_iterations" (int): Number of iterations performed before termination.
            - "success" (bool): Boolean indicating whether a solution has been found
              before reaching maxiter.
            - "x_unit" (np.ndarray): The optimal x in terms of the unit space.
            - "shape" (str): Whether the trustregion was a sphere or a cube, which in
              turn determines whether the sphere or cube solver was used.

    """
    old_x_unit = trustregion.map_to_unit(trustregion.center)

    solver = sphere_solver if trustregion.shape == "sphere" else cube_solver

    raw_result = solver(
        model=model,
        x_candidate=old_x_unit,
        # bounds can be passed to both solvers because the functions returned by
        # `get_component` ignore redundant arguments.
        lower_bounds=-np.ones_like(old_x_unit),
        upper_bounds=np.ones_like(old_x_unit),
    )

    if trustregion.shape == "cube":
        raw_result["x"] = np.clip(raw_result["x"], -1.0, 1.0)

    # make sure expected improvement is calculated accurately in case of clipping and
    # does not depend on whether the subsolver ignores intercepts or not.
    fval_old = model.predict(old_x_unit)
    fval_candidate = model.predict(raw_result["x"])

    expected_improvement = -(fval_candidate - fval_old)

    # in case of negative expected improvement, we return the old point
    if expected_improvement >= 0:
        success = raw_result["success"]
        x_unit = raw_result["x"]
        x = trustregion.map_from_unit(raw_result["x"])
    else:
        success = False
        x_unit = old_x_unit
        x = trustregion.center
        expected_improvement = 0.0

    result = SubproblemResult(
        x=x,
        expected_improvement=expected_improvement,
        n_iterations=raw_result["n_iterations"],
        success=success,
        x_unit=x_unit,
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
