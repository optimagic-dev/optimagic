from typing import NamedTuple

import numpy as np
from estimagic.optimization.subsolvers.bounded_newton_quadratic import (
    _evaluate_model_criterion as eval_criterion_orig,
)
from estimagic.optimization.subsolvers.bounded_newton_quadratic import (
    _get_fischer_burmeister_direction_vector as fb_vector_orig,
)
from estimagic.optimization.subsolvers.bounded_newton_quadratic import ActiveBounds
from estimagic.optimization.subsolvers.bounded_newton_quadratic import (
    apply_bounds_to_x_candidate as apply_bounds_orig,
)
from estimagic.optimization.subsolvers.bounded_newton_quadratic import (
    find_hessian_submatrix_where_bounds_inactive as find_hessian_inact_orig,
)
from estimagic.optimization.subsolvers.bounded_newton_quadratic import (
    get_information_on_active_bounds as get_info_bounds_orig,
)
from estimagic.optimization.subsolvers.bounded_newton_quadratic import (
    project_gradient_onto_feasible_set as grad_feas_orig,
)
from estimagic.optimization.subsolvers.bounded_newton_quadratic import (
    take_preliminary_gradient_descent_step_and_check_for_solution as pgd_orig,
)
from estimagic.optimization.subsolvers.bounded_newton_quadratic_fast import (
    _evaluate_model_criterion as eval_criterion_fast,
)
from estimagic.optimization.subsolvers.bounded_newton_quadratic_fast import (
    _get_fischer_burmeister_direction_vector as fb_vector_fast,
)
from estimagic.optimization.subsolvers.bounded_newton_quadratic_fast import (
    apply_bounds_to_x_candidate as apply_bounds_fast,
)
from estimagic.optimization.subsolvers.bounded_newton_quadratic_fast import (
    find_hessian_submatrix_where_bounds_inactive as find_hessian_inact_fast,
)
from estimagic.optimization.subsolvers.bounded_newton_quadratic_fast import (
    get_information_on_active_bounds as get_info_bounds_fast,
)
from estimagic.optimization.subsolvers.bounded_newton_quadratic_fast import (
    project_gradient_onto_feasible_set as grad_feas_fast,
)
from estimagic.optimization.subsolvers.bounded_newton_quadratic_fast import (
    take_preliminary_gradient_descent_step_and_check_for_solution as pgd_fast,
)
from numpy.testing import assert_array_equal as aae


def test_eval_criterion():
    x_candidate = np.zeros(5)
    linear_terms = np.arange(5).astype(float)
    square_terms = np.arange(25).reshape(5, 5).astype(float)
    assert eval_criterion_orig(
        x_candidate, linear_terms, square_terms
    ) == eval_criterion_fast(x_candidate, linear_terms, square_terms)


def test_get_info_on_active_bounds():
    x_candidate = np.array([-1.5, -1.5, 0, 1.5, 1.5])
    linear_terms = np.array([1, 1, 0, -1, -1])
    lower_bounds = -np.ones(5)
    upper_bounds = np.ones(5)
    info_orig = get_info_bounds_orig(
        x_candidate, linear_terms, lower_bounds, upper_bounds
    )
    info_fast = get_info_bounds_fast(
        x_candidate, linear_terms, lower_bounds, upper_bounds
    )
    aae(info_orig.lower, info_fast.lower)
    aae(info_orig.upper, info_fast.upper)
    aae(info_orig.fixed, info_fast.fixed)
    aae(info_orig.active, info_fast.active)
    aae(info_orig.inactive, info_fast.inactive)


def test_project_gradient_on_feasible_set():
    grad = np.arange(5).astype(float)
    bounds_info = ActiveBounds(
        inactive=np.array([0, 1, 2]),
    )
    aae(grad_feas_orig(grad, bounds_info), grad_feas_fast(grad, bounds_info))


def test_find_hessian_inactive_bounds():
    hessian = np.arange(25).reshape(5, 5).astype(float)

    class Model(NamedTuple):
        square_terms: np.ndarray = hessian

    bounds_info = ActiveBounds(
        inactive=np.array([2, 3, 4]),
    )
    aae(
        find_hessian_inact_orig(Model(), bounds_info),
        find_hessian_inact_fast(Model(), bounds_info),
    )


def test_fb_vector():
    x = np.array([-1.5, -1.5, 0, 1.5, 1.5])
    grad = np.ones(5)
    lb = -np.ones(5)
    ub = np.ones(5)
    aae(fb_vector_orig(x, grad, lb, ub), fb_vector_fast(x, grad, lb, ub))


def test_applyt_bounds_candidate_x():
    x = np.array([-1.5, -1.5, 0, 1.5, 1.5])
    lb = -np.ones(5)
    ub = np.ones(5)
    aae(apply_bounds_orig(x, lb, ub), apply_bounds_fast(x, lb, ub))


def test_prelim_grad_descent():
    class Model(NamedTuple):
        linear_terms: np.ndarray = np.array(
            [
                -5.71290e02,
                -3.11506e03,
                -8.18100e02,
                2.47760e02,
                -1.26540e02,
            ]
        )
        square_terms: np.ndarray = np.array(
            [
                [-619.23, -1229.2, 321.9, 106.98, -45.45],
                [-1229.2, -668.95, -250.05, 165.77, -47.47],
                [321.9, -250.05, -1456.88, -144.75, 900.99],
                [106.98, 165.77, -144.75, 686.35, -3.51],
                [-45.45, -47.47, 900.99, -3.51, -782.91],
            ]
        )

    x_candidate = np.zeros(5)
    lower_bounds = -np.ones(len(x_candidate))
    upper_bounds = np.ones(len(x_candidate))
    kwargs = {
        "x_candidate": x_candidate,
        "model": Model(),
        "lower_bounds": lower_bounds,
        "upper_bounds": upper_bounds,
        "maxiter_gradient_descent": 5,
        "gtol_abs": 1e-08,
        "gtol_rel": 1e-08,
        "gtol_scaled": 0,
    }
    res_speedup = pgd_fast(**kwargs)
    res_orig = pgd_orig(**kwargs)
    for i in range(5):
        aae(np.array(res_speedup[i]), np.array(res_orig[i]))
    bounds_info_orig = res_orig[5]
    bounds_info_fast = res_orig[5]
    for bounds in ["lower", "upper", "fixed", "active", "inactive"]:
        aae(
            np.array(getattr(bounds_info_orig, bounds)),
            np.array(getattr(bounds_info_fast, bounds)),
        )
    assert res_orig[6] == res_speedup[6]
