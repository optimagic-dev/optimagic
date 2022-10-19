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
