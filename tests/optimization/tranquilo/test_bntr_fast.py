import numpy as np
import pandas as pd
from estimagic.config import TEST_FIXTURES_DIR
from estimagic.optimization.subsolvers.bntr import (
    _apply_bounds_to_conjugate_gradient_step as bounds_cg_orig,
)
from estimagic.optimization.subsolvers.bntr import (
    _apply_bounds_to_x_candidate as apply_bounds_orig,
)
from estimagic.optimization.subsolvers.bntr import (
    _compute_conjugate_gradient_step as cg_step_orig,
)
from estimagic.optimization.subsolvers.bntr import (
    _compute_predicted_reduction_from_conjugate_gradient_step as reduction_cg_step_orig,
)
from estimagic.optimization.subsolvers.bntr import (
    _evaluate_model_criterion as eval_criterion_orig,
)
from estimagic.optimization.subsolvers.bntr import (
    _find_hessian_submatrix_where_bounds_inactive as find_hessian_inact_orig,
)
from estimagic.optimization.subsolvers.bntr import (
    _get_fischer_burmeister_direction_vector as fb_vector_orig,
)
from estimagic.optimization.subsolvers.bntr import (
    _get_information_on_active_bounds as get_info_bounds_orig,
)
from estimagic.optimization.subsolvers.bntr import (
    _perform_gradient_descent_step as gradient_descent_orig,
)
from estimagic.optimization.subsolvers.bntr import (
    _project_gradient_onto_feasible_set as grad_feas_orig,
)
from estimagic.optimization.subsolvers.bntr import (
    _take_preliminary_gradient_descent_step_and_check_for_solution as pgd_orig,
)
from estimagic.optimization.subsolvers.bntr import (
    _update_trustregion_radius_and_gradient_descent,
)
from estimagic.optimization.subsolvers.bntr import (
    _update_trustregion_radius_conjugate_gradient as update_radius_cg_orig,
)
from estimagic.optimization.subsolvers.bntr import ActiveBounds
from estimagic.optimization.subsolvers.bntr import (
    bntr,
)
from estimagic.optimization.subsolvers.bntr_fast import (
    _apply_bounds_to_conjugate_gradient_step as bounds_cg_fast,
)
from estimagic.optimization.subsolvers.bntr_fast import (
    _apply_bounds_to_x_candidate as apply_bounds_fast,
)
from estimagic.optimization.subsolvers.bntr_fast import (
    _bntr_fast_jitted,
)
from estimagic.optimization.subsolvers.bntr_fast import (
    _compute_conjugate_gradient_step as cg_step_fast,
)
from estimagic.optimization.subsolvers.bntr_fast import (
    _compute_predicted_reduction_from_conjugate_gradient_step as reduction_cg_step_fast,
)
from estimagic.optimization.subsolvers.bntr_fast import (
    _evaluate_model_criterion as eval_criterion_fast,
)
from estimagic.optimization.subsolvers.bntr_fast import (
    _find_hessian_submatrix_where_bounds_inactive as find_hessian_inact_fast,
)
from estimagic.optimization.subsolvers.bntr_fast import (
    _get_fischer_burmeister_direction_vector as fb_vector_fast,
)
from estimagic.optimization.subsolvers.bntr_fast import (
    _get_information_on_active_bounds as get_info_bounds_fast,
)
from estimagic.optimization.subsolvers.bntr_fast import (
    _perform_gradient_descent_step as gradient_descent_fast,
)
from estimagic.optimization.subsolvers.bntr_fast import (
    _project_gradient_onto_feasible_set as grad_feas_fast,
)
from estimagic.optimization.subsolvers.bntr_fast import (
    _take_preliminary_gradient_descent_step_and_check_for_solution as pgd_fast,
)
from estimagic.optimization.subsolvers.bntr_fast import (
    _update_trustregion_radius_and_gradient_descent as _update_trr_and_gd_fast,
)
from estimagic.optimization.subsolvers.bntr_fast import (
    _update_trustregion_radius_conjugate_gradient as update_radius_cg_fast,
)
from estimagic.optimization.tranquilo.models import ScalarModel
from numpy.testing import assert_array_almost_equal as aaae
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
    indices = np.arange(len(x_candidate))
    linear_terms = np.array([1, 1, 0, -1, -1])
    lower_bounds = -np.ones(5)
    upper_bounds = np.ones(5)
    info_orig = get_info_bounds_orig(
        x_candidate, linear_terms, lower_bounds, upper_bounds
    )
    (
        active_lower,
        active_upper,
        active_fixed,
        inactive,
    ) = get_info_bounds_fast(x_candidate, linear_terms, lower_bounds, upper_bounds)
    aae(info_orig.lower, indices[active_lower])
    aae(info_orig.upper, indices[active_upper])
    aae(info_orig.fixed, indices[active_fixed])
    aae(info_orig.active, indices[~inactive])
    aae(info_orig.inactive, indices[inactive])


def test_project_gradient_on_feasible_set():
    grad = np.arange(5).astype(float)
    bounds_info = ActiveBounds(
        inactive=np.array([0, 1, 2]),
    )
    inactive = np.array([True, True, True, False, False])
    aae(grad_feas_orig(grad, bounds_info), grad_feas_fast(grad, inactive))


def test_find_hessian_inactive_bounds():
    hessian = np.arange(25).reshape(5, 5).astype(float)
    inactive = np.array([False, False, True, True, True])
    model = ScalarModel(square_terms=hessian, intercept=0, linear_terms=np.zeros(5))

    bounds_info = ActiveBounds(
        inactive=np.arange(5)[inactive],
    )

    aae(
        find_hessian_inact_orig(model, bounds_info),
        find_hessian_inact_fast(hessian, inactive),
    )


def test_fischer_burmeister_direction_vector():
    x = np.array([-1.5, -1.5, 0, 1.5, 1.5])
    grad = np.ones(5)
    lb = -np.ones(5)
    ub = np.ones(5)
    aae(fb_vector_orig(x, grad, lb, ub), fb_vector_fast(x, grad, lb, ub))


def test_apply_bounds_candidate_x():
    x = np.array([-1.5, -1.5, 0, 1.5, 1.5])
    lb = -np.ones(5)
    ub = np.ones(5)
    aae(apply_bounds_orig(x, lb, ub), apply_bounds_fast(x, lb, ub))


def test_take_preliminary_gradient_descent_and_check_for_convergence():
    model_gradient = np.array(
        [
            -5.71290e02,
            -3.11506e03,
            -8.18100e02,
            2.47760e02,
            -1.26540e02,
        ]
    )
    model_hessian = np.array(
        [
            [-619.23, -1229.2, 321.9, 106.98, -45.45],
            [-1229.2, -668.95, -250.05, 165.77, -47.47],
            [321.9, -250.05, -1456.88, -144.75, 900.99],
            [106.98, 165.77, -144.75, 686.35, -3.51],
            [-45.45, -47.47, 900.99, -3.51, -782.91],
        ]
    )
    model = ScalarModel(
        linear_terms=model_gradient, square_terms=model_hessian, intercept=0
    )
    x_candidate = np.zeros(5)
    lower_bounds = -np.ones(len(x_candidate))
    upper_bounds = np.ones(len(x_candidate))
    kwargs = {
        "x_candidate": x_candidate,
        "model": model,
        "lower_bounds": lower_bounds,
        "upper_bounds": upper_bounds,
        "maxiter_gradient_descent": 5,
        "gtol_abs": 1e-08,
        "gtol_rel": 1e-08,
        "gtol_scaled": 0,
    }
    kwargs_fast = {
        "model_gradient": model_gradient,
        "model_hessian": model_hessian,
        "lower_bounds": lower_bounds,
        "upper_bounds": upper_bounds,
        "maxiter_gradient_descent": 5,
        "gtol_abs": 1e-08,
        "gtol_rel": 1e-08,
        "gtol_scaled": 0,
    }
    res_fast = pgd_fast(**kwargs_fast)
    res_orig = pgd_orig(**kwargs)
    for i in range(5):
        aae(np.array(res_fast[i]), np.array(res_orig[i]))
    bounds_info_orig = res_orig[5]
    indices = np.arange(5)
    for i, bounds in enumerate(["lower", "upper", "fixed", "inactive"]):
        aae(
            np.array(getattr(bounds_info_orig, bounds)),
            indices[res_fast[5 + i]],
        )
    assert res_orig[6] == res_fast[10]


def test_apply_bounds_to_conjugate_gradient_step():
    step_inactive = np.ones(7)
    x_candidate = np.zeros(10)
    lower_bounds = -np.ones(10)
    upper_bounds = np.array([1] * 7 + [-0.01] * 3)
    indices = np.arange(len(x_candidate))
    inactive_bounds = np.array([True] * 7 + [False] * 3)
    active_lower_bounds = np.array([False] * 10)
    active_upper_bounds = np.array([False] * 7 + [True] * 3)
    active_fixed_bounds = np.array([False] * 10)
    bounds_info = ActiveBounds(
        lower=indices[active_lower_bounds],
        upper=indices[active_upper_bounds],
        fixed=indices[active_fixed_bounds],
        inactive=indices[inactive_bounds],
    )
    res_fast = bounds_cg_fast(
        step_inactive,
        x_candidate,
        lower_bounds,
        upper_bounds,
        inactive_bounds,
        active_lower_bounds,
        active_upper_bounds,
        active_fixed_bounds,
    )
    res_orig = bounds_cg_orig(
        step_inactive, x_candidate, lower_bounds, upper_bounds, bounds_info
    )
    aae(res_orig, res_fast)
    pass


def test_compute_conjugate_gradient_setp():
    x_candidate = np.array([0] * 8 + [1.5] * 2)
    gradient_inactive = np.arange(6).astype(float)
    hessian_inactive = np.arange(36).reshape(6, 6).astype(float)
    lower_bounds = np.array([-1] * 6 + [0.5] * 2 + [-1] * 2)
    upper_bounds = np.ones(10)
    indices = np.arange(len(x_candidate))
    inactive = np.array([True] * 6 + [False] * 4)
    active_lower = np.array([False] * 5 + [True, True] + [False] * 3)
    active_upper = np.array([False] * 8 + [True] * 2)
    active_fixed = np.array([False] * 10)
    bounds_info = ActiveBounds(
        inactive=indices[inactive],
        lower=indices[active_lower],
        upper=indices[active_upper],
        fixed=indices[active_fixed],
    )
    tr_radius = 10.0
    cg_method = "trsbox"
    gtol_abs = 1e-8
    gtol_rel = 1e-8
    default_radius = 100.00
    min_radius = 1e-10
    max_radius = 1e10

    res_fast = cg_step_fast(
        x_candidate,
        gradient_inactive,
        hessian_inactive,
        lower_bounds,
        upper_bounds,
        inactive,
        active_lower,
        active_upper,
        active_fixed,
        tr_radius,
        cg_method,
        gtol_abs,
        gtol_rel,
        default_radius,
        min_radius,
        max_radius,
    )
    res_orig = cg_step_orig(
        x_candidate=x_candidate,
        gradient_inactive=gradient_inactive,
        hessian_inactive=hessian_inactive,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        active_bounds_info=bounds_info,
        trustregion_radius=tr_radius,
        conjugate_gradient_method=cg_method,
        gtol_abs_conjugate_gradient=gtol_abs,
        gtol_rel_conjugate_gradient=gtol_rel,
        options_update_radius={
            "default_radius": default_radius,
            "min_radius": min_radius,
            "max_radius": max_radius,
        },
    )
    aae(res_orig[0], res_fast[0])
    aae(res_orig[1], res_fast[1])
    aaae(res_orig[2], res_fast[2])


def test_compute_predicet_reduction_from_conjugate_gradient_step():
    cg_step = np.arange(10).astype(float) / 10
    cg_step_inactive = np.array([1, 2, 3]).astype(float)
    grad = np.arange(10).astype(float)
    grad_inactive = np.arange(3).astype(float)
    hessian_inactive = np.arange(9).reshape(3, 3).astype(float)
    indices = np.arange(10)
    inactive_bounds = np.array([False] + [True] * 3 + [False] * 6)
    res_fast = reduction_cg_step_fast(
        cg_step,
        cg_step_inactive,
        grad,
        grad_inactive,
        hessian_inactive,
        inactive_bounds,
    )
    bounds_info = ActiveBounds(
        inactive=indices[inactive_bounds], active=indices[~inactive_bounds]
    )
    res_orig = reduction_cg_step_orig(
        cg_step, cg_step_inactive, grad, grad_inactive, hessian_inactive, bounds_info
    )
    aae(res_orig, res_fast)


def test_update_trustregion_radius_conjugate_gradient():
    f_candidate = -1234.56
    predicted_reduction = 200
    actual_reduction = 150
    x_norm_cg = 3.16
    tr_radius = 5
    options_update_radius = {
        "eta1": 1.0e-4,
        "eta2": 0.25,
        "eta3": 0.50,
        "eta4": 0.90,
        "alpha1": 0.25,
        "alpha2": 0.50,
        "alpha3": 1.00,
        "alpha4": 2.00,
        "alpha5": 4.00,
        "min_radius": 1e-10,
        "max_radius": 1e10,
    }
    res_fast = update_radius_cg_fast(
        f_candidate=f_candidate,
        predicted_reduction=predicted_reduction,
        actual_reduction=actual_reduction,
        x_norm_cg=x_norm_cg,
        trustregion_radius=tr_radius,
        **options_update_radius,
    )
    res_orig = update_radius_cg_orig(
        f_candidate=f_candidate,
        predicted_reduction=predicted_reduction,
        actual_reduction=actual_reduction,
        x_norm_cg=x_norm_cg,
        trustregion_radius=tr_radius,
        options=options_update_radius,
    )
    assert res_orig[0] == res_fast[0]
    assert res_orig[1] == res_fast[1]


def test_perform_gradient_descent_step():
    x_candidate = np.zeros(10)
    f_candidate_initial = 1234.56
    gradient_projected = np.arange(10).astype(float)
    hessian_inactive = np.arange(64).reshape(8, 8).astype(float)
    model_gradient = gradient_projected / 2
    model_hessian = np.arange(100).reshape(10, 10).astype(float)
    lower_bounds = -np.ones(10)
    upper_bounds = np.array([1] * 8 + [-0.01] * 2)
    indices = np.arange(10)
    inactive_bounds = np.array([True] * 8 + [False] * 2)

    maxiter = 3
    options_update_radius = {
        "mu1": 0.35,
        "mu2": 0.50,
        "gamma1": 0.0625,
        "gamma2": 0.5,
        "gamma3": 2.0,
        "gamma4": 5.0,
        "theta": 0.25,
        "default_radius": 100,
    }
    model = ScalarModel(
        linear_terms=model_gradient, square_terms=model_hessian, intercept=0
    )
    bounds_info = ActiveBounds(inactive=indices[inactive_bounds])
    res_fast = gradient_descent_fast(
        x_candidate=x_candidate,
        f_candidate_initial=f_candidate_initial,
        gradient_projected=gradient_projected,
        hessian_inactive=hessian_inactive,
        model_gradient=model_gradient,
        model_hessian=model_hessian,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        inactive_bounds=inactive_bounds,
        maxiter_steepest_descent=maxiter,
        **options_update_radius,
    )
    res_orig = gradient_descent_orig(
        x_candidate=x_candidate,
        f_candidate_initial=f_candidate_initial,
        gradient_projected=gradient_projected,
        hessian_inactive=hessian_inactive,
        model=model,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        active_bounds_info=bounds_info,
        maxiter_steepest_descent=maxiter,
        options_update_radius=options_update_radius,
    )
    aae(res_orig[0], res_fast[0])
    for i in range(1, len(res_orig)):
        assert res_orig[i] == res_fast[i]


def test_update_trustregion_radius_and_gradient_descent():
    options_update_radius = {
        "mu1": 0.35,
        "mu2": 0.50,
        "gamma1": 0.0625,
        "gamma2": 0.5,
        "gamma3": 2.0,
        "gamma4": 5.0,
        "theta": 0.25,
        "min_radius": 1e-10,
        "max_radius": 1e10,
        "default_radius": 100,
    }

    trustregion_radius = 100.00
    radius_lower_bound = 90.00
    predicted_reduction = 0.9
    actual_reduction = 1.1
    gradient_norm = 10.0
    res_orig = _update_trustregion_radius_and_gradient_descent(
        trustregion_radius,
        radius_lower_bound,
        predicted_reduction,
        actual_reduction,
        gradient_norm,
        options_update_radius,
    )
    options_update_radius.pop("min_radius")
    options_update_radius.pop("max_radius")
    options_update_radius.pop("default_radius")
    res_fast = _update_trr_and_gd_fast(
        trustregion_radius,
        radius_lower_bound,
        predicted_reduction,
        actual_reduction,
        gradient_norm,
        **options_update_radius,
    )
    assert res_orig[0] == res_fast[0]
    assert res_fast[1] == res_orig[1]


def test_minimize_bntr():
    model = pd.read_pickle(TEST_FIXTURES_DIR / "scalar_model.pkl")
    lower_bounds = -np.ones(len(model.linear_terms))
    upper_bounds = np.ones(len(model.linear_terms))
    options = {
        "maxiter": 20,
        "maxiter_gradient_descent": 5,
        "conjugate_gradient_method": "cg",
        "gtol_abs": 1e-08,
        "gtol_rel": 1e-08,
        "gtol_scaled": 0.0,
        "gtol_abs_conjugate_gradient": 1e-08,
        "gtol_rel_conjugate_gradient": 1e-06,
    }
    res_orig = bntr(model, lower_bounds, upper_bounds, **options)
    res_fast = _bntr_fast_jitted(
        model.linear_terms, model.square_terms, lower_bounds, upper_bounds, **options
    )
    # using aaae to get tests run on windows machines.
    aaae(res_orig["x"], res_fast[0])
    aaae(res_orig["criterion"], res_fast[1])
    assert res_orig["success"] == res_fast[3]


def test_minimize_bntr_break_loop_early():
    model = pd.read_pickle(TEST_FIXTURES_DIR / "scalar_model.pkl")
    lower_bounds = -np.ones(len(model.linear_terms))
    upper_bounds = np.ones(len(model.linear_terms))
    options = {
        "maxiter": 20,
        "maxiter_gradient_descent": 5,
        "conjugate_gradient_method": "cg",
        "gtol_abs": 10,
        "gtol_rel": 10,
        "gtol_scaled": 10,
        "gtol_abs_conjugate_gradient": 10,
        "gtol_rel_conjugate_gradient": 10,
    }
    res_fast = _bntr_fast_jitted(
        model.linear_terms, model.square_terms, lower_bounds, upper_bounds, **options
    )
    # using aaae to get tests run on windows machines.
    aaae(np.zeros(len(model.linear_terms)), res_fast[0])
    aaae(0, res_fast[1])
    assert res_fast[3]
    assert res_fast[2] == 0
