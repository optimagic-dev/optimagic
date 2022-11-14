import numpy as np
from estimagic.optimization.subsolvers._conjugate_gradient_quadratic import (
    _get_distance_to_trustregion_boundary as gdtb,
)
from estimagic.optimization.subsolvers._conjugate_gradient_quadratic import (
    _update_vectors_for_next_iteration as uvnr,
)
from estimagic.optimization.subsolvers._conjugate_gradient_quadratic import (
    minimize_trust_cg,
)
from estimagic.optimization.subsolvers._conjugate_gradient_quadratic_fast import (
    _get_distance_to_trustregion_boundary as gdtb_fast,
)
from estimagic.optimization.subsolvers._conjugate_gradient_quadratic_fast import (
    _update_vectors_for_next_iteration as uvnr_fast,
)
from estimagic.optimization.subsolvers._conjugate_gradient_quadratic_fast import (
    minimize_trust_cg_fast,
)
from estimagic.optimization.subsolvers._steihaug_toint_quadratic import (
    minimize_trust_stcg,
)
from estimagic.optimization.subsolvers._steihaug_toint_quadratic_fast import (
    minimize_trust_stcg_fast,
)
from estimagic.optimization.subsolvers._trsbox_quadratic import (
    _apply_bounds_to_candidate_vector,
)
from estimagic.optimization.subsolvers._trsbox_quadratic import (
    _calc_greatest_criterion_reduction as greatest_reduction_orig,
)
from estimagic.optimization.subsolvers._trsbox_quadratic import (
    _calc_new_reduction as new_reduction_orig,
)
from estimagic.optimization.subsolvers._trsbox_quadratic import (
    _calc_upper_bound_on_tangent as upper_bound_tangent_orig,
)
from estimagic.optimization.subsolvers._trsbox_quadratic import (
    _compute_new_search_direction_and_norm as new_dir_and_norm_orig,
)
from estimagic.optimization.subsolvers._trsbox_quadratic import (
    _take_constrained_step_up_to_boundary as step_constrained_orig,
)
from estimagic.optimization.subsolvers._trsbox_quadratic import (
    _take_unconstrained_step_up_to_boundary as step_unconstrained_orig,
)
from estimagic.optimization.subsolvers._trsbox_quadratic import (
    _update_candidate_vectors_and_reduction as update_candidate_orig,
)
from estimagic.optimization.subsolvers._trsbox_quadratic import (
    _update_candidate_vectors_and_reduction_alt_step as update_candidate_alt_orig,
)
from estimagic.optimization.subsolvers._trsbox_quadratic import (
    _update_tangent as update_tanget_orig,
)
from estimagic.optimization.subsolvers._trsbox_quadratic import minimize_trust_trsbox
from estimagic.optimization.subsolvers._trsbox_quadratic_fast import (
    _calc_greatest_criterion_reduction as greatest_reduction_fast,
)
from estimagic.optimization.subsolvers._trsbox_quadratic_fast import (
    _calc_new_reduction as new_reduction_fast,
)
from estimagic.optimization.subsolvers._trsbox_quadratic_fast import (
    _calc_upper_bound_on_tangent as upper_bound_tangent_fast,
)
from estimagic.optimization.subsolvers._trsbox_quadratic_fast import (
    _compute_new_search_direction_and_norm as new_dir_and_norm_fast,
)
from estimagic.optimization.subsolvers._trsbox_quadratic_fast import (
    _perform_alternative_trustregion_step as perform_step_alt_fast,
)
from estimagic.optimization.subsolvers._trsbox_quadratic_fast import (
    _perform_alternative_trustregion_step as perform_step_alt_orig,
)
from estimagic.optimization.subsolvers._trsbox_quadratic_fast import (
    _take_constrained_step_up_to_boundary as step_constrained_fast,
)
from estimagic.optimization.subsolvers._trsbox_quadratic_fast import (
    _take_unconstrained_step_up_to_boundary as step_unconstrained_fast,
)
from estimagic.optimization.subsolvers._trsbox_quadratic_fast import (
    _update_candidate_vectors_and_reduction as update_candidate_fast,
)
from estimagic.optimization.subsolvers._trsbox_quadratic_fast import (
    _update_candidate_vectors_and_reduction_alt_step as update_candidate_alt_fast,
)
from estimagic.optimization.subsolvers._trsbox_quadratic_fast import (
    _update_tangent as update_tanget_fast,
)
from estimagic.optimization.subsolvers._trsbox_quadratic_fast import (
    minimize_trust_trsbox_fast,
)
from numpy.testing import assert_array_almost_equal as aaae
from numpy.testing import assert_array_equal as aae


def test_minimize_trust_cg():
    grad = np.arange(5).astype(float)
    hessian = np.arange(25).reshape(5, 5).astype(float)
    radius = 2
    gtol_abs = 1e-8
    gtol_rel = 1e-6
    aae(
        minimize_trust_cg(grad, hessian, radius),
        minimize_trust_cg_fast(grad, hessian, radius, gtol_abs, gtol_rel),
    )


def test_get_distance_to_trustregion_boundary():
    x = np.arange(5).astype(float)
    direction = np.arange(5).astype(float)
    radius = 2
    assert gdtb(x, direction, radius) == gdtb_fast(x, direction, radius)


def test_update_vectors():
    x = np.arange(5).astype(float)
    residual = np.ones(5) * 0.5
    direction = np.ones(5)
    hessian = np.arange(25).reshape(5, 5)
    alpha = 0.5
    res_orig = uvnr(x, residual, direction, hessian, alpha)
    res_fast = uvnr_fast(x, residual, direction, hessian, alpha)
    for i in range(len(res_orig)):
        aae(res_orig[i], res_fast[i])


def test_take_unconstrained_step_towards_boundary():
    raw_distance = np.array([0.5])
    gradient_sumsq = 5.0
    gradient_projected_sumsq = 2.5
    g_x = 0.3
    g_hess_g = -0.3
    for i in range(2):
        assert (
            step_unconstrained_orig(
                raw_distance, gradient_sumsq, gradient_projected_sumsq, g_x, g_hess_g
            )[i]
            == step_unconstrained_fast(
                raw_distance, gradient_sumsq, gradient_projected_sumsq, g_x, g_hess_g
            )[i]
        )


def test_take_constrained_step_towards_boundary():
    x_candidate = np.zeros(5)
    gradient_projected = np.ones(5)
    step_len = 2.5
    lower_bounds = np.array([-1.0] * 3 + [0.01] * 2)
    upper_bounds = np.ones(5)
    for i in range(2):
        assert (
            step_constrained_orig(
                x_candidate, gradient_projected, step_len, lower_bounds, upper_bounds
            )[i]
            == step_constrained_fast(
                x_candidate, gradient_projected, step_len, lower_bounds, upper_bounds
            )[i]
        )


def test_update_candidate_vector_and_reduction_alt_step():
    x = np.zeros(5)
    search_direction = 0.5 * np.ones(5)
    x_bounded = np.array([0] * 2 + [1] * 3)
    g = np.ones(5)
    cosine = 0.5
    sine = 0.5
    hessian_s = np.ones(5)
    hes_red = np.ones(5)
    res_orig = update_candidate_alt_orig(
        x, search_direction, x_bounded, g, cosine, sine, hessian_s, hes_red
    )

    res_fast = update_candidate_alt_fast(
        x, search_direction, x_bounded, g, cosine, sine, hessian_s, hes_red
    )
    for i in range(len(res_orig)):
        aae(res_orig[i], res_fast[i])


def test_update_candidate_vector_and_reduction():
    x_candidate = np.zeros(5)
    x_bounded = np.array([0] * 3 + [-0.01] * 2)
    gradient_candidate = np.ones(5)
    gradient_projected = 0.5 * np.ones(5)
    step_len = 0.05
    total_reduction = 0
    curve_min = -0.5
    index_bound_active = 3
    gradient_projected_sumsq = 25
    gradient_sumsq = 25
    g_hess_g = 100
    hess_g = np.arange(5).astype(float)
    res_fast = update_candidate_fast(
        x_candidate,
        x_bounded,
        gradient_candidate,
        gradient_projected,
        step_len,
        total_reduction,
        curve_min,
        index_bound_active,
        gradient_projected_sumsq,
        gradient_sumsq,
        g_hess_g,
        hess_g,
    )
    res_orig = update_candidate_orig(
        x_candidate,
        x_bounded,
        gradient_candidate,
        gradient_projected,
        step_len,
        total_reduction,
        curve_min,
        index_bound_active,
        gradient_projected_sumsq,
        gradient_sumsq,
        g_hess_g,
        hess_g,
    )
    for i in range(len(res_orig)):
        aae(res_orig[i], res_fast[i])


def test_update_candidate_vector_and_reduction_without_active_bounds():
    x_candidate = np.zeros(5)
    x_bounded = np.zeros(5)
    gradient_candidate = np.ones(5)
    gradient_projected = 0.5 * np.ones(5)
    step_len = 0.05
    total_reduction = 0
    curve_min = -0.5
    gradient_projected_sumsq = 25
    gradient_sumsq = 25
    g_hess_g = 100
    hess_g = np.arange(5).astype(float)
    res_fast = update_candidate_fast(
        x_candidate,
        x_bounded,
        gradient_candidate,
        gradient_projected,
        step_len,
        total_reduction,
        curve_min,
        np.array([]),
        gradient_projected_sumsq,
        gradient_sumsq,
        g_hess_g,
        hess_g,
    )
    res_orig = update_candidate_orig(
        x_candidate,
        x_bounded,
        gradient_candidate,
        gradient_projected,
        step_len,
        total_reduction,
        curve_min,
        None,
        gradient_projected_sumsq,
        gradient_sumsq,
        g_hess_g,
        hess_g,
    )
    for i in range(len(res_orig)):
        aae(res_orig[i], res_fast[i])


def test_perform_alternative_tr_step():
    x_candidate = np.zeros(5)
    x_bounded = np.array([0.1] * 2 + [0] * 3)
    gradient_candidate = np.ones(5).astype(float)
    model_hessian = np.arange(25).reshape(5, 5).astype(float)
    lower_bounds = np.array([0.1] * 2 + [-1] * 3)
    upper_bounds = np.ones(5)
    n_fixed_variables = 1
    total_reduction = 1.5
    res_orig = perform_step_alt_orig(
        x_candidate,
        x_bounded,
        gradient_candidate,
        model_hessian,
        lower_bounds,
        upper_bounds,
        n_fixed_variables,
        total_reduction,
    )
    res_fast = perform_step_alt_fast(
        x_candidate,
        x_bounded,
        gradient_candidate,
        model_hessian,
        lower_bounds,
        upper_bounds,
        n_fixed_variables,
        total_reduction,
    )
    aae(res_orig, res_fast)


def test_perform_alternative_tr_step_without_active_bounds():
    x_candidate = np.zeros(5)
    x_bounded = np.zeros(5)
    gradient_candidate = np.ones(5).astype(float)
    model_hessian = np.arange(25).reshape(5, 5).astype(float)
    lower_bounds = -10 * np.ones(5)
    upper_bounds = 10 * np.ones(5)
    n_fixed_variables = 1
    total_reduction = 1.5
    res_orig = perform_step_alt_orig(
        x_candidate,
        x_bounded,
        gradient_candidate,
        model_hessian,
        lower_bounds,
        upper_bounds,
        n_fixed_variables,
        total_reduction,
    )
    res_fast = perform_step_alt_fast(
        x_candidate,
        x_bounded,
        gradient_candidate,
        model_hessian,
        lower_bounds,
        upper_bounds,
        n_fixed_variables,
        total_reduction,
    )
    aae(res_orig, res_fast)


def test_calc_upper_bound_on_tangent():
    x_candidate = np.zeros(5)
    search_direction = 0.5 * np.ones(5)
    x_bounded = np.array([0] * 4 + [0.01])
    lower_bounds = np.array([-1] * 4 + [0.01])
    upper_bounds = np.ones(5)
    n_fixed_variables = 2
    res_orig = upper_bound_tangent_orig(
        x_candidate,
        search_direction,
        x_bounded,
        lower_bounds,
        upper_bounds,
        n_fixed_variables,
    )
    res_fast = upper_bound_tangent_fast(
        x_candidate,
        search_direction,
        x_bounded,
        lower_bounds,
        upper_bounds,
        n_fixed_variables,
    )
    for i in range(len(res_orig)):
        aae(res_orig[i], res_fast[i])


def test_calc_upper_bound_on_tangent_without_active_bounds():
    x_candidate = np.zeros(5)
    search_direction = 0.5 * np.ones(5)
    x_bounded = np.zeros(5)
    lower_bounds = -np.ones(5)
    upper_bounds = np.ones(5)
    n_fixed_variables = 2
    res_orig = upper_bound_tangent_orig(
        x_candidate,
        search_direction,
        x_bounded,
        lower_bounds,
        upper_bounds,
        n_fixed_variables,
    )
    res_fast = upper_bound_tangent_fast(
        x_candidate,
        search_direction,
        x_bounded,
        lower_bounds,
        upper_bounds,
        n_fixed_variables,
    )
    for i in range(len(res_orig)):
        if res_orig[i] is not None:
            aae(res_orig[i], res_fast[i])
        else:
            assert res_fast[i].size == 0


def test_minimize_trs_box_quadratic():
    model_gradient = np.arange(10).astype(float)
    model_hessian = np.arange(100).reshape(10, 10).astype(float)
    trustregion_radius = 10.0
    lower_bounds = -np.ones(10)
    upper_bounds = np.ones(10)
    res_fast = minimize_trust_trsbox_fast(
        model_gradient, model_hessian, trustregion_radius, lower_bounds, upper_bounds
    )
    res_orig = minimize_trust_trsbox(
        model_gradient,
        model_hessian,
        trustregion_radius,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )
    aae(res_fast, res_orig)


def test_minimize_stcg_fast():
    model_gradient = np.arange(10).astype(float)
    model_hessian = np.arange(100).reshape(10, 10).astype(float)
    trustregion_radius = 10.0
    res_orig = minimize_trust_stcg(model_gradient, model_hessian, trustregion_radius)
    res_fast = minimize_trust_stcg_fast(
        model_gradient, model_hessian, trustregion_radius
    )
    aaae(res_orig, res_fast)


def test_minimize_cg():
    model_gradient = np.arange(10).astype(float)
    model_hessian = np.arange(100).reshape(10, 10).astype(float)
    trustregion_radius = 10.0
    gtol_abs = 1e-8
    gtol_rel = 1e-6
    res_orig = minimize_trust_cg(model_gradient, model_hessian, trustregion_radius)
    res_fast = minimize_trust_cg_fast(
        model_gradient, model_hessian, trustregion_radius, gtol_abs, gtol_rel
    )
    aaae(res_orig, res_fast)


def test_apply_bounds_to_candidate_vector():
    x_bounded = np.array([-1, 1, 0, 0, 0])
    x_candidate = np.zeros(5)
    lower_bounds = np.array([-1, -1, 0.01, -1, -1])
    upper_bounds = np.array([1, 1, 1, -0.01, 1])
    res = _apply_bounds_to_candidate_vector(
        x_candidate, x_bounded, lower_bounds, upper_bounds
    )
    expected = np.array([-1, 1, 0.01, -0.01, 0])
    aae(res, expected)


def test_calc_greatest_criterion_reduction():
    res = greatest_reduction_fast(0.8, 1.1, 1.1, 1.1, 1.1, 1.1)
    expected = greatest_reduction_orig(0.8, 1.1, 1.1, 1.1, 1.1, 1.1)
    assert res == expected


def test_calc_new_reduction():
    res = new_reduction_fast(0.8, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1)
    expected = new_reduction_orig(0.8, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1)
    assert res == expected


def test_update_tangent():
    res = update_tanget_fast(0, 0.8, 2, 2, 1, 3)
    expected = update_tanget_orig(0, 0.8, 2, 2, 1, 3)
    assert res == expected


def test_compute_new_search_direction_and_norm():
    x_candidate = np.zeros(5)
    x_bounded = np.zeros(5)
    gradient_candidate = np.ones(5)
    x_reduced = 0.5
    x_grad = 1
    raw_reduction = 0.5
    res = new_dir_and_norm_fast(
        x_candidate, x_bounded, x_reduced, gradient_candidate, x_grad, raw_reduction
    )
    expected = new_dir_and_norm_orig(
        x_candidate, x_bounded, x_reduced, gradient_candidate, x_grad, raw_reduction
    )
    aaae(expected[0], res[0])
    aaae(expected[1], res[1])
