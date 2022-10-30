import numpy as np
import scipy.optimize._trustregion_exact as scipy_trustregion
from estimagic.optimization.subsolvers._numba_potrf import (
    compute_cholesky_factorization,
)
from estimagic.optimization.subsolvers.gqtpar_quadratic import (
    _compute_gershgorin_bounds,
)
from estimagic.optimization.subsolvers.gqtpar_quadratic import (
    _compute_newton_step as cns_orig,
)
from estimagic.optimization.subsolvers.gqtpar_quadratic import (
    _compute_smallest_step_len_for_candidate_vector as _smallest_step_len_orig,
)
from estimagic.optimization.subsolvers.gqtpar_quadratic import _get_new_lambda_candidate
from estimagic.optimization.subsolvers.gqtpar_quadratic import (
    _solve_scalar_quadratic_equation as _solve_quadratic_orig,
)
from estimagic.optimization.subsolvers.gqtpar_quadratic import (
    _update_candidate_and_parameters_when_candidate_within_trustregion,
)
from estimagic.optimization.subsolvers.gqtpar_quadratic import (
    add_lambda_and_factorize_hessian,
)
from estimagic.optimization.subsolvers.gqtpar_quadratic import (
    check_for_interior_convergence_and_update,
)
from estimagic.optimization.subsolvers.gqtpar_quadratic import DampingFactors
from estimagic.optimization.subsolvers.gqtpar_quadratic import (
    find_new_candidate_and_update_parameters,
)
from estimagic.optimization.subsolvers.gqtpar_quadratic import (
    get_initial_guess_for_lambdas,
)
from estimagic.optimization.subsolvers.gqtpar_quadratic import HessianInfo
from estimagic.optimization.subsolvers.gqtpar_quadratic_fast import (
    _compute_gershgorin_bounds as _compute_gershgorin_bounds_fast,
)
from estimagic.optimization.subsolvers.gqtpar_quadratic_fast import (
    _compute_newton_step as cns_fast,
)
from estimagic.optimization.subsolvers.gqtpar_quadratic_fast import (
    _compute_smallest_step_len_for_candidate_vector as _smallest_step_len_fast,
)
from estimagic.optimization.subsolvers.gqtpar_quadratic_fast import (
    _get_new_lambda_candidate as _get_new_lambda_candidate_fast,
)
from estimagic.optimization.subsolvers.gqtpar_quadratic_fast import (
    _solve_scalar_quadratic_equation as _solve_quadratic_fast,
)
from estimagic.optimization.subsolvers.gqtpar_quadratic_fast import (
    _update_candidate_and_parameters_when_candidate_within_trustregion_fast,
)
from estimagic.optimization.subsolvers.gqtpar_quadratic_fast import (
    add_lambda_and_factorize_hessian_fast,
)
from estimagic.optimization.subsolvers.gqtpar_quadratic_fast import (
    check_for_interior_convergence_and_update_fast,
)
from estimagic.optimization.subsolvers.gqtpar_quadratic_fast import (
    estimate_smallest_singular_value,
)
from estimagic.optimization.subsolvers.gqtpar_quadratic_fast import (
    find_new_candidate_and_update_parameters_fast,
)
from estimagic.optimization.subsolvers.gqtpar_quadratic_fast import (
    get_initial_guess_for_lambdas_fast,
)
from estimagic.optimization.tranquilo.models import ScalarModel
from numpy.testing import assert_array_almost_equal as aaae
from scipy.linalg.lapack import dpotrf


def test_get_initial_guess_for_lambdas():
    model_gradient = np.arange(10).astype(float)
    model_hessian = np.arange(100).reshape(10, 10).astype(float)
    model = ScalarModel(linear_terms=model_gradient, square_terms=model_hessian)
    lambdas = get_initial_guess_for_lambdas(model)
    (
        lambda_candidate,
        lambda_lower_bound,
        lambda_upper_bound,
    ) = get_initial_guess_for_lambdas_fast(model_gradient, model_hessian)
    assert lambda_candidate == lambdas.candidate
    assert lambda_lower_bound == lambdas.lower_bound
    assert lambda_upper_bound == lambdas.upper_bound


def test_compute_gershgorin_bounds():
    model_hessian = np.arange(100).reshape(10, 10).astype(float)
    model = ScalarModel(square_terms=model_hessian)
    res_orig = _compute_gershgorin_bounds(model)
    res_fast = _compute_gershgorin_bounds_fast(model_hessian)
    assert res_orig[0] == res_fast[0]
    assert res_orig[1] == res_fast[1]


def test_get_new_lambda_candidate():
    assert _get_new_lambda_candidate(0.001, 0.999) == _get_new_lambda_candidate_fast(
        0.001, 0.999
    )


def test_add_lambda_and_factorize_hessian():
    np.random.seed(42)
    n = 10
    x = np.random.random((n, n))
    hessian = np.dot(x, x.T)
    lambda_candidate = 0.8
    hessian_info = HessianInfo()
    model = ScalarModel(square_terms=hessian)
    lambdas = DampingFactors(candidate=lambda_candidate)
    res = add_lambda_and_factorize_hessian_fast(hessian, lambda_candidate)
    expected = add_lambda_and_factorize_hessian(model, hessian_info, lambdas)
    assert res[-1] == expected[-1]
    aaae(res[0], expected[0].hessian_plus_lambda)
    aaae(res[1], expected[0].upper_triangular)


def test_cholesky_factorization():
    rng = np.random.default_rng(12345)
    n = 10
    x = rng.random((n, n))
    a = np.dot(x, x.T)
    arrays = [a.copy()]
    for i in range(1, n + 1):
        a[-i, -i] = 0
        arrays.append(a)
    for array in arrays:
        res = compute_cholesky_factorization(array)
        expected = dpotrf(array)
        assert res[1] == expected[1]
        aaae(res[0], expected[0])


def test_estimate_smallest_singular_value():
    a = np.array(
        [
            [1.21364, 1.18991, 1.29241, 0.87672],
            [0.0, 0.40929, -0.15593, -0.23441],
            [0.0, 0.0, 0.53628, -0.30766],
            [0.0, 0.0, 0.0, 0.52325],
        ]
    )

    res = estimate_smallest_singular_value(a)
    expected = scipy_trustregion.estimate_smallest_singular_value(a)
    assert res[0] == expected[0]
    aaae(res[1], expected[1])


def test_compute_newton_step():
    lambda_candidate = 0.8
    p_norm = 2.5
    w_norm = 5
    lambdas = DampingFactors(candidate=lambda_candidate)
    assert cns_fast(lambda_candidate, p_norm, w_norm) == cns_orig(
        lambdas, p_norm, w_norm
    )


def test_solve_quadratic_equation():
    z = np.array([0.12, 0.56, 0.44, 0.65])
    d = np.array([1.45, 3.56, 9.20, 10.10])
    res = _solve_quadratic_fast(z, d)
    expected = _solve_quadratic_orig(z, d)
    aaae(expected[0], res[0])
    aaae(expected[1], res[1])


def test_compute_smallest_step_len_for_candidate_vector():
    x_candidate = np.array([0.12, 0.56, 0.44, 0.65])
    z_min = np.array([1.45, 3.56, 9.20, 10.10])
    aaae(
        _smallest_step_len_orig(x_candidate, z_min),
        _smallest_step_len_fast(x_candidate, z_min),
    )


def test_update_within_trustregion():
    x_candidate = np.zeros(4)
    model_hessian = np.array(
        [
            [1.473, 1.444, 1.569, 1.064],
            [1.444, 1.583, 1.474, 0.947],
            [1.569, 1.474, 1.982, 1.005],
            [1.064, 0.947, 1.005, 1.192],
        ]
    )
    hessian_upper_triangular = np.array(
        [
            [1.214, 1.19, 1.292, 0.877],
            [0.0, 0.409, -0.156, -0.234],
            [0.0, 0.0, 0.536, -0.308],
            [0.0, 0.0, 0.0, 0.523],
        ]
    )
    lambda_candidate = 0.87
    hessian_plus_lambda = model_hessian + lambda_candidate * np.eye(4)
    hessian_already_factorized = False
    lambda_lower_bound = 0.1
    newton_step = 0.55
    stopping_criterion = 0.2
    converged = False
    lambdas = DampingFactors(candidate=lambda_candidate, lower_bound=lambda_lower_bound)
    hessian_info = HessianInfo(
        hessian_plus_lambda=hessian_plus_lambda,
        upper_triangular=hessian_upper_triangular,
        already_factorized=hessian_already_factorized,
    )
    model = ScalarModel(square_terms=model_hessian)

    res = _update_candidate_and_parameters_when_candidate_within_trustregion_fast(
        x_candidate,
        model_hessian,
        hessian_upper_triangular,
        hessian_plus_lambda,
        hessian_already_factorized,
        lambda_candidate,
        lambda_lower_bound,
        newton_step,
        stopping_criterion,
        converged,
    )

    expected = _update_candidate_and_parameters_when_candidate_within_trustregion(
        x_candidate,
        model,
        hessian_info,
        lambdas,
        newton_step,
        {"k_hard": stopping_criterion},
        converged,
    )
    aaae(res[0], expected[0])
    aaae(res[1], expected[1].hessian_plus_lambda)
    aaae(res[2], expected[1].already_factorized)
    aaae(res[3], expected[2].candidate)
    aaae(res[4], expected[2].lower_bound)
    aaae(res[5], expected[2].upper_bound)
    aaae(res[6], expected[3])


def test_find_new_candidate_and_update_parameters():
    model_hessian = np.array(
        [
            [1.473, 1.444, 1.569, 1.064],
            [1.444, 1.583, 1.474, 0.947],
            [1.569, 1.474, 1.982, 1.005],
            [1.064, 0.947, 1.005, 1.192],
        ]
    )
    model_gradient = np.ones(4)
    hessian_upper_triangular = np.array(
        [
            [1.214, 1.19, 1.292, 0.877],
            [0.0, 0.409, -0.156, -0.234],
            [0.0, 0.0, 0.536, -0.308],
            [0.0, 0.0, 0.0, 0.523],
        ]
    )
    lambda_candidate = 0.87
    hessian_plus_lambda = model_hessian + lambda_candidate * np.eye(4)
    lambda_lower_bound = 0.1
    k_easy = 0.1
    k_hard = 0.2
    converged = False
    hessian_already_factorized = False
    hessian_info = HessianInfo(
        already_factorized=hessian_already_factorized,
        upper_triangular=hessian_upper_triangular,
        hessian_plus_lambda=hessian_plus_lambda,
    )
    model = ScalarModel(square_terms=model_hessian, linear_terms=model_gradient)
    lambdas = DampingFactors(
        candidate=lambda_candidate,
        lower_bound=lambda_lower_bound,
    )
    stopping_criteria = {"k_hard": k_hard, "k_easy": k_easy}
    res = find_new_candidate_and_update_parameters_fast(
        model_gradient,
        model_hessian,
        hessian_upper_triangular,
        hessian_plus_lambda,
        hessian_already_factorized,
        lambda_candidate,
        lambda_lower_bound,
        np.inf,
        k_easy,
        k_hard,
        converged,
    )
    expected = find_new_candidate_and_update_parameters(
        model, hessian_info, lambdas, stopping_criteria, converged
    )
    aaae(res[0], expected[0])
    aaae(res[1], expected[1].hessian_plus_lambda)
    aaae(res[2], expected[1].upper_triangular)
    assert res[3] == expected[1].already_factorized
    aaae(res[4], expected[2].candidate)
    aaae(res[5], expected[2].lower_bound)
    if expected[2].upper_bound is not None:
        assert res[6] == expected[2].upper_bound
    else:
        assert res[6] == np.inf
    assert res[7] == expected[3]


def test_find_new_candidate_and_update_parameters_with_smaller_gardient():
    model_hessian = np.array(
        [
            [1.473, 1.444, 1.569, 1.064],
            [1.444, 1.583, 1.474, 0.947],
            [1.569, 1.474, 1.982, 1.005],
            [1.064, 0.947, 1.005, 1.192],
        ]
    )
    model_gradient = np.ones(4) / 100
    hessian_upper_triangular = np.array(
        [
            [1.214, 1.19, 1.292, 0.877],
            [0.0, 0.409, -0.156, -0.234],
            [0.0, 0.0, 0.536, -0.308],
            [0.0, 0.0, 0.0, 0.523],
        ]
    )
    lambda_candidate = 0.87
    hessian_plus_lambda = model_hessian + lambda_candidate * np.eye(4)
    lambda_lower_bound = 0.1
    k_easy = 0.1
    k_hard = 0.2
    converged = False
    hessian_already_factorized = False
    hessian_info = HessianInfo(
        already_factorized=hessian_already_factorized,
        upper_triangular=hessian_upper_triangular,
        hessian_plus_lambda=hessian_plus_lambda,
    )
    model = ScalarModel(square_terms=model_hessian, linear_terms=model_gradient)
    lambdas = DampingFactors(
        candidate=lambda_candidate,
        lower_bound=lambda_lower_bound,
    )
    stopping_criteria = {"k_hard": k_hard, "k_easy": k_easy}
    res = find_new_candidate_and_update_parameters_fast(
        model_gradient,
        model_hessian,
        hessian_upper_triangular,
        hessian_plus_lambda,
        hessian_already_factorized,
        lambda_candidate,
        lambda_lower_bound,
        np.inf,
        k_easy,
        k_hard,
        converged,
    )
    expected = find_new_candidate_and_update_parameters(
        model, hessian_info, lambdas, stopping_criteria, converged
    )
    aaae(res[0], expected[0])
    aaae(res[1], expected[1].hessian_plus_lambda)
    aaae(res[2], expected[1].upper_triangular)
    assert res[3] == expected[1].already_factorized
    aaae(res[4], expected[2].candidate)
    aaae(res[5], expected[2].lower_bound)
    if expected[2].upper_bound is not None:
        assert res[6] == expected[2].upper_bound
    else:
        assert res[6] == np.inf
    assert res[7] == expected[3]


def test_check_for_interior_convergence():
    x_candidate = np.zeros(4)
    hessian_upper_triangular = np.array(
        [
            [1.214, 1.19, 1.292, 0.877],
            [0.0, 0.409, -0.156, -0.234],
            [0.0, 0.0, 0.536, -0.308],
            [0.0, 0.0, 0.0, 0.523],
        ]
    )
    lambda_candidate = 0.2
    lambda_lower_bound = 0.0
    lambda_upper_bound = 1.5
    stopping_criterion = 0.1
    converged = False
    lambdas = DampingFactors(
        candidate=lambda_candidate,
        lower_bound=lambda_lower_bound,
        upper_bound=lambda_upper_bound,
    )
    hessian_info = HessianInfo(upper_triangular=hessian_upper_triangular)
    stopping_criteria = {"k_hard": 0.2}
    res = check_for_interior_convergence_and_update_fast(
        x_candidate,
        hessian_upper_triangular,
        lambda_candidate,
        lambda_lower_bound,
        lambda_upper_bound,
        stopping_criterion,
        converged,
    )
    expected = check_for_interior_convergence_and_update(
        x_candidate, hessian_info, lambdas, stopping_criteria, converged
    )
    aaae(res[0], expected[0])
    assert res[1] == expected[1].candidate
    assert res[2] == expected[1].lower_bound
    assert res[3] == expected[1].upper_bound
    assert res[4] == expected[2]
