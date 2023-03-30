import numpy as np
from estimagic.optimization.subsolvers.gqtpar import (
    DampingFactors,
    HessianInfo,
)
from estimagic.optimization.subsolvers.gqtpar import (
    _compute_smallest_step_len_for_candidate_vector as compute_smallest_step_orig,
)
from estimagic.optimization.subsolvers.gqtpar import (
    _find_new_candidate_and_update_parameters as find_new_and_update_candidate_orig,
)
from estimagic.optimization.subsolvers.gqtpar import (
    _get_initial_guess_for_lambdas as init_lambdas_orig,
)
from estimagic.optimization.subsolvers.gqtpar_fast import (
    _compute_smallest_step_len_for_candidate_vector as compute_smallest_step_fast,
)
from estimagic.optimization.subsolvers.gqtpar_fast import (
    _find_new_candidate_and_update_parameters as find_new_and_update_candidate_fast,
)
from estimagic.optimization.subsolvers.gqtpar_fast import (
    _get_initial_guess_for_lambdas as init_lambdas_fast,
)
from estimagic.optimization.tranquilo.models import ScalarModel
from numpy.testing import assert_array_almost_equal as aaae


def test_get_initial_guess_for_lambda():
    rng = np.random.default_rng(12345)
    model_gradient = rng.random(10)
    model_hessian = rng.random((10, 10))
    model_hessian = model_hessian @ model_hessian.T
    model = ScalarModel(
        linear_terms=model_gradient, square_terms=model_hessian, intercept=None
    )
    res = init_lambdas_fast(model_gradient, model_hessian)
    expected = init_lambdas_orig(model)
    assert res[0] == expected.candidate
    assert res[1] == expected.lower_bound
    aaae(res[2], expected.upper_bound)


def test_find_new_candidate_and_update_parameters():
    rng = np.random.default_rng(12345)
    model_gradient = rng.random(10)
    model_hessian = rng.random((10, 10))
    model_hessian = model_hessian @ model_hessian.T
    model = ScalarModel(
        linear_terms=model_gradient, square_terms=model_hessian, intercept=None
    )
    hessian_upper_triangular = np.triu(model_hessian)
    candidate = 0.8
    hessian_plus_lambda = model_hessian + candidate * np.eye(10)
    lower_bound = 0.3
    upper_bound = 1.3
    criteria = {"k_easy": 0.1, "k_hard": 0.2}
    converged = False
    already_factorized = False
    lambdas = DampingFactors(
        candidate=candidate, lower_bound=lower_bound, upper_bound=upper_bound
    )
    hessian_info = HessianInfo(
        hessian_plus_lambda=hessian_plus_lambda,
        upper_triangular=hessian_upper_triangular,
        already_factorized=already_factorized,
    )
    res = find_new_and_update_candidate_fast(
        model_gradient,
        model_hessian,
        hessian_upper_triangular,
        hessian_plus_lambda,
        already_factorized,
        candidate,
        lower_bound,
        upper_bound,
        criteria,
        converged,
    )
    expected = find_new_and_update_candidate_orig(
        model, hessian_info, lambdas, criteria, converged
    )
    aaae(res[0], expected[0])
    aaae(res[1], expected[1].hessian_plus_lambda)
    aaae(res[2], expected[1].already_factorized)
    aaae(res[3], expected[2].candidate)
    aaae(res[4], expected[2].lower_bound)
    aaae(res[5], expected[2].upper_bound)
    assert res[6] == expected[3]


def test_compute_smallest_step_len_for_candidate_vector():
    rng = np.random.default_rng(12345)
    x_candidate = rng.random(10)
    rng = np.random.default_rng(45667)
    z_min = rng.random(10)
    res = compute_smallest_step_fast(x_candidate, z_min)
    expected = compute_smallest_step_orig(x_candidate, z_min)
    aaae(res, expected)
