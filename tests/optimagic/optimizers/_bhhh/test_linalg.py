import numpy as np
import pytest
from numpy.testing import assert_allclose

from optimagic.optimizers._bhhh.linalg import (
    condition_number,
    factorize_opg,
    rank,
    unconstrained_step,
    weakly_identified_directions,
)
from optimagic.utilities import get_rng


def _random_jac(n_obs=200, n_params=4, seed=0):
    rng = get_rng(seed)
    scales = np.array([1e-3, 1, 10, 1e4])[:n_params]
    return rng.normal(size=(n_obs, n_params)) * scales + 0.1 * scales


def test_factorization_reconstructs_opg():
    jac = _random_jac()
    fact = factorize_opg(jac)
    d, v, lam = fact.scale, fact.eigenvectors, fact.eigenvalues
    reconstructed = np.outer(d, d) * (v @ np.diag(lam) @ v.T)
    assert_allclose(fact.gradient, jac.sum(axis=0))
    assert_allclose(fact.scale, np.linalg.norm(jac, axis=0))
    assert_allclose(reconstructed, jac.T @ jac, rtol=1e-10)
    assert not fact.zero_columns.any()


def test_unconstrained_step_matches_least_squares_solution():
    jac = _random_jac()
    step, decrement = unconstrained_step(factorize_opg(jac))
    expected = -np.linalg.lstsq(jac, np.ones(len(jac)), rcond=None)[0]
    assert_allclose(step, expected, rtol=1e-8)
    assert_allclose(decrement, -jac.sum(axis=0) @ expected, rtol=1e-8)


def test_step_is_invariant_to_linear_reparametrization():
    # mixing columns of very different scale is ill-conditioned, so use unit scales
    jac = _random_jac()
    jac = jac / np.linalg.norm(jac, axis=0)
    transform = get_rng(1).normal(size=(4, 4)) + 3 * np.eye(4)
    step, decrement = unconstrained_step(factorize_opg(jac))
    new_step, new_decrement = unconstrained_step(factorize_opg(jac @ transform))
    assert_allclose(transform @ new_step, step, rtol=1e-7)
    assert_allclose(new_decrement, decrement, rtol=1e-8)


def test_decrement_is_invariant_to_scaling_the_contributions():
    jac = _random_jac()
    step, decrement = unconstrained_step(factorize_opg(jac))
    new_step, new_decrement = unconstrained_step(factorize_opg(1000 * jac))
    assert_allclose(new_decrement, decrement, rtol=1e-10)
    assert_allclose(new_step, step / 1000, rtol=1e-10)


def test_zero_column():
    jac = _random_jac()
    jac[:, 1] = 0
    fact = factorize_opg(jac)
    step, decrement = unconstrained_step(fact)
    assert fact.zero_columns.tolist() == [False, True, False, False]
    assert fact.scale[1] == 1
    assert step[1] == 0
    assert np.isfinite(decrement)
    assert rank(fact) == 3
    assert condition_number(fact) == np.inf
    assert [d.tolist() for d in weakly_identified_directions(fact)] == [[1]]


def test_duplicated_column():
    jac = _random_jac()
    jac[:, 3] = jac[:, 2]
    fact = factorize_opg(jac)
    step, _ = unconstrained_step(fact)
    expected = -np.linalg.lstsq(jac, np.ones(len(jac)), rcond=None)[0]
    assert np.all(np.isfinite(step))
    assert_allclose(jac @ step, jac @ expected, atol=1e-8)
    assert rank(fact) == 3
    directions = weakly_identified_directions(fact)
    assert len(directions) == 1
    assert sorted(directions[0].tolist()) == [2, 3]


def test_fewer_observations_than_parameters():
    jac = _random_jac(n_obs=2)
    fact = factorize_opg(jac)
    step, decrement = unconstrained_step(fact)
    assert np.all(np.isfinite(step))
    assert rank(fact) == 2
    assert len(weakly_identified_directions(fact)) == 2
    # with two observations, the OPG regression fits the vector of ones exactly
    assert_allclose(decrement, 2, rtol=1e-6)


def test_zero_jacobian():
    fact = factorize_opg(np.zeros((5, 3)))
    step, decrement = unconstrained_step(fact)
    assert_allclose(step, 0)
    assert decrement == 0
    assert rank(fact) == 0


@pytest.mark.parametrize("condition", [1e2, 1e4, 1e6])
def test_ill_conditioned_jacobian(condition):
    # the normal equations lose accuracy with the square of the condition number
    rng = get_rng(2)
    u, _ = np.linalg.qr(rng.normal(size=(500, 3)))
    v, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    jac = u @ np.diag([1, 1 / np.sqrt(condition), 1 / condition]) @ v.T
    step, decrement = unconstrained_step(factorize_opg(jac))
    expected = -np.linalg.lstsq(jac, np.ones(len(jac)), rcond=None)[0]
    tol = 1e-13 * condition**2
    assert_allclose(jac @ step, jac @ expected, rtol=tol, atol=tol)
    assert_allclose(decrement, -jac.sum(axis=0) @ expected, rtol=tol)


def test_condition_number_of_scaled_opg_ignores_units():
    jac = _random_jac()
    unit_scaled = jac / np.linalg.norm(jac, axis=0)
    assert_allclose(
        condition_number(factorize_opg(jac)),
        np.linalg.cond(unit_scaled.T @ unit_scaled),
        rtol=1e-8,
    )
