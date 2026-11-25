import numpy as np
import pytest
from scipy import special

from optimagic.benchmarking.likelihood import sampling

N_DRAWS = 200_000


def _rng():
    return np.random.default_rng(1234)


def test_draws_are_stable_across_versions():
    # The stored reference solutions are only valid if these values never change.
    got = sampling.uniform(np.random.default_rng(0), 3)
    expected = np.array([0.6369616873214543, 0.2697867137638703, 0.04097352393619469])
    np.testing.assert_allclose(got, expected, rtol=1e-15)

    got = sampling.standard_normal(np.random.default_rng(0), 3)
    expected = special.ndtri(expected)
    np.testing.assert_allclose(got, expected, rtol=1e-15)


def test_same_seed_gives_same_draws():
    first = sampling.standardized_t(np.random.default_rng(5), 4.0, 10)
    second = sampling.standardized_t(np.random.default_rng(5), 4.0, 10)
    np.testing.assert_array_equal(first, second)


def test_uniform_is_in_open_unit_interval():
    draws = sampling.uniform(_rng(), N_DRAWS)
    assert draws.min() > 0
    assert draws.max() < 1
    assert draws.shape == (N_DRAWS,)


@pytest.mark.parametrize(
    "draw, mean, var",
    [
        (lambda rng: sampling.standard_normal(rng, N_DRAWS), 0, 1),
        (lambda rng: sampling.standardized_t(rng, 6.0, N_DRAWS), 0, 1),
        (lambda rng: sampling.gamma(rng, 2.0, 1.5, N_DRAWS), 3.0, 4.5),
        (lambda rng: sampling.exponential(rng, N_DRAWS), 1, 1),
        (lambda rng: sampling.standard_gumbel(rng, N_DRAWS), np.euler_gamma, 1.645),
        (lambda rng: sampling.standard_logistic(rng, N_DRAWS), 0, np.pi**2 / 3),
        (lambda rng: sampling.poisson(rng, np.full(N_DRAWS, 2.5)), 2.5, 2.5),
        (lambda rng: sampling.binomial(rng, np.full(N_DRAWS, 10.0), 0.3), 3, 2.1),
    ],
)
def test_draws_have_expected_moments(draw, mean, var):
    draws = draw(_rng())
    assert np.abs(draws.mean() - mean) < 0.02
    assert np.abs(draws.var() / var - 1) < 0.03


def test_gamma_with_array_shape_and_scale():
    shape = np.array([0.5, 5.0])
    scale = np.array([2.0, 0.1])
    draws = sampling.gamma(_rng(), shape, scale, (N_DRAWS, 2))
    np.testing.assert_allclose(draws.mean(axis=0), shape * scale, rtol=0.02)


def test_categorical():
    probabilities = np.tile([0.2, 0.0, 0.5, 0.3], (N_DRAWS, 1))
    draws = sampling.categorical(_rng(), probabilities)
    shares = np.bincount(draws, minlength=4) / N_DRAWS
    np.testing.assert_allclose(shares, [0.2, 0.0, 0.5, 0.3], atol=0.005)
    assert draws.dtype.kind == "i"
