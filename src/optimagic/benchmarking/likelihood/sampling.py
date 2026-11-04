"""Deterministic random draws for the simulated likelihood benchmark problems.

NumPy only guarantees stable streams for its bit generators, not for the algorithms
that turn random bits into draws from a distribution. To keep the simulated datasets
(and thus the stored reference solutions) stable across NumPy versions, every draw is
computed from uniform draws via the inverse CDF of the target distribution.

"""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import special, stats

_SMALLEST_UNIFORM = 2.0**-53


def uniform(
    rng: np.random.Generator, size: int | tuple[int, ...]
) -> NDArray[np.float64]:
    """Draw from the uniform distribution on the open interval (0, 1)."""
    return np.maximum(rng.random(size), _SMALLEST_UNIFORM)


def standard_normal(
    rng: np.random.Generator, size: int | tuple[int, ...]
) -> NDArray[np.float64]:
    """Draw from the standard normal distribution."""
    return special.ndtri(uniform(rng, size))


def standardized_t(
    rng: np.random.Generator, df: float, size: int | tuple[int, ...]
) -> NDArray[np.float64]:
    """Draw from a t distribution with df > 2 degrees of freedom and unit variance."""
    return special.stdtrit(df, uniform(rng, size)) * np.sqrt((df - 2) / df)


def gamma(
    rng: np.random.Generator,
    shape: ArrayLike,
    scale: ArrayLike,
    size: int | tuple[int, ...],
) -> NDArray[np.float64]:
    """Draw from the gamma distribution with mean shape * scale."""
    return np.asarray(scale) * special.gammaincinv(shape, uniform(rng, size))


def exponential(
    rng: np.random.Generator, size: int | tuple[int, ...]
) -> NDArray[np.float64]:
    """Draw from the exponential distribution with mean one."""
    return -np.log(uniform(rng, size))


def standard_gumbel(
    rng: np.random.Generator, size: int | tuple[int, ...]
) -> NDArray[np.float64]:
    """Draw from the standard Gumbel (type I extreme value) distribution."""
    return -np.log(-np.log(uniform(rng, size)))


def standard_logistic(
    rng: np.random.Generator, size: int | tuple[int, ...]
) -> NDArray[np.float64]:
    """Draw from the standard logistic distribution."""
    return special.logit(uniform(rng, size))


def poisson(rng: np.random.Generator, mean: NDArray[np.float64]) -> NDArray[np.float64]:
    """Draw one Poisson count per entry of mean."""
    return stats.poisson.ppf(uniform(rng, mean.shape), mean)


def binomial(
    rng: np.random.Generator, n_trials: NDArray[np.float64], probability: float
) -> NDArray[np.float64]:
    """Draw one binomial count per entry of n_trials."""
    return stats.binom.ppf(uniform(rng, n_trials.shape), n_trials, probability)


def categorical(
    rng: np.random.Generator, probabilities: NDArray[np.float64]
) -> NDArray[np.int64]:
    """Draw one category per row of probabilities.

    Args:
        rng: Random number generator.
        probabilities: Array of shape (n, n_categories) whose rows sum to one.

    Returns:
        Integer array of shape (n,) with values in {0, ..., n_categories - 1}.

    """
    cumulative = np.cumsum(probabilities, axis=1)
    u = uniform(rng, len(probabilities)) * cumulative[:, -1]
    return (cumulative[:, :-1] < u[:, None]).sum(axis=1)
