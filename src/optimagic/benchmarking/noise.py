"""Noise that can be added to the function values of benchmark problems."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, TypeVar, cast, get_args

import numpy as np
from numpy.typing import NDArray

from optimagic.benchmarking.noise_distributions import NOISE_DISTRIBUTIONS

NoiseDistribution = Literal["normal", "gumbel", "uniform", "logistic", "laplace"]

FvalT = TypeVar("FvalT", float, NDArray[np.float64])
"""Type of a function value: a float or an array of residuals or contributions."""


class BenchmarkNoise(ABC):
    """Base class for noise that is added to the function values of benchmark problems.

    Subclasses implement ``draw_noise``. Noise objects can be combined with ``+``, e.g.
    ``AdditiveBenchmarkNoise() + MultiplicativeBenchmarkNoise()``.

    """

    @abstractmethod
    def draw_noise(
        self,
        fval: FvalT,
        params: NDArray[np.float64],
        rng: np.random.Generator,
    ) -> FvalT:
        """Draw the noise that is added to one function evaluation.

        Args:
            fval: The function value without noise. A float for scalar functions; an
                array of shape (n_residuals,) for least-squares functions and of shape
                (n_contributions,) for likelihood functions.
            params: The parameters at which the function was evaluated, shape
                (n_params,). Can be used for noise that depends on the parameters.
            rng: The random number generator that is used to draw the noise.

        Returns:
            The noise. A float if fval is a float, otherwise an array with the same
            shape as fval.

        """

    def __add__(self, other: BenchmarkNoise) -> CombinedBenchmarkNoise:
        if not isinstance(other, BenchmarkNoise):
            return NotImplemented
        return CombinedBenchmarkNoise((*_get_components(self), *_get_components(other)))


@dataclass(frozen=True)
class AdditiveBenchmarkNoise(BenchmarkNoise):
    """Noise whose standard deviation does not depend on the function value."""

    distribution: NoiseDistribution = "normal"
    """The distribution the noise is drawn from."""

    std: float = 0.01
    """The standard deviation of the noise. This works for all distributions, even if
    they are normally not specified via a standard deviation (e.g. uniform)."""

    correlation: float = 0.0
    """Number between 0 and 1. The correlation between the noise terms of the residuals
    or likelihood contributions of one function evaluation. Has no effect on scalar
    functions."""

    def __post_init__(self) -> None:
        _fail_if_invalid_noise_arguments(
            distribution=self.distribution, std=self.std, correlation=self.correlation
        )

    def draw_noise(
        self,
        fval: FvalT,
        params: NDArray[np.float64],  # noqa: ARG002
        rng: np.random.Generator,
    ) -> FvalT:
        return _draw_noise_like(
            fval,
            distribution=self.distribution,
            std=self.std,
            correlation=self.correlation,
            rng=rng,
        )


@dataclass(frozen=True)
class MultiplicativeBenchmarkNoise(BenchmarkNoise):
    """Noise whose standard deviation is proportional to the function value.

    Multiplicative noise would become zero if the function value is zero. To avoid
    this, the noise is not implemented as ``f_noisy = f * epsilon`` but as ``f_noisy = f
    + (epsilon - 1) * f_clipped``, where ``f_clipped`` is bounded away from zero from
    both sides by ``clipping_value``.

    """

    distribution: NoiseDistribution = "normal"
    """The distribution the noise is drawn from."""

    std: float = 0.01
    """The standard deviation of the noise, relative to the absolute function value.
    This works for all distributions, even if they are normally not specified via a
    standard deviation (e.g. uniform)."""

    correlation: float = 0.0
    """Number between 0 and 1. The correlation between the noise terms of the residuals
    or likelihood contributions of one function evaluation. Has no effect on scalar
    functions."""

    clipping_value: float = 1.0
    """Non-negative number. Function values that are closer to zero than this are
    replaced by this number (with the sign of the function value) when the standard
    deviation of the noise is calculated."""

    def __post_init__(self) -> None:
        _fail_if_invalid_noise_arguments(
            distribution=self.distribution, std=self.std, correlation=self.correlation
        )
        if self.clipping_value < 0:
            raise ValueError(
                f"clipping_value must be non-negative, not {self.clipping_value}."
            )

    def draw_noise(
        self,
        fval: FvalT,
        params: NDArray[np.float64],  # noqa: ARG002
        rng: np.random.Generator,
    ) -> FvalT:
        scaled_std = self.std * _clip_away_from_zero(fval, self.clipping_value)
        return _draw_noise_like(
            fval,
            distribution=self.distribution,
            std=scaled_std,
            correlation=self.correlation,
            rng=rng,
        )


@dataclass(frozen=True)
class CombinedBenchmarkNoise(BenchmarkNoise):
    """The sum of several sources of noise.

    Instances are created by adding noise objects with ``+``.

    """

    components: tuple[BenchmarkNoise, ...]
    """The sources of noise. All of them are drawn for the function value without
    noise and then added up."""

    def __post_init__(self) -> None:
        if len(self.components) == 0:
            raise ValueError("components must contain at least one source of noise.")

    def draw_noise(
        self,
        fval: FvalT,
        params: NDArray[np.float64],
        rng: np.random.Generator,
    ) -> FvalT:
        draws = [
            component.draw_noise(fval, params=params, rng=rng)
            for component in self.components
        ]
        out = draws[0]
        for draw in draws[1:]:
            out = out + draw
        return out


def pre_process_noise(noise: bool | BenchmarkNoise) -> BenchmarkNoise | None:
    """Convert all valid ways of specifying noise to a BenchmarkNoise or None.

    Args:
        noise: False for no noise, True for additive normally distributed noise with
            default settings or a BenchmarkNoise.

    Returns:
        The noise or None if no noise is added.

    Raises:
        TypeError: If noise is neither a bool nor a BenchmarkNoise.

    """
    if isinstance(noise, BenchmarkNoise):
        return noise
    if noise is True:
        return AdditiveBenchmarkNoise()
    if noise is False:
        return None
    raise TypeError(
        f"noise must be a bool or an instance of BenchmarkNoise, not {type(noise)}."
    )


def sample_from_distribution(
    distribution: NoiseDistribution,
    std: float | NDArray[np.float64],
    size: int | tuple[int, int],
    rng: np.random.Generator,
    correlation: float = 0.0,
) -> NDArray[np.float64]:
    """Draw noise with mean zero from a distribution.

    Args:
        distribution: The distribution to draw from.
        std: The standard deviation of the noise. A float or an array of shape
            (n_columns,).
        size: The shape of the sample, either n_columns or (n_draws, n_columns).
        rng: The random number generator.
        correlation: The correlation between the columns of the sample.

    Returns:
        The sample of shape (n_columns,) or (n_draws, n_columns).

    """
    sample = NOISE_DISTRIBUTIONS[distribution](size=size, rng=rng)
    dim = size if isinstance(size, int) else size[1]
    if correlation != 0 and dim > 1:
        chol = np.linalg.cholesky(np.diag(np.ones(dim) - correlation) + correlation)
        sample = (chol @ sample.T).T
        sample = sample / sample.std()
    return sample * std


def _draw_noise_like(
    fval: FvalT,
    distribution: NoiseDistribution,
    std: float | NDArray[np.float64],
    correlation: float,
    rng: np.random.Generator,
) -> FvalT:
    """Draw noise that has the same type and shape as fval."""
    if isinstance(fval, np.ndarray):
        sample = sample_from_distribution(
            distribution, std=std, size=len(fval), rng=rng, correlation=correlation
        )
        return cast(FvalT, sample)
    sample = sample_from_distribution(distribution, std=std, size=1, rng=rng)
    return cast(FvalT, float(sample[0]))


def _get_components(noise: BenchmarkNoise) -> tuple[BenchmarkNoise, ...]:
    if isinstance(noise, CombinedBenchmarkNoise):
        return noise.components
    return (noise,)


def _fail_if_invalid_noise_arguments(
    distribution: str, std: float, correlation: float
) -> None:
    valid_distributions = get_args(NoiseDistribution)
    if distribution not in valid_distributions:
        raise ValueError(
            f"Invalid distribution: {distribution}. Allowed are {valid_distributions}."
        )
    if std < 0:
        raise ValueError(f"std must be non-negative, not {std}.")
    if not 0 <= correlation <= 1:
        raise ValueError(f"correlation must be between 0 and 1, not {correlation}.")


def _clip_away_from_zero(
    a: float | NDArray[np.float64], clipval: float
) -> float | NDArray[np.float64]:
    is_scalar = np.isscalar(a)
    arr = np.atleast_1d(a)

    is_positive = arr >= 0

    clipped = np.where(is_positive, np.clip(arr, clipval, np.inf), arr)
    clipped = np.where(~is_positive, np.clip(clipped, -np.inf, -clipval), clipped)

    if is_scalar:
        return float(clipped[0])
    return clipped
