"""Scalers that make benchmark problems badly scaled."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


class BenchmarkScaler(ABC):
    """Base class for rescaling the parameters of benchmark problems.

    A scaler changes the parameters an optimizer works with. The optimizer sees
    ``scale(params)`` and the benchmark function is evaluated at ``unscale`` of what the
    optimizer proposes. Scaling is used to make a problem harder without changing its
    solution.

    Subclasses implement ``scale`` and ``unscale``. Both have to work element by element
    and be strictly increasing, such that bounds can be scaled like parameters.

    """

    @abstractmethod
    def scale(self, params: NDArray[np.float64]) -> NDArray[np.float64]:
        """Convert parameters of the original problem to what the optimizer sees.

        Args:
            params: Parameters of the original problem, shape (n_params,).

        Returns:
            The scaled parameters, shape (n_params,).

        """

    @abstractmethod
    def unscale(self, params: NDArray[np.float64]) -> NDArray[np.float64]:
        """Convert parameters the optimizer sees to those of the original problem.

        Args:
            params: Scaled parameters, shape (n_params,).

        Returns:
            Parameters of the original problem, shape (n_params,).

        """


@dataclass(frozen=True)
class LinspaceBenchmarkScaler(BenchmarkScaler):
    """Multiply the parameters by evenly spaced numbers.

    The parameters the optimizer sees are the original parameters multiplied by
    ``np.linspace(min_scale, max_scale, n_params)``. If ``min_scale`` and ``max_scale``
    have very different orders of magnitude, the problem becomes harder to solve for
    many optimizers.

    """

    min_scale: float = 0.1
    """The factor by which the first parameter is multiplied. Must be positive."""

    max_scale: float = 10.0
    """The factor by which the last parameter is multiplied. Must be positive."""

    def __post_init__(self) -> None:
        if self.min_scale <= 0 or self.max_scale <= 0:
            raise ValueError(
                "min_scale and max_scale must be positive, not "
                f"{self.min_scale} and {self.max_scale}."
            )

    def scale(self, params: NDArray[np.float64]) -> NDArray[np.float64]:
        return params * self._get_scaling_factor(len(params))

    def unscale(self, params: NDArray[np.float64]) -> NDArray[np.float64]:
        return params / self._get_scaling_factor(len(params))

    def _get_scaling_factor(self, n_params: int) -> NDArray[np.float64]:
        return np.linspace(self.min_scale, self.max_scale, n_params)


def pre_process_scaling(scaling: bool | BenchmarkScaler) -> BenchmarkScaler | None:
    """Convert all valid ways of specifying scaling to a BenchmarkScaler or None.

    Args:
        scaling: False for no scaling, True for a LinspaceBenchmarkScaler with default
            settings or a BenchmarkScaler.

    Returns:
        The scaler or None if the problems are not rescaled.

    Raises:
        TypeError: If scaling is neither a bool nor a BenchmarkScaler.

    """
    if isinstance(scaling, BenchmarkScaler):
        return scaling
    if scaling is True:
        return LinspaceBenchmarkScaler()
    if scaling is False:
        return None
    raise TypeError(
        "scaling must be a bool or an instance of BenchmarkScaler, not "
        f"{type(scaling)}."
    )
