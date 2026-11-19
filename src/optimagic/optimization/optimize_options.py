"""Options of an optimization that do not depend on the optimization problem."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from optimagic.differentiation.numdiff_options import NumdiffOptions
from optimagic.optimization.algorithm import Algorithm
from optimagic.optimization.multistart_options import MultistartOptions
from optimagic.parameters.scaling import ScalingOptions


@dataclass(frozen=True)
class OptimizeOptions:
    """Options for ``minimize`` or ``maximize`` that do not depend on the problem.

    The attributes have the same meaning as the arguments of ``minimize`` with the same
    name. ``OptimizeOptions`` are used where a function runs optimizations for you, e.g.
    in ``run_benchmark``.

    """

    algorithm: str | Algorithm | type[Algorithm]
    """The optimization algorithm."""

    algo_options: dict[str, Any] | None = None
    """Options of the algorithm. Only needed if ``algorithm`` is not a configured
    instance of ``Algorithm``."""

    numdiff_options: NumdiffOptions | None = None
    """Options for the calculation of numerical derivatives."""

    scaling: bool | ScalingOptions = False
    """Whether and how the parameters are rescaled by optimagic."""

    multistart: bool | MultistartOptions = False
    """Whether and how a multistart optimization is done."""

    def __post_init__(self) -> None:
        is_algorithm_class = isinstance(self.algorithm, type) and issubclass(
            self.algorithm, Algorithm
        )
        if not (isinstance(self.algorithm, str | Algorithm) or is_algorithm_class):
            raise TypeError(
                "algorithm must be the name of an algorithm, a subclass of Algorithm "
                f"or an instance of Algorithm, not {type(self.algorithm)}."
            )
        if not isinstance(self.algo_options, dict | None):
            raise TypeError(
                f"algo_options must be a dict or None, not {type(self.algo_options)}."
            )
