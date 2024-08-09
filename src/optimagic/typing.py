from enum import Enum
from typing import Any, Callable

PyTree = Any
PyTreeRegistry = dict[type | str, dict[str, Callable[[Any], Any]]]
Scalar = Any


class AggregationLevel(Enum):
    """Enum to specify the aggregation level of objective functions and solvers."""

    SCALAR = "scalar"
    LEAST_SQUARES = "least_squares"
    LIKELIHOOD = "likelihood"
