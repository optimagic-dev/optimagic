from enum import Enum
from typing import Annotated, Any, Callable, Literal, Protocol, TypeVar

from annotated_types import Ge, Gt, Lt

PyTree = Any
PyTreeRegistry = dict[type | str, dict[str, Callable[[Any], Any]]]
Scalar = Any

T = TypeVar("T")


class AggregationLevel(Enum):
    """Enum to specify the aggregation level of objective functions and solvers."""

    SCALAR = "scalar"
    LEAST_SQUARES = "least_squares"
    LIKELIHOOD = "likelihood"


class Direction(Enum):
    """Enum to specify the direction of optimization."""

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class ErrorHandling(Enum):
    """Enum to specify the error handling strategy of the optimization algorithm."""

    RAISE = "raise"
    RAISE_STRICT = "raise_strict"
    CONTINUE = "continue"


class EvalTask(Enum):
    """Enum to specify the task of the evaluation function."""

    FUN = "fun"
    JAC = "jac"
    FUN_AND_JAC = "fun_and_jac"
    EXPLORATION = "exploration"


class BatchEvaluator(Protocol):
    def __call__(
        self,
        func: Callable[..., T],
        arguments: list[Any],
        n_cores: int = 1,
        error_handling: ErrorHandling
        | Literal["raise", "continue"] = ErrorHandling.CONTINUE,
        unpack_symbol: Literal["*", "**"] | None = None,
    ) -> list[T]:
        pass


PositiveInt = Annotated[int, Gt(0)]
NonNegativeInt = Annotated[int, Ge(0)]
PositiveFloat = Annotated[float, Gt(0)]
NonNegativeFloat = Annotated[float, Ge(0)]
NegativeFloat = Annotated[float, Lt(0)]
