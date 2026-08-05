import functools
import typing
from dataclasses import dataclass, fields
from enum import Enum
from typing import (
    Annotated,
    Any,
    Callable,
    ItemsView,
    Iterator,
    KeysView,
    Literal,
    Protocol,
    TypeVar,
    ValuesView,
    runtime_checkable,
)

import numpy as np
import pydantic
from annotated_types import Ge, Gt, Le, Lt
from numpy._typing import NDArray

PyTree = Any
PyTreeRegistry = dict[type | str, dict[str, Callable[[Any], Any]]]
Scalar = Any

T = TypeVar("T")


class AggregationLevel(Enum):
    """Enum to specify the aggregation level of objective functions and solvers."""

    SCALAR = "scalar"
    LEAST_SQUARES = "least_squares"
    LIKELIHOOD = "likelihood"


class Direction(str, Enum):
    """Enum to specify the direction of optimization."""

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


@dataclass(frozen=True)
class DictLikeAccess:
    r"""Useful base class for replacing string-based dictionaries with dataclass
    instances and keeping backward compatability regarding read access to the data
    structure.
    """

    def __getitem__(self, key: str) -> Any:
        if key in self.__dict__:
            return getattr(self, key)
        else:
            raise KeyError(f"{key} not found in {self.__class__.__name__}")

    def __iter__(self) -> Iterator[str]:
        return iter(self._dict_repr())

    def _dict_repr(self) -> dict[str, Any]:
        return {field.name: getattr(self, field.name) for field in fields(self)}

    def keys(self) -> KeysView[str]:
        return self._dict_repr().keys()

    def items(self) -> ItemsView[str, Any]:
        return self._dict_repr().items()

    def values(self) -> ValuesView[str]:
        return self._dict_repr().values()


@dataclass(frozen=True)
class TupleLikeAccess:
    r"""Useful base class for replacing tuples with dataclass instances and keeping
    backward compatability regarding read access to the data structure.
    """

    def __getitem__(self, index: int | slice) -> Any:
        field_values = [getattr(self, field.name) for field in fields(self)]
        return field_values[index]

    def __len__(self) -> int:
        return len(fields(self))

    def __iter__(self) -> Iterator[str]:
        for field in fields(self):
            yield getattr(self, field.name)


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


@runtime_checkable
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
"""Type alias for positive integers (greater than 0)."""
NonNegativeInt = Annotated[int, Ge(0)]
"""Type alias for non-negative integers (greater than or equal to 0)."""
PositiveFloat = Annotated[float, Gt(0)]
"""Type alias for positive floats (greater than 0)."""
NonNegativeFloat = Annotated[float, Ge(0)]
"""Type alias for non-negative floats (greater than or equal to 0)."""
ProbabilityFloat = Annotated[float, Ge(0), Le(1)]
"""Type alias for probability floats (between 0 and 1, inclusive)."""
NegativeFloat = Annotated[float, Lt(0)]
"""Type alias for negative floats (less than 0)."""
GtOneFloat = Annotated[float, Gt(1)]
"""Type alias for floats greater than 1."""
UnitIntervalFloat = Annotated[float, Gt(0), Le(1)]
"""Type alias for floats in (0, 1]."""
YesNoBool = Literal["yes", "no"] | bool
"""Type alias for boolean values represented as 'yes' or 'no' strings or as boolean
values."""
DirectionLiteral = Literal["minimize", "maximize"]
"""Type alias for optimization direction, either 'minimize' or 'maximize'."""
BatchEvaluatorLiteral = Literal["joblib", "pathos", "threading"]
"""Type alias for batch evaluator types, can be 'joblib', 'pathos', or 'threading'."""
ErrorHandlingLiteral = Literal["raise", "continue"]
"""Type alias for error handling strategies, can be 'raise' or 'continue'."""


DataclassT = TypeVar("DataclassT")

DEFAULT_PYDANTIC_CONFIG = pydantic.ConfigDict(
    arbitrary_types_allowed=True,
    extra="forbid",
    validate_default=True,
)
"""Pydantic config for user-facing options: coerce generous inputs to strict types."""

STRICT_PYDANTIC_CONFIG = pydantic.ConfigDict(
    strict=True,
    arbitrary_types_allowed=True,
    extra="forbid",
    validate_default=True,
)
"""Pydantic config for internal types: reject inputs that need conversion."""


def validated_dataclass(
    config: pydantic.ConfigDict,
    make_error: Callable[[pydantic.ValidationError], Exception],
) -> Callable[[type[DataclassT]], type[DataclassT]]:
    """Create a class decorator that adds pydantic validation to a frozen dataclass.

    The decorated class is re-created as a pydantic dataclass, so field values are
    validated and converted according to their type annotations on every
    instantiation (including via ``dataclasses.replace``). Annotations are resolved
    at runtime, so this also works in modules using
    ``from __future__ import annotations``.

    This is deliberately a layer on top of existing dataclasses rather than a
    replacement for ``pydantic.dataclasses.dataclass`` at the definition site, for
    two reasons. First, it keeps the classes themselves plain frozen dataclasses,
    which made adopting pydantic non-breaking: algorithm classes — including ones
    defined outside optimagic — are still written as regular dataclasses and gain
    validation through ``mark.minimizer`` without any change to their definition.
    Second, it raises domain-specific exceptions (built by ``make_error``) instead
    of ``pydantic.ValidationError``, which preserves optimagic's exception
    contracts; pydantic itself has no hook to customize the raised exception type.

    Args:
        config: The pydantic config that controls validation behavior.
        make_error: Called with the raised ``pydantic.ValidationError`` to build the
            exception that is raised in its place.

    Returns:
        A class decorator for frozen dataclasses.

    """

    def decorator(cls: type[DataclassT]) -> type[DataclassT]:
        out = pydantic.dataclasses.dataclass(frozen=True, config=config)(cls)
        # pydantic re-creates the class, which loses attributes that tooling and
        # introspection rely on.
        out.__doc__ = cls.__doc__
        out.__annotations__ = dict(cls.__annotations__)
        original_init = out.__init__

        @functools.wraps(original_init)
        def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
            try:
                original_init(self, *args, **kwargs)
            except pydantic.ValidationError as e:
                raise make_error(e) from e

        out.__init__ = __init__  # type: ignore[method-assign]
        return typing.cast("type[DataclassT]", out)

    return decorator


@dataclass(frozen=True)
class IterationHistory(DictLikeAccess):
    """History of iterations in a process.

    Attributes:
        params: A list of parameters used in each iteration.
        criterion: A list of criterion values obtained in each iteration.
        runtime: A list or array of runtimes associated with each iteration.

    """

    params: list[PyTree]
    fun: list[float]
    time: list[float] | NDArray[np.float64]


@dataclass(frozen=True)
class MultiStartIterationHistory(TupleLikeAccess):
    """History of multiple start iterations.

    Attributes:
        history: The main iteration history, representing the best end value.
        local_histories: Optional, a list of local iteration histories.
        exploration: Optional, iteration history for exploration steps.

    """

    history: IterationHistory
    local_histories: list[IterationHistory] | None = None
    exploration: IterationHistory | None = None
