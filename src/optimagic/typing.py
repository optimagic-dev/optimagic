import functools
from dataclasses import dataclass, fields
from enum import Enum, StrEnum
from typing import (
    TYPE_CHECKING,
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
from pydantic_core import ArgsKwargs, SchemaValidator

if TYPE_CHECKING:
    from pydantic.plugin._schema_validator import PluggableSchemaValidator

PyTree = Any
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

    The ``__init__`` of the decorated class is replaced by one that validates and
    converts the field values according to their type annotations on every
    instantiation (including via ``dataclasses.replace``). Annotations are resolved
    at runtime, so this also works in modules using
    ``from __future__ import annotations``.

    The pydantic validator of a class is only built when the class is instantiated
    for the first time. Building it takes about one millisecond per class, which
    would add up to a noticeable share of the import time of optimagic if it was
    done for all algorithm classes when they are defined.

    This is deliberately a layer on top of existing dataclasses rather than a
    replacement for ``pydantic.dataclasses.dataclass`` at the definition site, for
    three reasons. First, it keeps the classes themselves plain frozen dataclasses,
    which made adopting pydantic non-breaking: algorithm classes — including ones
    defined outside optimagic — are still written as regular dataclasses and gain
    validation through ``mark.minimizer`` without any change to their definition.
    Second, it raises domain-specific exceptions (built by ``make_error``) instead
    of ``pydantic.ValidationError``, which preserves optimagic's exception
    contracts; pydantic itself has no hook to customize the raised exception type.
    Third, pydantic dataclasses collect their fields when the class is defined,
    which is slow even if building the validator is deferred.

    Args:
        config: The pydantic config that controls validation behavior.
        make_error: Called with the raised ``pydantic.ValidationError`` to build the
            exception that is raised in its place.

    Returns:
        A class decorator for frozen dataclasses.

    """

    def decorator(cls: type[DataclassT]) -> type[DataclassT]:
        # pydantic reads the config of a standard dataclass from this attribute.
        cls.__pydantic_config__ = config  # ty:ignore[unresolved-attribute]
        lazy_validator = _LazyValidator(cls)

        @functools.wraps(cls.__init__)
        def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
            try:
                # With self_instance, the validated field values are set on self
                # instead of on a new instance. This is how the __init__ of a
                # pydantic dataclass works.
                lazy_validator.validator.validate_python(
                    ArgsKwargs(args, kwargs), self_instance=self
                )
            except pydantic.ValidationError as e:
                raise make_error(e) from e

        cls.__init__ = __init__  # ty:ignore[invalid-assignment]
        return cls

    return decorator


class _LazyValidator:
    """Build the pydantic validator of a dataclass when it is first needed."""

    def __init__(self, cls: type) -> None:
        self._cls = cls

    @functools.cached_property
    def validator(self) -> "SchemaValidator | PluggableSchemaValidator":
        """The pydantic validator of the dataclass."""
        return pydantic.TypeAdapter(self._cls).validator

    def __getstate__(self) -> dict[str, Any]:
        """Build the validator before pickling.

        cloudpickle pickles classes that are defined in a script or notebook by value
        and does not preserve the identity of ``dataclasses.MISSING``. pydantic cannot
        build a validator from the fields of such an unpickled dataclass, so the
        validator has to be part of the pickled state.

        """
        return {"_cls": self._cls, "validator": self.validator}


@dataclass(frozen=True)
class IterationHistory(DictLikeAccess):
    """History of iterations in a process."""

    params: list[PyTree]
    """A list of parameters used in each iteration."""

    fun: list[float]
    """A list of criterion values obtained in each iteration."""

    time: list[float] | NDArray[np.float64]
    """A list or array of runtimes associated with each iteration."""


@dataclass(frozen=True)
class MultiStartIterationHistory(TupleLikeAccess):
    """History of multiple start iterations."""

    history: IterationHistory
    """The main iteration history, representing the best end value."""

    local_histories: list[IterationHistory] | None = None
    """Optional, a list of local iteration histories."""

    exploration: IterationHistory | None = None
    """Optional, iteration history for exploration steps."""


class PyTreeNamespace(StrEnum):
    """Optree namespaces used by optimagic's pytree functions.

    In the default namespace, numpy arrays, pandas objects and jax arrays are leaves.
    In all extended namespaces they are internal nodes and a params DataFrame with a
    "value" column contributes the entries of the column given by ``data_col``.

    """

    DEFAULT = "optimagic"
    VALUE = "optimagic.value"
    LOWER_BOUND = "optimagic.lower_bound"
    UPPER_BOUND = "optimagic.upper_bound"
    SOFT_LOWER_BOUND = "optimagic.soft_lower_bound"
    SOFT_UPPER_BOUND = "optimagic.soft_upper_bound"

    @property
    def is_extended(self) -> bool:
        """Whether arrays and pandas objects are internal nodes in this namespace."""
        return self is not PyTreeNamespace.DEFAULT

    @property
    def data_col(self) -> str:
        """The params DataFrame column that is flattened in this namespace."""
        if not self.is_extended:
            raise ValueError(
                "The default namespace has no data column; DataFrames are leaves."
            )
        return self.value.removeprefix(f"{PyTreeNamespace.DEFAULT.value}.")
