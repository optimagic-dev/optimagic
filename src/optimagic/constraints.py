from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar

import pandas as pd
from numpy.typing import ArrayLike

from optimagic.exceptions import InvalidConstraintError
from optimagic.typing import PyTree


class Constraint(ABC):
    """Base class for all constraints used for subtyping."""

    selector: Callable[[PyTree], PyTree]

    @abstractmethod
    def _to_dict(self) -> dict[str, Any]:
        pass


def identity_selector(x: PyTree) -> PyTree:
    return x


ConstraintValue = TypeVar("ConstraintValue", bound=PyTree)


@dataclass(frozen=True)
class FixedConstraint(Constraint, Generic[ConstraintValue]):
    selector: Callable[[PyTree], ConstraintValue] = identity_selector
    value: ConstraintValue | None = None

    def _to_dict(self) -> dict[str, Any]:
        return {
            "type": "fixed",
            "selector": self.selector,
            **_select_non_none(value=self.value),
        }


@dataclass(frozen=True)
class IncreasingConstraint(Constraint):
    selector: Callable[[PyTree], PyTree] = identity_selector

    def _to_dict(self) -> dict[str, Any]:
        return {"type": "increasing", "selector": self.selector}


@dataclass(frozen=True)
class DecreasingConstraint(Constraint):
    selector: Callable[[PyTree], PyTree] = identity_selector

    def _to_dict(self) -> dict[str, Any]:
        return {"type": "decreasing", "selector": self.selector}


@dataclass(frozen=True)
class EqualityConstraint(Constraint):
    selector: Callable[[PyTree], PyTree] = identity_selector

    def _to_dict(self) -> dict[str, Any]:
        return {"type": "equality", "selector": self.selector}


@dataclass(frozen=True)
class PairwiseEqualityConstraint(Constraint):
    selectors: list[Callable[[PyTree], PyTree]]

    def _to_dict(self) -> dict[str, Any]:
        return {"type": "pairwise_equality", "selectors": self.selectors}


@dataclass(frozen=True)
class ProbabilityConstraint(Constraint):
    selector: Callable[[PyTree], PyTree] = identity_selector

    def _to_dict(self) -> dict[str, Any]:
        return {"type": "probability", "selector": self.selector}


@dataclass(frozen=True)
class CovarianceConstraint(Constraint):
    selector: Callable[[PyTree], PyTree] = identity_selector

    def _to_dict(self) -> dict[str, Any]:
        return {"type": "covariance", "selector": self.selector}


@dataclass(frozen=True)
class SDCorrConstraint(Constraint):
    selector: Callable[[PyTree], PyTree] = identity_selector

    def _to_dict(self) -> dict[str, Any]:
        return {"type": "sdcorr", "selector": self.selector}


ArrayLikeSeriesOrFloat = TypeVar(
    "ArrayLikeSeriesOrFloat",
    bound=(ArrayLike | pd.Series | float),  # type: ignore
)


@dataclass(frozen=True)
class LinearConstraint(Constraint, Generic[ArrayLikeSeriesOrFloat]):
    weights: ArrayLike | pd.Series | pd.DataFrame  # type: ignore
    lower_bound: ArrayLikeSeriesOrFloat | None = None
    upper_bound: ArrayLikeSeriesOrFloat | None = None
    value: ArrayLikeSeriesOrFloat | None = None
    selector: Callable[[PyTree], PyTree] = identity_selector

    def _to_dict(self) -> dict[str, Any]:
        return {
            "type": "linear",
            "selector": self.selector,
            "weights": self.weights,
            **_select_non_none(
                lower_bound=self.lower_bound,
                upper_bound=self.upper_bound,
                value=self.value,
            ),
        }

    def __post_init__(self) -> None:
        if _all_none(self.lower_bound, self.upper_bound, self.value):
            raise InvalidConstraintError(
                "At least one of 'lower_bound', 'upper_bound', or 'value' must be "
                "non-None."
            )
        if self.value is not None and not _all_none(self.lower_bound, self.upper_bound):
            raise InvalidConstraintError(
                "'value' cannot be used with 'lower_bound' or 'upper_bound'."
            )


@dataclass(frozen=True)
class NonlinearConstraint(Constraint, Generic[ArrayLikeSeriesOrFloat]):
    func: Callable[[PyTree], ArrayLikeSeriesOrFloat]
    derivative: Callable[[PyTree], PyTree] | None = None
    lower_bound: ArrayLikeSeriesOrFloat | None = None
    upper_bound: ArrayLikeSeriesOrFloat | None = None
    value: ArrayLikeSeriesOrFloat | None = None
    tol: float | None = None
    selector: Callable[[PyTree], PyTree] = identity_selector

    def _to_dict(self) -> dict[str, Any]:
        return {
            "type": "nonlinear",
            "selector": self.selector,
            "func": self.func,
            **_select_non_none(
                derivative=self.derivative,
                lower_bound=self.lower_bound,
                upper_bound=self.upper_bound,
                value=self.value,
                tol=self.tol,
            ),
        }

    def __post_init__(self) -> None:
        if _all_none(self.lower_bound, self.upper_bound, self.value):
            raise InvalidConstraintError(
                "At least one of 'lower_bound', 'upper_bound', or 'value' must be "
                "non-None."
            )
        if self.value is not None and not _all_none(self.lower_bound, self.upper_bound):
            raise InvalidConstraintError(
                "'value' cannot be used with 'lower_bound' or 'upper_bound'."
            )

        if self.tol is not None and self.tol < 0:
            raise InvalidConstraintError("'tol' must be non-negative.")


def _all_none(*args: Any) -> bool:
    return all(v is None for v in args)


def _select_non_none(**kwargs: Any) -> dict[str, Any]:
    return {k: v for k, v in kwargs.items() if v is not None}
