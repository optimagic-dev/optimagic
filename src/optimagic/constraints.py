from abc import ABC, abstractmethod
from dataclasses import KW_ONLY, dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from optimagic.exceptions import InvalidConstraintError
from optimagic.optimization.algo_options import CONSTRAINTS_ABSOLUTE_TOLERANCE
from optimagic.typing import PyTree


class Constraint(ABC):
    """Base class for all constraints used for subtyping."""

    @abstractmethod
    def _to_dict(self) -> dict[str, Any]:
        pass


def identity_selector(x: PyTree) -> PyTree:
    return x


@dataclass(frozen=True)
class FixedConstraint(Constraint):
    """Constraint that fixes the selected parameters at their starting values.

    Attributes:
        selector: A function that takes as input the parameters and returns the subset
            of parameters to be constrained. By default, all parameters are constrained.

    Raises:
        InvalidConstraintError: If the selector is not callable.

    """

    selector: Callable[[PyTree], PyTree] = identity_selector

    def _to_dict(self) -> dict[str, Any]:
        return {"type": "fixed", "selector": self.selector}

    def __post_init__(self) -> None:
        if not callable(self.selector):
            raise InvalidConstraintError("'selector' must be callable.")


@dataclass(frozen=True)
class IncreasingConstraint(Constraint):
    """Constraint that ensures the selected parameters are increasing.

    Attributes:
        selector: A function that takes as input the parameters and returns the subset
            of parameters to be constrained. By default, all parameters are constrained.

    Raises:
        InvalidConstraintError: If the selector is not callable.

    """

    selector: Callable[[PyTree], PyTree] = identity_selector

    def _to_dict(self) -> dict[str, Any]:
        return {"type": "increasing", "selector": self.selector}

    def __post_init__(self) -> None:
        if not callable(self.selector):
            raise InvalidConstraintError("'selector' must be callable.")


@dataclass(frozen=True)
class DecreasingConstraint(Constraint):
    """Constraint that ensures that the selected parameters are decreasing.

    Attributes:
        selector: A function that takes as input the parameters and returns the subset
            of parameters to be constrained. By default, all parameters are constrained.

    Raises:
        InvalidConstraintError: If the selector is not callable.

    """

    selector: Callable[[PyTree], PyTree] = identity_selector

    def _to_dict(self) -> dict[str, Any]:
        return {"type": "decreasing", "selector": self.selector}

    def __post_init__(self) -> None:
        if not callable(self.selector):
            raise InvalidConstraintError("'selector' must be callable.")


@dataclass(frozen=True)
class EqualityConstraint(Constraint):
    """Constraint that ensures that the selected parameters are equal.

    Attributes:
        selector: A function that takes as input the parameters and returns the subset
            of parameters to be constrained. By default, all parameters are constrained.

    Raises:
        InvalidConstraintError: If the selector is not callable.

    """

    selector: Callable[[PyTree], PyTree] = identity_selector

    def _to_dict(self) -> dict[str, Any]:
        return {"type": "equality", "selector": self.selector}

    def __post_init__(self) -> None:
        if not callable(self.selector):
            raise InvalidConstraintError("'selector' must be callable.")


@dataclass(frozen=True)
class ProbabilityConstraint(Constraint):
    """Constraint that ensures that the selected parameters are probabilities.

    This constraint ensures that each of the selected parameters is positive and that
    the sum of the selected parameters is 1.

    Attributes:
        selector: A function that takes as input the parameters and returns the subset
            of parameters to be constrained. By default, all parameters are constrained.

    Raises:
        InvalidConstraintError: If the selector is not callable.

    """

    selector: Callable[[PyTree], PyTree] = identity_selector

    def _to_dict(self) -> dict[str, Any]:
        return {"type": "probability", "selector": self.selector}

    def __post_init__(self) -> None:
        if not callable(self.selector):
            raise InvalidConstraintError("'selector' must be callable.")


@dataclass(frozen=True)
class PairwiseEqualityConstraint(Constraint):
    """Constraint that ensures that groups of selected parameters are equal.

    This constraint ensures that each pair between the selected parameters is equal.

    Attributes:
        selectors: A list of functions that take as input the parameters and return the
            subsets of parameters to be constrained.

    Raises:
        InvalidConstraintError: If the selector is not callable.

    """

    selectors: list[Callable[[PyTree], PyTree]]

    def _to_dict(self) -> dict[str, Any]:
        return {"type": "pairwise_equality", "selectors": self.selectors}

    def __post_init__(self) -> None:
        if len(self.selectors) < 2:
            raise InvalidConstraintError("At least two selectors must be provided.")

        if not all(callable(s) for s in self.selectors):
            raise InvalidConstraintError("All selectors must be callable.")


@dataclass(frozen=True)
class FlatCovConstraint(Constraint):
    """Constraint that ensures the selected parameters are a valid covariance matrix.

    Attributes:
        selector: A function that takes as input the parameters and returns the subset
            of parameters to be constrained. By default, all parameters are constrained.
        regularization: Helps in guiding the optimization towards finding a
            positive definite covariance matrix instead of only a positive semi-definite
            matrix. Larger values correspond to a higher likelihood of positive
            definiteness. Defaults to 0.

    Raises:
        InvalidConstraintError: If the selector is not callable or regularization is
            not a non-negative float or int.

    """

    selector: Callable[[PyTree], PyTree] = identity_selector
    _: KW_ONLY
    regularization: float = 0.0

    def _to_dict(self) -> dict[str, Any]:
        return {
            "type": "covariance",
            "selector": self.selector,
            "regularization": self.regularization,
        }

    def __post_init__(self) -> None:
        if not callable(self.selector):
            raise InvalidConstraintError("'selector' must be callable.")

        if not isinstance(self.regularization, float | int) or self.regularization < 0:
            raise InvalidConstraintError(
                "'regularization' must be a non-negative float or int."
            )


@dataclass(frozen=True)
class FlatSDCorrConstraint(Constraint):
    """Constraint that ensures the selected parameters are a valid correlation matrix.

    This constraint ensures that each of the selected parameters is positive and that
    the sum of the selected parameters is 1.

    Attributes:
        selector: A function that takes as input the parameters and returns the subset
            of parameters to be constrained. By default, all parameters are constrained.
        regularization: Helps in guiding the optimization towards finding a
            positive definite covariance matrix instead of only a positive semi-definite
            matrix. Larger values correspond to a higher likelihood of positive
            definiteness. Defaults to 0.

    Raises:
        InvalidConstraintError: If the selector is not callable or regularization is
            not a non-negative float or int.

    """

    selector: Callable[[PyTree], PyTree] = identity_selector
    _: KW_ONLY
    regularization: float = 0.0

    def _to_dict(self) -> dict[str, Any]:
        return {
            "type": "sdcorr",
            "selector": self.selector,
            "regularization": self.regularization,
        }

    def __post_init__(self) -> None:
        if not callable(self.selector):
            raise InvalidConstraintError("'selector' must be callable.")

        if not isinstance(self.regularization, float | int) or self.regularization < 0:
            raise InvalidConstraintError(
                "'regularization' must be a non-negative float or int."
            )


@dataclass(frozen=True)
class LinearConstraint(Constraint):
    """Constraint that bounds a linear combination of the selected parameters.

    This constraint ensures that a linear combination of the selected parameters with
    the 'weights' is either equal to 'value', or is bounded by 'lower_bound' and
    'upper_bound'.

    Attributes:
        selector: A function that takes as input the parameters and returns the subset
            of parameters to be constrained. By default, all parameters are constrained.
        weights: The weights for the linear combination. If a scalar is provided, it is
            used for all parameters. Otherwise, it must have the same structure as the
            selected parameters.
        lower_bound: The lower bound for the linear combination. Defaults to None.
        upper_bound: The upper bound for the linear combination. Defaults to None.
        value: The value to compare the linear combination to. Defaults to None.

    Raises:
        InvalidConstraintError: If the selector is not callable, or if the weights,
            lower_bound, upper_bound, or value are not valid.

    """

    selector: Callable[[PyTree], ArrayLike | "pd.Series[float]" | float | int] = (
        identity_selector
    )
    _: KW_ONLY
    weights: ArrayLike | "pd.Series[float]" | float | int | None = None
    lower_bound: float | int | None = None
    upper_bound: float | int | None = None
    value: float | int | None = None

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
        if not callable(self.selector):
            raise InvalidConstraintError("'selector' must be callable.")

        if _all_none(self.lower_bound, self.upper_bound, self.value):
            raise InvalidConstraintError(
                "At least one of 'lower_bound', 'upper_bound', or 'value' must be "
                "non-None."
            )
        if self.value is not None and not _all_none(self.lower_bound, self.upper_bound):
            raise InvalidConstraintError(
                "'value' cannot be used with 'lower_bound' or 'upper_bound'."
            )

        if not isinstance(self.weights, np.ndarray | list | pd.Series | float | int):
            raise InvalidConstraintError(
                "'weights' must be an array-like, a pandas Series, a float, or an int."
            )

        if self.lower_bound is not None and not isinstance(
            self.lower_bound, float | int
        ):
            raise InvalidConstraintError("'lower_bound' must be a float or an int.")

        if self.upper_bound is not None and not isinstance(
            self.upper_bound, float | int
        ):
            raise InvalidConstraintError("'upper_bound' must be a float or an int.")

        if self.value is not None and not isinstance(self.value, float | int):
            raise InvalidConstraintError("'value' must be a float or an int.")


@dataclass(frozen=True)
class NonlinearConstraint(Constraint):
    """Constraint that bounds a nonlinear function of the selected parameters.

    This constraint ensures that a nonlinear function of the selected parameters is
    either equal to 'value', or is bounded by 'lower_bound' and 'upper_bound'.

    Attributes:
        selector: A function that takes as input the parameters and returns the subset
            of parameters to be constrained. By default, all parameters are constrained.
        func: The constraint function which is applied to the selected parameters.
        derivative: The derivative of the constraint function with respect to the
            selected parameters. Defaults to None.
        lower_bound: The lower bound for the nonlinear function. Can be a scalar or of
            the same structure as output of the constraint function. Defaults to None.
        upper_bound: The upper bound for the nonlinear function. Can be a scalar or of
            the same structure as output of the constraint function. Defaults to None.
        value: The value to compare the nonlinear function to. Can be a scalar or of
            the same structure as output of the constraint function. Defaults to None.
        tol: The tolerance for the constraint function. Defaults to
            `optimagic.optimization.algo_options.CONSTRAINTS_ABSOLUTE_TOLERANCE`.

    Raises:
        InvalidConstraintError: If the selector is not callable, or if the func,
            derivative, lower_bound, upper_bound, or value are not valid.

    """

    selector: Callable[[PyTree], PyTree] = identity_selector
    _: KW_ONLY
    func: Callable[[PyTree], ArrayLike | "pd.Series[float]" | float] | None = None
    derivative: Callable[[PyTree], PyTree] | None = None
    lower_bound: ArrayLike | "pd.Series[float]" | float | None = None
    upper_bound: ArrayLike | "pd.Series[float]" | float | None = None
    value: ArrayLike | "pd.Series[float]" | float | None = None
    tol: float = CONSTRAINTS_ABSOLUTE_TOLERANCE

    def _to_dict(self) -> dict[str, Any]:
        return {
            "type": "nonlinear",
            "selector": self.selector,
            **_select_non_none(
                func=self.func,
                derivative=self.derivative,
                # In the dict representation, we write _bounds instead of _bound.
                lower_bounds=self.lower_bound,
                upper_bounds=self.upper_bound,
                value=self.value,
                tol=self.tol,
            ),
        }

    def __post_init__(self) -> None:
        if not callable(self.selector):
            raise InvalidConstraintError("'selector' must be callable.")

        if _all_none(self.lower_bound, self.upper_bound, self.value):
            raise InvalidConstraintError(
                "At least one of 'lower_bound', 'upper_bound', or 'value' must be "
                "non-None."
            )
        if self.value is not None and not _all_none(self.lower_bound, self.upper_bound):
            raise InvalidConstraintError(
                "'value' cannot be used with 'lower_bound' or 'upper_bound'."
            )

        if self.tol is not None and (
            not isinstance(self.tol, float | int) or self.tol < 0
        ):
            raise InvalidConstraintError("'tol' must be non-negative.")

        if self.func is None or not callable(self.func):
            raise InvalidConstraintError("'func' must be callable.")

        if self.derivative is not None and not callable(self.derivative):
            raise InvalidConstraintError("'derivative' must be callable.")


def _all_none(*args: Any) -> bool:
    return all(v is None for v in args)


def _select_non_none(**kwargs: Any) -> dict[str, Any]:
    return {k: v for k, v in kwargs.items() if v is not None}
