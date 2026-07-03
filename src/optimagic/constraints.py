"""User-facing constraint classes and their resolved internal counterparts.

Each constraint class describes a constraint on a subset of the parameters that is
selected via a selector function. During constraints processing, the selectors are
resolved to positions in the flat parameter vector (``Constraint._resolve``), which
produces the ``Resolved*`` dataclass defined next to each constraint class. A
resolved constraint refers to parameters by their integer positions and carries
provenance information that links it back to the user provided constraints it was
derived from. The provenance is used to phrase error messages in terms of what the
user actually wrote, even after constraints have been rewritten or merged.

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING, Any, Callable, TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from optimagic.exceptions import InvalidConstraintError
from optimagic.optimization.algo_options import CONSTRAINTS_ABSOLUTE_TOLERANCE
from optimagic.typing import PyTree

if TYPE_CHECKING:
    from optimagic.parameters.constraints.resolution import ResolutionContext

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]


class Constraint(ABC):
    """Base class for all constraints used for subtyping."""

    @abstractmethod
    def _to_dict(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def _resolve(self, context: ResolutionContext) -> ResolvedConstraint | None:
        """Resolve the constraint's selectors to flat parameter positions.

        Returns None if the selection is empty, in which case the constraint is
        dropped.

        """


@dataclass(frozen=True)
class ConstraintSource:
    """User constraint from which an internal constraint was derived.

    Attributes:
        constraint: The user provided constraint object. Dictionary constraints are
            converted to constraint objects before resolution, so this is always a
            Constraint instance.
        position: The position of the constraint in the user provided list of
            constraints.

    """

    constraint: Constraint
    position: int

    def describe(self) -> str:
        return f"constraint {self.position}: {self.constraint!r}"


class ResolvedConstraint(ABC):  # noqa: B024
    """Base class for all resolved constraints used for subtyping."""


def _as_position_array(positions: Any) -> IntArray:
    """Convert positions to an int64 array."""
    return np.asarray(positions, dtype=np.int64)


def _as_float_array(values: Any) -> FloatArray:
    """Convert values to a float64 array."""
    return np.asarray(values, dtype=np.float64)


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

    def _resolve(self, context: ResolutionContext) -> ResolvedFixedConstraint | None:
        index = context.select(self.selector)
        if len(index) == 0:
            return None
        return ResolvedFixedConstraint(index=index, sources=(context.source,))


@dataclass(frozen=True, eq=False)
class ResolvedFixedConstraint(ResolvedConstraint):
    """Fix the selected parameters.

    Attributes:
        index: Positions of the fixed parameters in the flat parameter vector.
        sources: The user constraints this constraint was derived from.
        value: Explicit values at which the parameters are fixed. None means the
            parameters are fixed at their start values. Explicit values only exist
            for deprecated dictionary constraints and must coincide with the start
            values.

    """

    index: IntArray
    sources: tuple[ConstraintSource, ...]
    value: Any = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "index", _as_position_array(self.index))


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

    def _resolve(
        self, context: ResolutionContext
    ) -> ResolvedIncreasingConstraint | None:
        index = context.select(self.selector)
        if len(index) == 0:
            return None
        return ResolvedIncreasingConstraint(index=index, sources=(context.source,))


@dataclass(frozen=True, eq=False)
class ResolvedIncreasingConstraint(ResolvedConstraint):
    """Enforce that the selected parameters are weakly increasing.

    Attributes:
        index: Positions of the parameters in the flat parameter vector, in the
            order in which they have to be increasing.
        sources: The user constraints this constraint was derived from.

    """

    index: IntArray
    sources: tuple[ConstraintSource, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "index", _as_position_array(self.index))


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

    def _resolve(
        self, context: ResolutionContext
    ) -> ResolvedDecreasingConstraint | None:
        index = context.select(self.selector)
        if len(index) == 0:
            return None
        return ResolvedDecreasingConstraint(index=index, sources=(context.source,))


@dataclass(frozen=True, eq=False)
class ResolvedDecreasingConstraint(ResolvedConstraint):
    """Enforce that the selected parameters are weakly decreasing.

    Attributes:
        index: Positions of the parameters in the flat parameter vector, in the
            order in which they have to be decreasing.
        sources: The user constraints this constraint was derived from.

    """

    index: IntArray
    sources: tuple[ConstraintSource, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "index", _as_position_array(self.index))


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

    def _resolve(self, context: ResolutionContext) -> ResolvedEqualityConstraint | None:
        index = context.select(self.selector)
        if len(index) == 0:
            return None
        return ResolvedEqualityConstraint(index=index, sources=(context.source,))


@dataclass(frozen=True, eq=False)
class ResolvedEqualityConstraint(ResolvedConstraint):
    """Enforce that the selected parameters are equal.

    Attributes:
        index: Positions of the equal parameters in the flat parameter vector.
        sources: The user constraints this constraint was derived from.

    """

    index: IntArray
    sources: tuple[ConstraintSource, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "index", _as_position_array(self.index))


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

    def _resolve(
        self, context: ResolutionContext
    ) -> ResolvedProbabilityConstraint | None:
        index = context.select(self.selector)
        if len(index) == 0:
            return None
        return ResolvedProbabilityConstraint(index=index, sources=(context.source,))


@dataclass(frozen=True, eq=False)
class ResolvedProbabilityConstraint(ResolvedConstraint):
    """Enforce that the selected parameters are positive and sum to one.

    Attributes:
        index: Positions of the parameters in the flat parameter vector.
        sources: The user constraints this constraint was derived from.

    """

    index: IntArray
    sources: tuple[ConstraintSource, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "index", _as_position_array(self.index))


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

    def _resolve(
        self, context: ResolutionContext
    ) -> ResolvedPairwiseEqualityConstraint | None:
        indices = tuple(context.select(selector) for selector in self.selectors)

        lengths = [len(index) for index in indices]
        if len(set(lengths)) != 1:
            msg = (
                "All selections of a pairwise equality constraint need to have the "
                f"same length. You have lengths {lengths} in "
                f"{context.source.describe()}."
            )
            raise InvalidConstraintError(msg)

        if len(indices[0]) == 0:
            return None

        return ResolvedPairwiseEqualityConstraint(
            indices=indices, sources=(context.source,)
        )


@dataclass(frozen=True, eq=False)
class ResolvedPairwiseEqualityConstraint(ResolvedConstraint):
    """Enforce equality between corresponding parameters of multiple selections.

    Attributes:
        indices: One position array per selection. All arrays have the same length
            and corresponding entries are constrained to be equal.
        sources: The user constraints this constraint was derived from.

    """

    indices: tuple[IntArray, ...]
    sources: tuple[ConstraintSource, ...]

    def __post_init__(self) -> None:
        frozen = tuple(_as_position_array(index) for index in self.indices)
        object.__setattr__(self, "indices", frozen)


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

    def _resolve(self, context: ResolutionContext) -> ResolvedFlatCovConstraint | None:
        index = context.select(self.selector)
        if len(index) == 0:
            return None
        return ResolvedFlatCovConstraint(
            index=index,
            regularization=self.regularization,
            sources=(context.source,),
        )


@dataclass(frozen=True, eq=False)
class ResolvedFlatCovConstraint(ResolvedConstraint):
    """Enforce that the selected parameters form a valid covariance matrix.

    Attributes:
        index: Positions of the parameters in the flat parameter vector. The
            parameters are the lower triangle of the covariance matrix in C order.
        regularization: Lower bound on the diagonal of the Cholesky factor of the
            covariance matrix that helps to keep the matrix positive definite.
        sources: The user constraints this constraint was derived from.

    """

    index: IntArray
    regularization: float
    sources: tuple[ConstraintSource, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "index", _as_position_array(self.index))


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

    def _resolve(
        self, context: ResolutionContext
    ) -> ResolvedFlatSDCorrConstraint | None:
        index = context.select(self.selector)
        if len(index) == 0:
            return None
        return ResolvedFlatSDCorrConstraint(
            index=index,
            regularization=self.regularization,
            sources=(context.source,),
        )


@dataclass(frozen=True, eq=False)
class ResolvedFlatSDCorrConstraint(ResolvedConstraint):
    """Enforce that the selected parameters are valid standard deviations and
    correlations.

    Attributes:
        index: Positions of the parameters in the flat parameter vector. The
            parameters are the standard deviations followed by the lower triangle
            of the correlation matrix in C order.
        regularization: Lower bound on the diagonal of the Cholesky factor of the
            implied covariance matrix that helps to keep the matrix positive
            definite.
        sources: The user constraints this constraint was derived from.

    """

    index: IntArray
    regularization: float
    sources: tuple[ConstraintSource, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "index", _as_position_array(self.index))


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

    def _resolve(self, context: ResolutionContext) -> ResolvedLinearConstraint | None:
        index = context.select(self.selector)
        if len(index) == 0:
            return None
        return ResolvedLinearConstraint(
            index=index,
            weights=self._aligned_weights(index, context.source),
            lower_bound=-np.inf if self.lower_bound is None else self.lower_bound,
            upper_bound=np.inf if self.upper_bound is None else self.upper_bound,
            value=np.nan if self.value is None else self.value,
            sources=(context.source,),
        )

    def _aligned_weights(self, index: IntArray, source: ConstraintSource) -> FloatArray:
        """Broadcast and length-check the weights against the selected positions."""
        if isinstance(self.weights, (np.ndarray, list, tuple, pd.Series)):
            if len(self.weights) != len(index):
                msg = (
                    f"weights of length {len(self.weights)} could not be aligned "
                    f"with the {len(index)} selected parameters in "
                    f"{source.describe()}."
                )
                raise InvalidConstraintError(msg)
            out = np.asarray(self.weights, dtype=np.float64)
        elif isinstance(self.weights, (float, int)):
            out = np.full(len(index), float(self.weights))
        else:
            msg = (
                f"Invalid type for linear weights: {type(self.weights)}. The "
                f"problematic constraint is {source.describe()}."
            )
            raise InvalidConstraintError(msg)
        return out


@dataclass(frozen=True, eq=False)
class ResolvedLinearConstraint(ResolvedConstraint):
    """Restrict a weighted sum of the selected parameters.

    Attributes:
        index: Positions of the parameters in the flat parameter vector.
        weights: Weights of the parameters in the weighted sum, aligned with index.
        lower_bound: Lower bound on the weighted sum; -inf if there is none.
        upper_bound: Upper bound on the weighted sum; inf if there is none.
        value: Value at which the weighted sum is fixed; nan if it is not fixed.
        sources: The user constraints this constraint was derived from.

    """

    index: IntArray
    weights: FloatArray
    sources: tuple[ConstraintSource, ...]
    lower_bound: float = -np.inf
    upper_bound: float = np.inf
    value: float = np.nan

    def __post_init__(self) -> None:
        object.__setattr__(self, "index", _as_position_array(self.index))
        object.__setattr__(self, "weights", _as_float_array(self.weights))


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

    def _resolve(self, context: ResolutionContext) -> ResolvedConstraint | None:
        msg = (
            f"Constraints of type {type(self).__name__} cannot be enforced "
            f"via reparametrization. The problematic constraint is "
            f"{context.source.describe()}."
        )
        raise InvalidConstraintError(msg)


def _all_none(*args: Any) -> bool:
    return all(v is None for v in args)


def _select_non_none(**kwargs: Any) -> dict[str, Any]:
    return {k: v for k, v in kwargs.items() if v is not None}
