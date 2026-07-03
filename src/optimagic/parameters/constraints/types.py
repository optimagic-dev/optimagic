"""Typed internal representation of constraints.

User provided constraints are resolved into the dataclasses defined here before any
further processing. A resolved constraint refers to parameters by their integer
positions in the flat parameter vector and carries provenance information that links
it back to the user provided constraints it was derived from. The provenance is used
to phrase error messages in terms of what the user actually wrote, even after
constraints have been rewritten or merged.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from optimagic.constraints import Constraint

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]


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


def _as_position_array(positions: Any) -> IntArray:
    """Convert positions to a read-only integer array."""
    out = np.array(positions).astype(np.int64)
    out.flags.writeable = False
    return out


def _as_float_array(values: Any) -> FloatArray:
    """Convert values to a read-only float array."""
    out = np.array(values).astype(np.float64)
    out.flags.writeable = False
    return out


@dataclass(frozen=True, eq=False)
class ResolvedFixed:
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


@dataclass(frozen=True, eq=False)
class ResolvedEquality:
    """Enforce that the selected parameters are equal.

    Attributes:
        index: Positions of the equal parameters in the flat parameter vector.
        sources: The user constraints this constraint was derived from.

    """

    index: IntArray
    sources: tuple[ConstraintSource, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "index", _as_position_array(self.index))


@dataclass(frozen=True, eq=False)
class ResolvedPairwiseEquality:
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


@dataclass(frozen=True, eq=False)
class ResolvedIncreasing:
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


@dataclass(frozen=True, eq=False)
class ResolvedDecreasing:
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


@dataclass(frozen=True, eq=False)
class ResolvedProbability:
    """Enforce that the selected parameters are positive and sum to one.

    Attributes:
        index: Positions of the parameters in the flat parameter vector.
        sources: The user constraints this constraint was derived from.

    """

    index: IntArray
    sources: tuple[ConstraintSource, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "index", _as_position_array(self.index))


@dataclass(frozen=True, eq=False)
class ResolvedCovariance:
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


@dataclass(frozen=True, eq=False)
class ResolvedSDCorr:
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


@dataclass(frozen=True, eq=False)
class ResolvedLinear:
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


ResolvedConstraint: TypeAlias = (
    ResolvedFixed
    | ResolvedEquality
    | ResolvedPairwiseEquality
    | ResolvedIncreasing
    | ResolvedDecreasing
    | ResolvedProbability
    | ResolvedCovariance
    | ResolvedSDCorr
    | ResolvedLinear
)
