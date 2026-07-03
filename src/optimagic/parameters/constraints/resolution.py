"""Resolve the selectors of user constraints to flat parameter positions.

This is the first stage of constraints processing. Each user constraint selects a
subset of the parameters via a selector function. Here, the selectors are evaluated
on a helper pytree that has the same structure as the user provided params but
contains the positions of the parameters in the flat parameter vector. The result is
a list of resolved constraints (see
:mod:`optimagic.parameters.constraints.types`) that refer to parameters by
position and carry provenance information for error messages.

"""

from __future__ import annotations

import warnings
from collections import Counter
from typing import Any, Callable

import numpy as np
import pandas as pd
from pybaum import tree_just_flatten

from optimagic.constraints import (
    Constraint,
    DecreasingConstraint,
    EqualityConstraint,
    FixedConstraint,
    FlatCovConstraint,
    FlatSDCorrConstraint,
    IncreasingConstraint,
    LinearConstraint,
    PairwiseEqualityConstraint,
    ProbabilityConstraint,
)
from optimagic.exceptions import InvalidConstraintError
from optimagic.parameters.constraints.types import (
    ConstraintSource,
    FloatArray,
    IntArray,
    ResolvedConstraint,
    ResolvedCovariance,
    ResolvedDecreasing,
    ResolvedEquality,
    ResolvedFixed,
    ResolvedIncreasing,
    ResolvedLinear,
    ResolvedPairwiseEquality,
    ResolvedProbability,
    ResolvedSDCorr,
)
from optimagic.parameters.tree_conversion import TreeConverter
from optimagic.parameters.tree_registry import get_registry
from optimagic.typing import PyTree


def resolve_constraints(
    constraints: list[Constraint],
    params: PyTree,
    tree_converter: TreeConverter,
    param_names: list[str],
) -> list[ResolvedConstraint]:
    """Resolve the selectors of user constraints to flat parameter positions.

    Args:
        constraints: The user provided constraints. Nonlinear constraints are not
            allowed here because they are not handled via reparametrization.
        params: The user provided params.
        tree_converter: Converter between the params pytree and its flat version.
        param_names: Names of the flat parameters. Used for error messages.

    Returns:
        The resolved constraints. Constraints with empty selections are dropped.

    Raises:
        InvalidConstraintError: If a selector fails or selects parameters more than
            once, if the selections of a pairwise equality constraint have different
            lengths, or if the weights of a linear constraint cannot be aligned with
            the selected parameters.

    """
    registry = get_registry(extended=True)
    n_params = len(tree_converter.params_flatten(params))
    helper = tree_converter.params_unflatten(np.arange(n_params))

    resolved: list[ResolvedConstraint] = []
    for position, constraint in enumerate(constraints):
        source = ConstraintSource(constraint=constraint, position=position)
        new: ResolvedConstraint | None
        if isinstance(constraint, PairwiseEqualityConstraint):
            new = _resolve_pairwise_equality_constraint(
                constraint=constraint,
                source=source,
                helper=helper,
                registry=dict(registry),
                param_names=param_names,
            )
        else:
            new = _resolve_single_selector_constraint(
                constraint=constraint,
                source=source,
                helper=helper,
                registry=dict(registry),
                param_names=param_names,
            )
        if new is not None:
            resolved.append(new)

    return resolved


def to_legacy_dicts(resolved: list[ResolvedConstraint]) -> list[dict[str, Any]]:
    """Convert resolved constraints to the dict format of the old pipeline.

    This is a temporary seam during the constraints refactoring: selector resolution
    is already typed, whereas checking and consolidation still work on dictionaries
    with an "index" field. It is removed once the remaining pipeline stages are
    typed.

    """
    out: list[dict[str, Any]] = []
    for constraint in resolved:
        new: dict[str, Any]
        if isinstance(constraint, ResolvedFixed):
            new = {"type": "fixed", "index": constraint.index}
            if constraint.value is not None:
                new["value"] = constraint.value
        elif isinstance(constraint, ResolvedEquality):
            new = {"type": "equality", "index": constraint.index}
        elif isinstance(constraint, ResolvedPairwiseEquality):
            new = {"type": "pairwise_equality", "indices": list(constraint.indices)}
        elif isinstance(constraint, ResolvedIncreasing):
            new = {"type": "increasing", "index": constraint.index}
        elif isinstance(constraint, ResolvedDecreasing):
            new = {"type": "decreasing", "index": constraint.index}
        elif isinstance(constraint, ResolvedProbability):
            new = {"type": "probability", "index": constraint.index}
        elif isinstance(constraint, ResolvedCovariance):
            new = {
                "type": "covariance",
                "index": constraint.index,
                "regularization": constraint.regularization,
            }
        elif isinstance(constraint, ResolvedSDCorr):
            new = {
                "type": "sdcorr",
                "index": constraint.index,
                "regularization": constraint.regularization,
            }
        else:
            new = {
                "type": "linear",
                "index": constraint.index,
                "weights": constraint.weights,
            }
            if np.isfinite(constraint.lower_bound):
                new["lower_bound"] = constraint.lower_bound
            if np.isfinite(constraint.upper_bound):
                new["upper_bound"] = constraint.upper_bound
            if np.isfinite(constraint.value):
                new["value"] = constraint.value
        out.append(new)
    return out


def _resolve_single_selector_constraint(
    constraint: Constraint,
    source: ConstraintSource,
    helper: PyTree,
    registry: dict[type, Any],
    param_names: list[str],
) -> ResolvedConstraint | None:
    index = _select_positions(
        selector=constraint.selector,  # type: ignore[attr-defined]
        source=source,
        helper=helper,
        registry=registry,
        param_names=param_names,
    )

    if len(index) == 0:
        return None

    out: ResolvedConstraint
    if isinstance(constraint, FixedConstraint):
        # the value attribute only exists on the FixedValueConstraint subclass that
        # supports deprecated dictionary constraints with explicit values
        out = ResolvedFixed(
            index=index,
            sources=(source,),
            value=getattr(constraint, "value", None),
        )
    elif isinstance(constraint, EqualityConstraint):
        out = ResolvedEquality(index=index, sources=(source,))
    elif isinstance(constraint, IncreasingConstraint):
        out = ResolvedIncreasing(index=index, sources=(source,))
    elif isinstance(constraint, DecreasingConstraint):
        out = ResolvedDecreasing(index=index, sources=(source,))
    elif isinstance(constraint, ProbabilityConstraint):
        out = ResolvedProbability(index=index, sources=(source,))
    elif isinstance(constraint, FlatCovConstraint):
        out = ResolvedCovariance(
            index=index,
            regularization=constraint.regularization,
            sources=(source,),
        )
    elif isinstance(constraint, FlatSDCorrConstraint):
        out = ResolvedSDCorr(
            index=index,
            regularization=constraint.regularization,
            sources=(source,),
        )
    elif isinstance(constraint, LinearConstraint):
        out = ResolvedLinear(
            index=index,
            weights=_align_linear_weights(constraint.weights, index, source),
            lower_bound=(
                -np.inf if constraint.lower_bound is None else constraint.lower_bound
            ),
            upper_bound=(
                np.inf if constraint.upper_bound is None else constraint.upper_bound
            ),
            value=np.nan if constraint.value is None else constraint.value,
            sources=(source,),
        )
    else:
        msg = (
            f"Constraints of type {type(constraint).__name__} cannot be enforced "
            f"via reparametrization. The problematic constraint is "
            f"{source.describe()}."
        )
        raise InvalidConstraintError(msg)

    return out


def _resolve_pairwise_equality_constraint(
    constraint: PairwiseEqualityConstraint,
    source: ConstraintSource,
    helper: PyTree,
    registry: dict[type, Any],
    param_names: list[str],
) -> ResolvedPairwiseEquality | None:
    indices = tuple(
        _select_positions(
            selector=selector,
            source=source,
            helper=helper,
            registry=registry,
            param_names=param_names,
        )
        for selector in constraint.selectors
    )

    lengths = [len(index) for index in indices]
    if len(set(lengths)) != 1:
        msg = (
            "All selections of a pairwise equality constraint need to have the "
            f"same length. You have lengths {lengths} in {source.describe()}."
        )
        raise InvalidConstraintError(msg)

    if len(indices[0]) == 0:
        return None

    return ResolvedPairwiseEquality(indices=indices, sources=(source,))


def _select_positions(
    selector: Callable[[PyTree], PyTree],
    source: ConstraintSource,
    helper: PyTree,
    registry: dict[type, Any],
    param_names: list[str],
) -> IntArray:
    """Evaluate a selector on the position helper tree and validate the selection."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
            raw = selector(helper)
            flat = tree_just_flatten(raw, registry=registry)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        msg = (
            "An error occurred when trying to select parameters for "
            f"{source.describe()}."
        )
        raise InvalidConstraintError(msg) from e

    index = np.array(flat).astype(np.int64)
    _fail_if_duplicates(index, source, param_names)
    return index


def _align_linear_weights(
    weights: Any, index: IntArray, source: ConstraintSource
) -> FloatArray:
    """Broadcast and length-check the weights of a linear constraint."""
    if isinstance(weights, (np.ndarray, list, tuple, pd.Series)):
        if len(weights) != len(index):
            msg = (
                f"weights of length {len(weights)} could not be aligned with the "
                f"{len(index)} selected parameters in {source.describe()}."
            )
            raise InvalidConstraintError(msg)
        out = np.asarray(weights, dtype=np.float64)
    elif isinstance(weights, (float, int)):
        out = np.full(len(index), float(weights))
    else:
        msg = (
            f"Invalid type for linear weights: {type(weights)}. The problematic "
            f"constraint is {source.describe()}."
        )
        raise InvalidConstraintError(msg)
    return out


def _fail_if_duplicates(
    index: IntArray, source: ConstraintSource, param_names: list[str]
) -> None:
    duplicates = [pos for pos, count in Counter(index.tolist()).items() if count > 1]
    if duplicates:
        names = [param_names[pos] for pos in duplicates]
        msg = (
            "Error while processing constraints. There are duplicates in the "
            f"selected parameters. The parameters that were selected more than "
            f"once are {names}. The problematic constraint is {source.describe()}."
        )
        raise InvalidConstraintError(msg)
