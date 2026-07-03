"""Resolve the selectors of user constraints to flat parameter positions.

This is the first stage of constraints processing. Each user constraint selects a
subset of the parameters via a selector function. Here, the selectors are evaluated
on a helper pytree that has the same structure as the user provided params but
contains the positions of the parameters in the flat parameter vector. The result is
a list of resolved constraints (see :mod:`optimagic.constraints`) that refer to
parameters by position and carry provenance information for error messages.

The per-constraint resolution logic lives in the ``Constraint._resolve`` methods in
:mod:`optimagic.constraints`. This module provides the loop over all constraints and
the :class:`ResolutionContext` that the methods work with.

"""

from __future__ import annotations

import warnings
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from pybaum import tree_just_flatten

from optimagic.constraints import (
    Constraint,
    ConstraintSource,
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
from optimagic.exceptions import InvalidConstraintError
from optimagic.parameters.tree_conversion import TreeConverter
from optimagic.parameters.tree_registry import get_registry
from optimagic.typing import PyTree


@dataclass(frozen=True)
class ResolutionContext:
    """Everything a constraint needs to resolve its selectors to flat positions.

    Attributes:
        helper: Pytree with the same structure as the user provided params whose
            leaves are the positions of the parameters in the flat parameter vector.
        registry: Pytree registry used to flatten selections on the helper tree.
        param_names: Names of the flat parameters. Used for error messages.
        source: Provenance of the constraint that is being resolved.

    """

    helper: PyTree
    registry: dict[type, Any]
    param_names: list[str]
    source: ConstraintSource

    def select(self, selector: Callable[[PyTree], PyTree]) -> IntArray:
        """Evaluate a selector on the helper tree and validate the selection."""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
                raw = selector(self.helper)
                flat = tree_just_flatten(raw, registry=self.registry)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            msg = (
                "An error occurred when trying to select parameters for "
                f"{self.source.describe()}."
            )
            raise InvalidConstraintError(msg) from e

        index = np.array(flat).astype(np.int64)
        self._fail_if_duplicates(index)
        return index

    def _fail_if_duplicates(self, index: IntArray) -> None:
        duplicates = [
            pos for pos, count in Counter(index.tolist()).items() if count > 1
        ]
        if duplicates:
            names = [self.param_names[pos] for pos in duplicates]
            msg = (
                "Error while processing constraints. There are duplicates in the "
                f"selected parameters. The parameters that were selected more than "
                f"once are {names}. The problematic constraint is "
                f"{self.source.describe()}."
            )
            raise InvalidConstraintError(msg)


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
        context = ResolutionContext(
            helper=helper,
            registry=dict(registry),
            param_names=param_names,
            source=ConstraintSource(constraint=constraint, position=position),
        )
        new = constraint._resolve(context)
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
        elif isinstance(constraint, ResolvedLinear):
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
        else:
            raise TypeError(
                f"Unsupported resolved constraint type: {type(constraint).__name__}."
            )
        out.append(new)
    return out
