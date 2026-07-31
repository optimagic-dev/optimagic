import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal as aae
from pybaum import tree_flatten, tree_just_flatten, tree_unflatten

import optimagic as om
from optimagic.constraints import (
    ResolvedEqualityConstraint,
    ResolvedFixedConstraint,
    ResolvedPairwiseEqualityConstraint,
)
from optimagic.exceptions import InvalidConstraintError
from optimagic.parameters.constraints.resolution import (
    resolve_constraints,
    to_legacy_dicts,
)
from optimagic.parameters.tree_conversion import TreeConverter
from optimagic.parameters.tree_registry import get_registry


@pytest.fixture()
def tree_params():
    df = pd.DataFrame({"value": [3, 4], "lower_bound": [0, 0]}, index=["c", "d"])
    params = ([0, np.array([1, 2]), {"a": df, "b": 5}], 6)
    return params


@pytest.fixture()
def tree_params_converter(tree_params):
    registry = get_registry(extended=True)
    _, treedef = tree_flatten(tree_params, registry=registry)

    converter = TreeConverter(
        params_flatten=lambda params: np.array(
            tree_just_flatten(params, registry=registry)
        ),
        params_unflatten=lambda x: tree_unflatten(
            treedef, x.tolist(), registry=registry
        ),
        derivative_flatten=None,  # ty:ignore[invalid-argument-type]
    )
    return converter


@pytest.fixture()
def np_params_converter():
    converter = TreeConverter(
        lambda x: x,
        lambda x: x,
        lambda x: x,
    )
    return converter


PARAM_NAMES = list("abcdefg")


def test_no_constraints(np_params_converter):
    calculated = resolve_constraints(
        constraints=[],
        params=np.arange(5),
        tree_converter=np_params_converter,
        param_names=PARAM_NAMES,
    )
    assert calculated == []


def test_tree_selector(tree_params, tree_params_converter):
    calculated = resolve_constraints(
        constraints=[om.EqualityConstraint(selector=lambda x: x[1])],
        params=tree_params,
        tree_converter=tree_params_converter,
        param_names=PARAM_NAMES,
    )
    assert isinstance(calculated[0], ResolvedEqualityConstraint)
    aae(calculated[0].index, np.array([6]))


def test_tree_selectors_pairwise(tree_params, tree_params_converter):
    constraints = [
        om.PairwiseEqualityConstraint(selectors=[lambda x: x[1], lambda x: x[0][1][0]])
    ]
    calculated = resolve_constraints(
        constraints=constraints,  # ty:ignore[invalid-argument-type]
        params=tree_params,
        tree_converter=tree_params_converter,
        param_names=PARAM_NAMES,
    )
    assert isinstance(calculated[0], ResolvedPairwiseEqualityConstraint)
    aae(calculated[0].indices[0], np.array([6]))
    aae(calculated[0].indices[1], np.array([1]))


def test_numpy_selector(np_params_converter):
    calculated = resolve_constraints(
        constraints=[om.FixedConstraint(selector=lambda x: x[[1, 4]])],
        params=np.arange(6) + 10.0,
        tree_converter=np_params_converter,
        param_names=PARAM_NAMES,
    )
    assert isinstance(calculated[0], ResolvedFixedConstraint)
    aae(calculated[0].index, np.array([1, 4]))


def test_provenance_is_attached(np_params_converter):
    constraints = [
        om.FixedConstraint(selector=lambda x: x[0]),
        om.EqualityConstraint(selector=lambda x: x[[1, 2]]),
    ]
    calculated = resolve_constraints(
        constraints=constraints,  # ty:ignore[invalid-argument-type]
        params=np.arange(6) + 10.0,
        tree_converter=np_params_converter,
        param_names=PARAM_NAMES,
    )
    for position, resolved in enumerate(calculated):
        assert len(resolved.sources) == 1  # ty:ignore[unresolved-attribute]
        assert resolved.sources[0].position == position  # ty:ignore[unresolved-attribute]
        assert resolved.sources[0].constraint is constraints[position]  # ty:ignore[unresolved-attribute]


def test_empty_selections_are_dropped(np_params_converter):
    constraints = [
        om.FixedConstraint(selector=lambda x: x[np.array([], dtype=int)]),
        om.PairwiseEqualityConstraint(
            selectors=[
                lambda x: x[np.array([], dtype=int)],
                lambda x: x[np.array([], dtype=int)],
            ]
        ),
    ]
    calculated = resolve_constraints(
        constraints=constraints,  # ty:ignore[invalid-argument-type]
        params=np.arange(6) + 10.0,
        tree_converter=np_params_converter,
        param_names=PARAM_NAMES,
    )
    assert calculated == []


def test_duplicates_raise(np_params_converter):
    constraints = [om.EqualityConstraint(selector=lambda x: x[[0, 0, 1]])]
    with pytest.raises(InvalidConstraintError, match="duplicates"):
        resolve_constraints(
            constraints=constraints,  # ty:ignore[invalid-argument-type]
            params=np.arange(6) + 10.0,
            tree_converter=np_params_converter,
            param_names=PARAM_NAMES,
        )


def test_failing_selector_raises_invalid_constraint_error(np_params_converter):
    constraints = [om.FixedConstraint(selector=lambda x: x["invalid"])]
    with pytest.raises(InvalidConstraintError, match="select parameters"):
        resolve_constraints(
            constraints=constraints,  # ty:ignore[invalid-argument-type]
            params=np.arange(6) + 10.0,
            tree_converter=np_params_converter,
            param_names=PARAM_NAMES,
        )


# ======================================================================================
# The temporary seam to the dict based pipeline. These tests die with the seam.
# ======================================================================================


def test_to_legacy_dicts_shapes(np_params_converter):
    constraints = [
        om.FixedConstraint(selector=lambda x: x[0]),
        om.EqualityConstraint(selector=lambda x: x[[0, 1]]),
        om.PairwiseEqualityConstraint(selectors=[lambda x: x[0], lambda x: x[1]]),
        om.IncreasingConstraint(selector=lambda x: x[[0, 1]]),
        om.ProbabilityConstraint(selector=lambda x: x[[2, 3]]),
        om.FlatCovConstraint(selector=lambda x: x[:3], regularization=0.2),
        om.LinearConstraint(selector=lambda x: x[[4, 5]], weights=1, upper_bound=5),
    ]
    resolved = resolve_constraints(
        constraints=constraints,  # ty:ignore[invalid-argument-type]
        params=np.arange(6) + 10.0,
        tree_converter=np_params_converter,
        param_names=PARAM_NAMES,
    )
    dicts = to_legacy_dicts(resolved)

    assert [d["type"] for d in dicts] == [
        "fixed",
        "equality",
        "pairwise_equality",
        "increasing",
        "probability",
        "covariance",
        "linear",
    ]
    assert "value" not in dicts[0]
    aae(dicts[2]["indices"][0], np.array([0]))
    assert dicts[5]["regularization"] == 0.2
    aae(dicts[6]["weights"], np.array([1.0, 1.0]))
    assert dicts[6]["upper_bound"] == 5
    assert "lower_bound" not in dicts[6]
    assert "value" not in dicts[6]
