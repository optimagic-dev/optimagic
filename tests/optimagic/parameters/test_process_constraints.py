"""Test the pc processing."""

import numpy as np
import pandas as pd
import pytest

import optimagic as om
from optimagic.exceptions import InvalidConstraintError
from optimagic.parameters.bounds import Bounds
from optimagic.parameters.consolidate_constraints import (
    _fold_fixes_into_probability_constraints,
)
from optimagic.parameters.constraint_tools import check_constraints
from optimagic.parameters.process_constraints import (
    _replace_pairwise_equality_by_equality,
    process_constraints,
)


def test_replace_pairwise_equality_by_equality():
    constr = {"indices": [[0, 1], [2, 3]], "type": "pairwise_equality"}

    expected = [
        {"index": [0, 2], "type": "equality"},
        {"index": [1, 3], "type": "equality"},
    ]

    calculated = _replace_pairwise_equality_by_equality([constr])

    assert calculated == expected


@pytest.mark.filterwarnings("ignore:Specifying constraints as a dictionary is")
def test_empty_constraints_work():
    params = pd.DataFrame()
    params["value"] = np.arange(5)
    params["bla"] = list("abcde")

    constraints = [{"query": "bla == 'blubb'", "type": "equality"}]

    check_constraints(params, constraints)


def test_to_many_bounds_in_increasing_constraint_raise_good_error():
    with pytest.raises(InvalidConstraintError):
        check_constraints(
            params=np.arange(3),
            bounds=Bounds(lower=np.arange(3) - 1),
            constraints=om.IncreasingConstraint(selector=lambda x: x[:3]),
        )


def test_fold_fixes_into_probability_constraints_passes_through_when_no_fixes():
    constraints = [{"type": "probability", "index": [0, 1, 2, 3]}]
    fixed_values = np.array([np.nan, np.nan, np.nan, np.nan])
    is_fixed_to_value = np.zeros(4, dtype=bool)

    result = _fold_fixes_into_probability_constraints(
        constraints,
        fixed_values=fixed_values,
        is_fixed_to_value=is_fixed_to_value,
        param_names=["a", "b", "c", "d"],
    )

    assert result == constraints


def test_fold_fixes_into_probability_constraints_shrinks_index_on_zero_fix():
    constraints = [{"type": "probability", "index": [0, 1, 2, 3, 4]}]
    fixed_values = np.array([0.0, np.nan, 0.0, np.nan, np.nan])
    is_fixed_to_value = np.array([True, False, True, False, False])

    result = _fold_fixes_into_probability_constraints(
        constraints,
        fixed_values=fixed_values,
        is_fixed_to_value=is_fixed_to_value,
        param_names=["a", "b", "c", "d", "e"],
    )

    assert len(result) == 1
    assert result[0]["type"] == "probability"
    assert result[0]["index"] == [1, 3, 4]
    # Pure zero fixes leave the transformation dict identical to the no-fix path.
    assert "sum_target" not in result[0]


def test_fold_fixes_into_probability_constraints_passes_other_types_through():
    constraints = [
        {"type": "probability", "index": [0, 1, 2]},
        {"type": "linear", "index": [3, 4]},
    ]
    fixed_values = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
    is_fixed_to_value = np.zeros(5, dtype=bool)

    result = _fold_fixes_into_probability_constraints(
        constraints,
        fixed_values=fixed_values,
        is_fixed_to_value=is_fixed_to_value,
        param_names=["a", "b", "c", "d", "e"],
    )

    assert result == constraints


def test_fold_fixes_into_probability_constraints_shrinks_and_scales_on_non_zero_fix():
    constraints = [{"type": "probability", "index": [0, 1, 2, 3]}]
    fixed_values = np.array([0.3, np.nan, np.nan, np.nan])
    is_fixed_to_value = np.array([True, False, False, False])

    result = _fold_fixes_into_probability_constraints(
        constraints,
        fixed_values=fixed_values,
        is_fixed_to_value=is_fixed_to_value,
        param_names=["a", "b", "c", "d"],
    )

    assert len(result) == 1
    assert result[0]["type"] == "probability"
    assert result[0]["index"] == [1, 2, 3]
    assert np.isclose(result[0]["sum_target"], 0.7)


def test_fold_fixes_into_probability_constraints_rejects_negative_fix():
    constraints = [{"type": "probability", "index": [0, 1, 2]}]
    fixed_values = np.array([-0.1, np.nan, np.nan])
    is_fixed_to_value = np.array([True, False, False])

    with pytest.raises(InvalidConstraintError, match=r"fixed to a value in \[0, 1\)"):
        _fold_fixes_into_probability_constraints(
            constraints,
            fixed_values=fixed_values,
            is_fixed_to_value=is_fixed_to_value,
            param_names=["a", "b", "c"],
        )


def test_fold_fixes_into_probability_constraints_rejects_fixes_summing_to_one():
    constraints = [{"type": "probability", "index": [0, 1, 2, 3]}]
    fixed_values = np.array([0.7, 0.3, np.nan, np.nan])
    is_fixed_to_value = np.array([True, True, False, False])

    with pytest.raises(InvalidConstraintError, match="sum to strictly less than 1"):
        _fold_fixes_into_probability_constraints(
            constraints,
            fixed_values=fixed_values,
            is_fixed_to_value=is_fixed_to_value,
            param_names=["a", "b", "c", "d"],
        )


def test_fold_fixes_into_probability_constraints_rejects_too_few_free():
    constraints = [{"type": "probability", "index": [0, 1, 2]}]
    fixed_values = np.array([0.0, 0.0, np.nan])
    is_fixed_to_value = np.array([True, True, False])

    with pytest.raises(
        InvalidConstraintError, match="at least two non-fixed selected parameters"
    ):
        _fold_fixes_into_probability_constraints(
            constraints,
            fixed_values=fixed_values,
            is_fixed_to_value=is_fixed_to_value,
            param_names=["a", "b", "c"],
        )


def test_process_constraints_folds_zero_fix_on_probability():
    params_vec = np.array([0.0, 0.3, 0.0, 0.3, 0.4])
    constraints = [
        {"type": "probability", "index": [0, 1, 2, 3, 4]},
        {"type": "fixed", "index": [0, 2], "value": np.array([0.0, 0.0])},
    ]

    transformations, constr_info = process_constraints(
        constraints=constraints,
        params_vec=params_vec,
        lower_bounds=np.full(5, -np.inf),
        upper_bounds=np.full(5, np.inf),
        param_names=["p0", "p1", "p2", "p3", "p4"],
    )

    probability_transformations = [
        c for c in transformations if c["type"] == "probability"
    ]
    assert len(probability_transformations) == 1
    assert probability_transformations[0]["index"] == [1, 3, 4]
    assert "sum_target" not in probability_transformations[0]

    # Zero-fixed positions are driven by the fixed-value pipeline.
    assert constr_info["internal_fixed_values"][0] == 0.0
    assert constr_info["internal_fixed_values"][2] == 0.0
    # The pivot (last free selector position) is internally fixed at 1.
    assert constr_info["internal_fixed_values"][4] == 1.0
    # Only free non-pivot selector positions remain in internal_free.
    assert list(constr_info["internal_free"]) == [False, True, False, True, False]


def test_process_constraints_folds_non_zero_fix_on_probability():
    params_vec = np.array([0.2, 0.2, 0.3, 0.3])
    constraints = [
        {"type": "probability", "index": [0, 1, 2, 3]},
        {"type": "fixed", "index": [0], "value": np.array([0.2])},
    ]

    transformations, constr_info = process_constraints(
        constraints=constraints,
        params_vec=params_vec,
        lower_bounds=np.full(4, -np.inf),
        upper_bounds=np.full(4, np.inf),
        param_names=["p0", "p1", "p2", "p3"],
    )

    probability_transformations = [
        c for c in transformations if c["type"] == "probability"
    ]
    assert len(probability_transformations) == 1
    assert probability_transformations[0]["index"] == [1, 2, 3]
    assert np.isclose(probability_transformations[0]["sum_target"], 0.8)
