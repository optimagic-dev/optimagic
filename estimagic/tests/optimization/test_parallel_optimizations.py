"""Test the constraints processing."""
import numpy as np
import pandas as pd
import pytest

from estimagic.optimization.process_constraints import _process_selectors
from estimagic.optimization.process_constraints import (
    _replace_pairwise_equality_by_equality,
)

constr1 = {"loc": 0, "type": "equality"}
expected1 = {"index": [(0, "a"), (0, "b"), (0, "c")]}


constr2 = {"loc": 2, "type": "fixed"}
expected2 = {"index": [(2, "a"), (2, "b")]}

constr3 = {"locs": [0, 1], "type": "pairwise_equality"}
expected3 = {
    "indices": [[(0, "a"), (0, "b"), (0, "c")], [(1, "f"), (1, "e"), (1, "d")]]
}

test_cases = [(constr1, expected1), (constr2, expected2), (constr3, expected3)]


@pytest.mark.parametrize("constraint, expected", test_cases)
def test_process_selectors(constraint, expected):
    ind_tups = [
        (0, "a"),
        (0, "b"),
        (0, "c"),
        (1, "f"),
        (1, "e"),
        (1, "d"),
        (2, "a"),
        (3, "a"),
        (2, "b"),
    ]
    params_df = pd.DataFrame(
        index=pd.MultiIndex.from_tuples(ind_tups),
        columns=["bla", "blubb"],
        data=np.ones((9, 2)),
    )

    calculated = _process_selectors([constraint], params_df)[0]

    if constraint["type"] != "pairwise_equality":
        indices = [calculated["index"]]
        expected_indices = [expected["index"]]
    else:
        indices = calculated["indices"]
        expected_indices = expected["indices"]

    for calc, exp in zip(indices, expected_indices):
        for ind_tup1, ind_tup2 in zip(calc, exp):
            assert ind_tup1 == ind_tup2


def test_replace_pairwise_equality_by_equality():
    ind_tups = [(0, "a"), (0, "b"), (1, "f"), (1, "e")]
    df = pd.DataFrame(index=pd.MultiIndex.from_tuples(ind_tups))

    constr = {
        "indices": [
            pd.MultiIndex.from_tuples(ind_tups[:2]),
            pd.MultiIndex.from_tuples(ind_tups[2:]),
        ],
        "type": "pairwise_equality",
    }

    expected_ind_tups = [[(0, "a"), (1, "f")], [(0, "b"), (1, "e")]]

    calculated = _replace_pairwise_equality_by_equality([constr], df)

    for constr, ind_tups in zip(calculated, expected_ind_tups):
        for tup_calc, tup_exp in zip(constr["index"], ind_tups):
            assert tup_calc == tup_exp

        assert constr["type"] == "equality"
