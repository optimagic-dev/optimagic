"""Test the pc processing."""
import numpy as np
import pandas as pd
from estimagic.parameters.constraint_tools import check_constraints
from estimagic.parameters.process_constraints import (
    _replace_pairwise_equality_by_equality,
)


def test_replace_pairwise_equality_by_equality():
    constr = {"indices": [[0, 1], [2, 3]], "type": "pairwise_equality"}

    expected = [
        {"index": [0, 2], "type": "equality"},
        {"index": [1, 3], "type": "equality"},
    ]

    calculated = _replace_pairwise_equality_by_equality([constr])

    assert calculated == expected


def test_empty_constraints_work():
    params = pd.DataFrame()
    params["value"] = np.arange(5)
    params["bla"] = list("abcde")

    constraints = [{"query": "bla == 'blubb'", "type": "equality"}]

    check_constraints(params, constraints)
