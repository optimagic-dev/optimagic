"""Test the pc processing."""
from copy import deepcopy

import numpy as np
import pandas as pd
from estimagic.optimization.optimize import maximize
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


def test_bug_from_copenhagen_presentation():
    # make sure maximum of work hours is optimal
    def u(params):
        return params["work"]["hours"] ** 2

    start_params = {
        "work": {"hourly_wage": 25.5, "hours": 2_000},
        "time_budget": 24 * 7 * 365,
    }

    def return_all_but_working_hours(params):
        out = deepcopy(params)
        del out["work"]["hours"]
        return out

    res = maximize(
        criterion=u,
        params=start_params,
        algorithm="scipy_lbfgsb",
        constraints=[
            {"selector": return_all_but_working_hours, "type": "fixed"},
            {
                "selector": lambda p: [p["work"]["hours"], p["time_budget"]],
                "type": "increasing",
            },
        ],
        lower_bounds={"work": {"hours": 0}},
    )

    assert np.allclose(res.params["work"]["hours"], start_params["time_budget"])
