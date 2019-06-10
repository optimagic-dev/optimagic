import numpy as np
import pandas as pd
import pytest

from estimagic.optimization.start_helpers import get_start_params_from_free_params
from estimagic.optimization.start_helpers import make_start_params_helpers


@pytest.fixture
def helpers_fixture():
    out = {}
    ind_tups = [("a", 0), ("a", 1), ("a", 2), ("b", 1), ("b", 2)]
    index = pd.MultiIndex.from_tuples(ind_tups)
    out["params_index"] = index

    out["constraints"] = [
        {"loc": ("b", 1), "type": "fixed", "value": 3},
        {"loc": [("a", 0), ("a", 1), ("a", 2)], "type": "equality"},
    ]

    out["free"] = pd.DataFrame(
        index=pd.MultiIndex.from_tuples([("a", 0), ("b", 2)]),
        columns=["value", "lower", "upper"],
        data=[[np.nan, -np.inf, np.inf]] * 2,
    )

    out["fixed"] = pd.DataFrame(
        index=pd.MultiIndex.from_tuples([("a", 1), ("a", 2), ("b", 1)]),
        columns=["value", "lower", "upper"],
        data=[[np.nan, -np.inf, np.inf]] * 2 + [[3.0, -np.inf, np.inf]],
    )
    return out


def test_make_start_params_helpers(helpers_fixture):
    calculated_free, calculated_fixed = make_start_params_helpers(
        helpers_fixture["params_index"], helpers_fixture["constraints"]
    )
    assert calculated_free.equals(helpers_fixture["free"])
    assert calculated_fixed.equals(helpers_fixture["fixed"])


def test_get_start_params_from_free_params(helpers_fixture):
    helpers_fixture["free"].loc[("a", 0), "value"] = 2
    helpers_fixture.pop("fixed")
    calculated = get_start_params_from_free_params(**helpers_fixture)
    expected = pd.DataFrame(
        index=helpers_fixture["params_index"],
        data=[[2.0, -np.inf, np.inf]] * 3
        + [[3.0, -np.inf, np.inf]]
        + [[np.nan, -np.inf, np.inf]],
        columns=["value", "lower", "upper"],
    )
    assert calculated.equals(expected)
