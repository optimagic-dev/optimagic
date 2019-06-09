import warnings
from os import path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pandas.testing import assert_series_equal

from estimagic.optimization.process_constraints import process_constraints
from estimagic.optimization.reparametrize import get_start_params_from_helpers
from estimagic.optimization.reparametrize import make_start_params_helpers
from estimagic.optimization.reparametrize import reparametrize_from_internal
from estimagic.optimization.reparametrize import reparametrize_to_internal

dirname = path.dirname(path.abspath(__file__))
params_fixture = pd.read_csv(path.join(dirname, "fixtures/reparametrize_fixtures.csv"))
params_fixture.set_index(["category", "subcategory", "name"], inplace=True)
for col in ["lower", "internal_lower"]:
    params_fixture[col].fillna(-np.inf, inplace=True)
for col in ["upper", "internal_upper"]:
    params_fixture[col].fillna(np.inf, inplace=True)

external = []
internal = []

for i in range(3):
    ext = params_fixture.copy(deep=True)
    ext.rename(columns={"value{}".format(i): "value"}, inplace=True)
    external.append(ext)

    int_ = params_fixture.copy(deep=True)
    int_.rename(columns={"internal_value{}".format(i): "value"}, inplace=True)
    int_.dropna(subset=["value"], inplace=True)
    int_.drop(columns=["lower", "upper"], inplace=True)
    int_.rename(
        columns={"internal_lower": "lower", "internal_upper": "upper"}, inplace=True
    )
    internal.append(int_)


def constraints(params):
    constr = [
        {"loc": ("c", "c2"), "type": "probability"},
        {
            "loc": [("a", "a", "0"), ("a", "a", "2"), ("a", "a", "4")],
            "type": "fixed",
            "value": [0.1, 0.3, 0.5],
        },
        {"loc": ("e", "off"), "type": "fixed", "value": 0},
        {"loc": "d", "type": "increasing"},
        {"loc": "e", "type": "covariance"},
        {"loc": "f", "type": "covariance"},
        {"loc": "g", "type": "sum", "value": 5},
        {"loc": "h", "type": "equality"},
        {"loc": "i", "type": "equality"},
        {"query": 'subcategory == "j1" | subcategory == "i1"', "type": "equality"},
        {"loc": "k", "type": "sdcorr"},
        {"loc": "l", "type": "covariance"},
        {"locs": ["f", "l"], "type": "pairwise_equality"},
    ]
    constr = process_constraints(constr, params)
    return constr


internal_categories = list("abcdefghik")
external_categories = internal_categories + ["j1", "j2", "l"]

to_test = []
for ext, int_ in zip(external, internal):
    for category in internal_categories:
        to_test.append((ext, int_, category))


@pytest.mark.parametrize("params, expected_internal, category", to_test)
def test_reparametrize_to_internal(params, expected_internal, category):
    constr = constraints(params)
    cols = ["value", "lower", "upper"]
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="indexing past lexsort depth may impact performance."
        )
        calculated = reparametrize_to_internal(params, constr)
        assert_frame_equal(
            calculated.loc[category, cols], expected_internal.loc[category, cols]
        )


to_test = []
for int_, ext in zip(internal, external):
    for category in external_categories:
        to_test.append((int_, ext, category))


@pytest.mark.parametrize("internal, expected_external, category", to_test)
def test_reparametrize_from_internal(internal, expected_external, category):
    constr = constraints(expected_external)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="indexing past lexsort depth may impact performance."
        )
        calculated = reparametrize_from_internal(internal, constr, expected_external)
        assert_series_equal(
            calculated[category], expected_external.loc[category, "value"]
        )


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


def test_get_start_params_from_helpers(helpers_fixture):
    calculated = get_start_params_from_helpers(**helpers_fixture)
    row = [[np.nan, -np.inf, np.inf]]
    expected = pd.DataFrame(
        index=helpers_fixture["params_index"],
        data=row * 3 + [[3.0, -np.inf, np.inf]] + row,
        columns=["value", "lower", "upper"],
    )
    assert calculated.equals(expected)
