from os import path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from estimagic.optimization.process_constraints import process_constraints
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
    int_.drop(["lower", "upper"], axis=1, inplace=True)
    int_.rename(
        columns={"internal_lower": "lower", "internal_upper": "upper"}, inplace=True
    )
    internal.append(int_)


def constraints(params):
    constr = [
        {"loc": ("c", "c2"), "type": "probability"},
        {"loc": [("a", "a", "0"), ("a", "a", "2"), ("a", "a", "4")], "type": "fixed"},
        {"loc": ("e", "off"), "type": "fixed"},
        {"loc": "d", "type": "increasing"},
        {"loc": "e", "type": "covariance"},
        {"loc": "f", "type": "covariance"},
        {"loc": "g", "type": "sum", "value": 5},
        {"loc": "h", "type": "equality"},
        {"loc": "i", "type": "equality"},
        {"query": 'subcategory == "j1" | subcategory == "i1"', "type": "equality"},
        {"loc": "k", "type": "sdcorr"},
    ]
    constr = process_constraints(constr, params)
    return constr


internal_categories = list("abcdefghik")
external_categories = internal_categories + ["j1", "j2"]

to_test = []
for ext, int_ in zip(external, internal):
    for col in ["value", "lower", "upper"]:
        for category in internal_categories:
            to_test.append((ext, int_, col, category))


@pytest.mark.parametrize("params, expected_internal, col, category", to_test)
def test_reparametrize_to_internal(params, expected_internal, col, category):
    constr = constraints(params)
    calculated = reparametrize_to_internal(params, constr)
    assert_series_equal(calculated[col][category], expected_internal[col][category])


to_test = []
for int_, ext in zip(internal, external):
    for category in external_categories:
        to_test.append((int_, ext, category))


@pytest.mark.parametrize("internal, expected_external, category", to_test)
def test_reparametrize_from_internal(internal, expected_external, category):
    constr = constraints(expected_external)
    calculated = reparametrize_from_internal(internal, constr, expected_external)
    assert_series_equal(calculated[category], expected_external.loc[category, "value"])
