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


def constraints_and_processed_params(params):
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
    ]
    constr, params = process_constraints(constr, params)
    return constr, params


to_test = []
for ext, int_ in zip(external, internal):
    for col in ["value", "lower", "upper"]:
        to_test.append((ext, int_, col))


@pytest.mark.parametrize("params, expected_internal, col", to_test)
def test_reparametrize_to_internal(params, expected_internal, col):
    constraints, params = constraints_and_processed_params(params)
    calculated = reparametrize_to_internal(params, constraints)
    assert_series_equal(calculated[col], expected_internal[col])


@pytest.mark.parametrize("internal, expected_external", zip(internal, external))
def test_reparametrize_from_internal(internal, expected_external):
    constraints, params = constraints_and_processed_params(expected_external)
    calculated = reparametrize_from_internal(internal, constraints, params)
    assert_series_equal(calculated, expected_external["value"])
