from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal as aaae
from pandas.testing import assert_series_equal

from estimagic.optimization.process_constraints import process_constraints
from estimagic.optimization.reparametrize import reparametrize_from_internal
from estimagic.optimization.reparametrize import reparametrize_to_internal


@pytest.fixture
def example_params():
    p = Path(__file__).resolve().parent / "fixtures" / "reparametrize_fixtures.csv"
    params = pd.read_csv(p)
    params.set_index(["category", "subcategory", "name"], inplace=True)
    for col in ["lower", "internal_lower"]:
        params[col].fillna(-np.inf, inplace=True)
    for col in ["upper", "internal_upper"]:
        params[col].fillna(np.inf, inplace=True)
    return params


@pytest.fixture
def all_constraints():
    constraints_dict = {
        "basic_probability": [{"loc": ("c", "c2"), "type": "probability"}],
        "uncorrelated_covariance": [
            {"loc": ("e", "off"), "type": "fixed", "value": 0},
            {"loc": "e", "type": "covariance"},
        ],
        "basic_covariance": [{"loc": "f", "type": "covariance"}],
        "basic_fixed": [
            {
                "loc": [("a", "a", "0"), ("a", "a", "2"), ("a", "a", "4")],
                "type": "fixed",
                "value": [0.1, 0.3, 0.5],
            }
        ],
        "basic_increasing": [{"loc": "d", "type": "increasing"}],
        "basic_equality": [{"loc": "h", "type": "equality"}],
        "query_equality": [
            {"query": 'subcategory == "j1" | subcategory == "i1"', "type": "equality"}
        ],
        "basic_sdcorr": [{"loc": "k", "type": "sdcorr"}],
        "normalized_covariance": [
            {"loc": "m", "type": "covariance"},
            {"loc": ("m", "diag", "a"), "type": "fixed", "value": 4.0},
        ],
    }
    return constraints_dict


to_test = list(
    product(
        [
            "basic_probability",
            "uncorrelated_covariance",
            "basic_covariance",
            "basic_fixed",
            "basic_increasing",
            "basic_equality",
            "query_equality",
            "basic_sdcorr",
            "normalized_covariance",
        ],
        [0, 1, 2],
    )
)


def reduce_params(params, constraints):
    all_locs = []
    for constr in constraints:
        if "query" in constr:
            all_locs = ["i", "j"]
        elif isinstance(constr["loc"], tuple):
            all_locs.append(constr["loc"][0])
        elif isinstance(constr["loc"], list):
            all_locs.append(constr["loc"][0][0])
        else:
            all_locs.append(constr["loc"])
    all_locs = sorted(set(all_locs))
    return params.loc[all_locs].copy()


@pytest.mark.parametrize("case, number", to_test)
def test_reparametrize_to_internal(example_params, all_constraints, case, number):
    constraints = all_constraints[case]
    params = reduce_params(example_params, constraints)
    params["value"] = params[f"value{number}"]

    keep = params[f"internal_value{number}"].notnull()
    expected_internal_values = params[f"internal_value{number}"][keep]
    expected_internal_lower = params["internal_lower"]
    expected_internal_upper = params["internal_upper"]

    pc, pp = process_constraints(constraints, params)

    calculated_internal_values = reparametrize_to_internal(pp, pc)
    calculated_internal_lower = pp["_internal_lower"]
    calculated_internal_upper = pp["_internal_upper"]

    aaae(calculated_internal_values, expected_internal_values)
    aaae(calculated_internal_lower, expected_internal_lower)
    aaae(calculated_internal_upper, expected_internal_upper)


@pytest.mark.parametrize("case, number", to_test)
def test_reparametrize_from_internal(example_params, all_constraints, case, number):
    constraints = all_constraints[case]
    params = reduce_params(example_params, constraints)
    params["value"] = params[f"value{number}"]

    keep = params[f"internal_value{number}"].notnull()

    pc, pp = process_constraints(constraints, params)

    external = reparametrize_from_internal(
        internal=params[f"internal_value{number}"][keep].to_numpy(),
        fixed_values=pp["_internal_fixed_value"].to_numpy(),
        pre_replacements=pp["_pre_replacements"].to_numpy(),
        processed_constraints=pc,
        post_replacements=pp["_post_replacements"].to_numpy(),
        processed_params=pp,
    )

    calculated_external_value = external["value"]
    expected_external_value = params["value"]

    assert_series_equal(calculated_external_value, expected_external_value)


invalid_cases = [
    "basic_probability",
    "uncorrelated_covariance",
    "basic_covariance",
    "basic_increasing",
    "basic_equality",
    "query_equality",
    "basic_sdcorr",
]


@pytest.mark.parametrize("case", invalid_cases)
def test_value_error_if_constraints_are_violated(example_params, all_constraints, case):
    constraints = all_constraints[case]
    params = reduce_params(example_params, constraints)
    for val in ["invalid_value0", "invalid_value1"]:
        params["value"] = params[val]

        with pytest.raises(ValueError):
            process_constraints(constraints, params)
