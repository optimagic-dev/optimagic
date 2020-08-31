from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="function", autouse=True)
def fresh_directory(tmpdir):
    """Each test is executed in a fresh directory."""
    tmpdir.chdir()


@pytest.fixture
def example_params():
    p = (
        Path(__file__).resolve().parent
        / "optimization"
        / "fixtures"
        / "reparametrize_fixtures.csv"
    )
    params = pd.read_csv(p)
    params.set_index(["category", "subcategory", "name"], inplace=True)
    for col in ["lower_bound", "internal_lower"]:
        params[col].fillna(-np.inf, inplace=True)
    for col in ["upper_bound", "internal_upper"]:
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
