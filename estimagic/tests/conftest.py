import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm


@pytest.fixture(scope="function", autouse=True)
def fresh_directory(tmp_path):
    """Each test is executed in a fresh directory."""
    os.chdir(tmp_path)


@pytest.fixture(scope="session")
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
def logit_inputs():
    spector_data = sm.datasets.spector.load_pandas()
    spector_data.exog = sm.add_constant(spector_data.exog)
    y = spector_data.endog
    x_df = sm.add_constant(spector_data.exog)
    out = {
        "y": spector_data.endog,
        "x": x_df.to_numpy(),
        "params": pd.DataFrame([-10, 2, 0.2, 2], index=x_df.columns, columns=["value"]),
    }
    return out


@pytest.fixture
def logit_object():
    spector_data = sm.datasets.spector.load_pandas()
    spector_data.exog = sm.add_constant(spector_data.exog)
    logit_mod = sm.Logit(spector_data.endog, spector_data.exog)
    return logit_mod


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
