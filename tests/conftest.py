import os

import pandas as pd
import pytest
import statsmodels.api as sm


@pytest.fixture(scope="function", autouse=True)
def fresh_directory(tmp_path):
    """Each test is executed in a fresh directory."""
    os.chdir(tmp_path)


@pytest.fixture
def logit_inputs():
    spector_data = sm.datasets.spector.load_pandas()
    spector_data.exog = sm.add_constant(spector_data.exog)
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
