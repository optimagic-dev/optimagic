import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from estimagic.inference.likelihood_inference import do_likelihood_inference


def fake_loglike_pd(params):
    return {"contributions": params["value"], "value": params["value"].sum()}


def fake_loglike_np(params):
    return {"contributions": params["value"].to_numpy(), "value": params["value"].sum()}


loglikes = [fake_loglike_pd, fake_loglike_np]


@pytest.mark.parametrize("loglike", loglikes)
def test_do_likelihood_inference(loglike):
    params = pd.DataFrame()
    params["value"] = np.zeros(3)
    calculated = do_likelihood_inference(loglike, params)

    calculated_cov = calculated["cov"]

    if isinstance(calculated_cov, pd.DataFrame):
        calculated_cov = calculated_cov.to_numpy()

    aaae(calculated_cov, np.eye(3))
