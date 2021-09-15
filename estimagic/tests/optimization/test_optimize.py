"""Tests for (almost) algorithm independent properties of maximize and minimize."""
import numpy as np
import pandas as pd

from estimagic.optimization.optimize import maximize


def test_sign_is_switched_back_after_maximization():
    params = pd.DataFrame()
    params["value"] = [1, 2, 3]
    res = maximize(
        lambda params: 1 - params["value"] @ params["value"],
        params=params,
        algorithm="scipy_lbfgsb",
    )

    assert np.allclose(res["solution_criterion"], 1)
