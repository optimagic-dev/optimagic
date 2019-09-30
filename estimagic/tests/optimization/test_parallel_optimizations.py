"""Test the constraints processing."""
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal

from estimagic.optimization.optimize import minimize


def test_replace_pairwise_equality_by_equality():
    """
    Test an easy single optimization.
    """

    def rosen(x):
        """The Rosenbrock function

        Args:
            x (pd.Series): Series with the parameters.

        """
        return np_rosen(x["value"].to_numpy())

    def np_rosen(x):

        return sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)

    params = pd.DataFrame()
    params["value"] = np.array([1.3, 0.7, 1.0, 1.9, 1.2])
    params["fixed"] = [False, False, True, False, False]
    params["lower"] = [-1, -1, -1, -1, -1]
    params["upper"] = [5, 5, 5, 5, 5]

    result = minimize(rosen, params, "nlopt_neldermead")[0]["internal_x"]
    expected_result = [1, 1, 1, 1, 1]

    assert_array_almost_equal(result, expected_result)
