"""Test all availabl algorithms on a simple sum of squares function.

- only minimize
- only numerical derivative

"""
import inspect

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from estimagic.examples.criterion_functions import sos_dict_criterion
from estimagic.optimization import AVAILABLE_ALGORITHMS
from estimagic.optimization.optimize import minimize


BOUNDED_ALGORITHMS = []
for name, func in AVAILABLE_ALGORITHMS.items():
    arguments = list(inspect.signature(func).parameters)
    if "lower_bounds" in arguments and "upper_bounds" in arguments:
        BOUNDED_ALGORITHMS.append(name)


@pytest.mark.parametrize("algorithm", AVAILABLE_ALGORITHMS)
def test_algorithm_on_sum_of_squares(algorithm):
    params = pd.DataFrame()
    params["value"] = [1, 2, 3]

    res = minimize(
        criterion=sos_dict_criterion,
        params=params,
        algorithm=algorithm,
    )

    aaae(res["solution_params"]["value"].to_numpy(), np.zeros(3), decimal=4)


@pytest.mark.parametrize("algorithm", BOUNDED_ALGORITHMS)
def test_algorithm_on_sum_of_squares_with_binding_bounds(algorithm):
    params = pd.DataFrame()
    params["value"] = [3, 2, -3]
    params["lower_bound"] = [1, np.nan, np.nan]
    params["upper_bound"] = [np.nan, np.nan, -1]

    res = minimize(
        criterion=sos_dict_criterion,
        params=params,
        algorithm=algorithm,
    )

    aaae(res["solution_params"]["value"].to_numpy(), np.array([1, 0, -1]), decimal=3)
