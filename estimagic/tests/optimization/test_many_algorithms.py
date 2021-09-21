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
from estimagic.optimization import GLOBAL_ALGORITHMS
from estimagic.optimization.optimize import minimize

AVAILABLE_ALGORITHMS = {
    alg: AVAILABLE_ALGORITHMS[alg]
    for alg in AVAILABLE_ALGORITHMS
    if alg.startswith("nlopt")
}

LOCAL_ALGORITHMS = {
    name: AVAILABLE_ALGORITHMS[name]
    for name in AVAILABLE_ALGORITHMS
    if name not in GLOBAL_ALGORITHMS
}

GLOBAL_ALGORITHMS_AVAILABLE = [
    name for name in AVAILABLE_ALGORITHMS if name in GLOBAL_ALGORITHMS
]

BOUNDED_ALGORITHMS = []
for name, func in LOCAL_ALGORITHMS.items():
    arguments = list(inspect.signature(func).parameters)
    if "lower_bounds" in arguments and "upper_bounds" in arguments:
        BOUNDED_ALGORITHMS.append(name)


@pytest.mark.parametrize("algorithm", LOCAL_ALGORITHMS)
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


@pytest.mark.parametrize("algorithm", GLOBAL_ALGORITHMS_AVAILABLE)
def test_global_algorithms_on_sum_of_squares(algorithm):
    params = pd.DataFrame()
    params["value"] = [2.5, 2, -1]
    params["lower_bound"] = [0.5, -1, -1.5]
    params["upper_bound"] = [3, 3, -0.5]
    res = minimize(
        criterion=sos_dict_criterion,
        params=params,
        algorithm=algorithm,
    )
    aaae(
        res["solution_params"]["value"].to_numpy(), np.array([0.5, 0, -0.5]), decimal=1
    )
