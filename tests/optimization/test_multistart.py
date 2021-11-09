import numpy as np
import pandas as pd
from estimagic.examples.criterion_functions import sos_dict_criterion
from estimagic.optimization.optimize import minimize
from numpy.testing import assert_array_almost_equal as aaae


def test_multistart_minimize_with_sum_of_squares_at_defaults():
    params = pd.DataFrame()
    params["value"] = np.arange(4)
    params["soft_lower_bound"] = [-10] * 4
    params["soft_upper_bounds"] = [10] * 4

    res = minimize(
        criterion=sos_dict_criterion,
        params=params,
        algorithm="scipy_lbfgsb",
        multistart=True,
    )

    # assert on history of local optima
    # assert on history of optimal parameters

    assert np.allclose(res["solution_criterion"], 0)
    aaae(res["solution_params"]["value"], np.zeros(4))
