import numpy as np
import pandas as pd
import pytest
from estimagic.examples.criterion_functions import sos_dict_criterion
from estimagic.examples.criterion_functions import sos_scalar_criterion
from estimagic.optimization.optimize import minimize
from numpy.testing import assert_array_almost_equal as aaae


criteria = [sos_dict_criterion, sos_scalar_criterion]


@pytest.mark.parametrize("criterion", criteria)
def test_multistart_minimize_with_sum_of_squares_at_defaults(criterion):
    params = pd.DataFrame()
    params["value"] = np.arange(4)
    params["soft_lower_bound"] = [-5] * 4
    params["soft_upper_bound"] = [10] * 4

    res = minimize(
        criterion=sos_dict_criterion,
        params=params,
        algorithm="scipy_lbfgsb",
        multistart=True,
    )

    assert "multistart_info" in res
    ms_info = res["multistart_info"]
    assert len(ms_info["exploration_sample"]) == 40
    assert len(ms_info["exploration_results"]) == 40
    assert all(isinstance(entry, dict) for entry in ms_info["exploration_results"])
    assert all(isinstance(entry, dict) for entry in ms_info["local_optima"])
    assert all(isinstance(entry, pd.DataFrame) for entry in ms_info["start_parameters"])
    assert np.allclose(res["solution_criterion"], 0)
    aaae(res["solution_params"]["value"], np.zeros(4))
