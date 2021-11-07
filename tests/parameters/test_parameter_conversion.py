import numpy as np
import pandas as pd
from numpy.testing import assert_array_almost_equal as aaae

from estimagic.parameters.parameter_conversion import get_derivative_conversion_function
from estimagic.parameters.parameter_conversion import get_reparametrize_functions


def test_get_parametrize_functions_with_back_and_forth_conversion():
    params = pd.DataFrame()
    params["value"] = np.arange(10)

    constraints = [{"loc": [2, 3, 4], "type": "fixed"}]
    scaling_factor = np.full(7, 2)
    scaling_offset = np.full(7, -1)

    to_internal, from_internal = get_reparametrize_functions(
        params=params,
        constraints=constraints,
        scaling_factor=scaling_factor,
        scaling_offset=scaling_offset,
    )

    internal = to_internal(params["value"].to_numpy())
    external = from_internal(internal)

    aaae(external, params["value"].to_numpy())


def test_get_derivative_conversion_function_runs():
    params = pd.DataFrame()
    params["value"] = np.arange(10)

    constraints = [{"loc": [2, 3, 4], "type": "fixed"}]
    scaling_factor = np.full(7, 2)
    scaling_offset = np.full(7, -1)

    get_derivative_conversion_function(
        params=params,
        constraints=constraints,
        scaling_factor=scaling_factor,
        scaling_offset=scaling_offset,
    )
