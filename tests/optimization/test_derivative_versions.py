"""Test different versions of specifying a criterion functions.

Here we want to take:
- Few representative algorithms (derivative based, derivative free, least squares)
- One basic criterion function (sum of squares)
- Many ways of specifying derivatives or not.
- Maximize and minimize since this might make a difference

"""
import numpy as np
import pandas as pd
import pytest
from estimagic.decorators import switch_sign
from estimagic.examples.criterion_functions import sos_criterion_and_gradient
from estimagic.examples.criterion_functions import sos_criterion_and_jacobian
from estimagic.examples.criterion_functions import sos_dict_criterion
from estimagic.examples.criterion_functions import sos_dict_derivative
from estimagic.examples.criterion_functions import sos_dict_derivative_with_pd_objects
from estimagic.examples.criterion_functions import (
    sos_double_dict_criterion_and_derivative_with_pd_objects,
)
from estimagic.examples.criterion_functions import sos_gradient
from estimagic.examples.criterion_functions import sos_jacobian
from estimagic.examples.criterion_functions import sos_pandas_gradient
from estimagic.examples.criterion_functions import sos_pandas_jacobian
from estimagic.optimization.optimize import maximize
from estimagic.optimization.optimize import minimize
from numpy.testing import assert_array_almost_equal as aaae

algorithms = ["scipy_lbfgsb", "scipy_ls_dogbox", "scipy_neldermead"]

ls_algorithms = {"scipy_ls_dogbox"}


scalar_derivatives = [None, sos_gradient, sos_pandas_gradient]

scalar_criterion_and_derivtives = [None, sos_criterion_and_gradient]

ls_derivatives = [None, sos_jacobian, sos_pandas_jacobian]


ls_criterion_and_derivatives = [sos_criterion_and_jacobian]

dict_derivatives = [sos_dict_derivative, None, sos_dict_derivative_with_pd_objects]

dict_criterion_and_derivatives = [
    None,
    sos_double_dict_criterion_and_derivative_with_pd_objects,
]


valid_cases = []
invalid_cases = []
for algo in algorithms:
    if algo in ls_algorithms:
        for deriv in ls_derivatives:
            for crit_and_deriv in ls_criterion_and_derivatives:
                valid_cases.append(("minimize", algo, deriv, crit_and_deriv))
                invalid_cases.append(("maximize", algo, deriv, crit_and_deriv))
    else:
        for deriv in scalar_derivatives:
            for crit_and_deriv in scalar_criterion_and_derivtives:
                for direction in ["maximize", "minimize"]:
                    valid_cases.append((direction, algo, deriv, crit_and_deriv))

    for deriv in dict_derivatives:
        for crit_and_deriv in dict_criterion_and_derivatives:
            if algo in ls_algorithms:
                valid_cases.append(("minimize", algo, deriv, crit_and_deriv))
            else:
                for direction in ["maximize", "minimize"]:
                    valid_cases.append((direction, algo, deriv, crit_and_deriv))


@pytest.mark.parametrize(
    "direction, algorithm, derivative, criterion_and_derivative", valid_cases
)
def test_valid_derivative_versions(
    direction, algorithm, derivative, criterion_and_derivative
):
    start_params = pd.DataFrame()
    start_params["value"] = [1, 2, 3]

    if direction == "minimize":
        res = minimize(
            criterion=sos_dict_criterion,
            params=start_params,
            algorithm=algorithm,
            derivative=derivative,
            criterion_and_derivative=criterion_and_derivative,
            error_handling="raise",
        )
    else:
        deriv = derivative if derivative is None else switch_sign(derivative)
        crit_and_deriv = (
            criterion_and_derivative
            if criterion_and_derivative is None
            else switch_sign(criterion_and_derivative)
        )
        res = maximize(
            criterion=switch_sign(sos_dict_criterion),
            params=start_params,
            algorithm=algorithm,
            derivative=deriv,
            criterion_and_derivative=crit_and_deriv,
            error_handling="raise",
        )

    aaae(res["solution_params"]["value"].to_numpy(), np.zeros(3), decimal=4)


@pytest.mark.parametrize(
    "direction, algorithm, derivative, criterion_and_derivative", invalid_cases
)
def test_invalid_derivative_versions(
    direction, algorithm, derivative, criterion_and_derivative
):
    start_params = pd.DataFrame()
    start_params["value"] = [1, 2, 3]

    if direction == "minimize":
        with pytest.raises(ValueError):
            minimize(
                criterion=sos_dict_criterion,
                params=start_params,
                algorithm=algorithm,
                derivative=derivative,
                criterion_and_derivative=criterion_and_derivative,
            )
    else:
        deriv = derivative if derivative is None else switch_sign(derivative)
        crit_and_deriv = (
            criterion_and_derivative
            if criterion_and_derivative is None
            else switch_sign(criterion_and_derivative)
        )
        with pytest.raises(ValueError):
            maximize(
                criterion=switch_sign(sos_dict_criterion),
                params=start_params,
                algorithm=algorithm,
                derivative=deriv,
                criterion_and_derivative=crit_and_deriv,
            )
