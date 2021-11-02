"""Test the different options of fides."""
import numpy as np
import pytest
from estimagic.config import IS_FIDES_INSTALLED
from estimagic.optimization.fides_optimizers import fides
from fides.hessian_approximation import Broyden
from fides.hessian_approximation import SR1
from numpy.testing import assert_array_almost_equal as aaae


test_cases = [
    {},
    {"hessian_update_strategy": "bfgs"},
    {"hessian_update_strategy": "BFGS"},
    {"hessian_update_strategy": SR1()},
    {"hessian_update_strategy": Broyden(phi=0.5)},
]


def criterion_and_derivative(x, task, algorithm_info):
    if task == "criterion":
        return (x ** 2).sum()
    elif task == "derivative":
        return 2 * x
    elif task == "criterion_and_derivative":
        return (x ** 2).sum(), 2 * x
    else:
        raise ValueError(f"Unknown task: {task}")


@pytest.mark.skipif(not IS_FIDES_INSTALLED, reason="fides not installed.")
@pytest.mark.parametrize("algo_options", test_cases)
def test_ipopt_algo_options(algo_options):
    res = fides(
        criterion_and_derivative=criterion_and_derivative,
        x=np.array([1, -50, 30]),
        lower_bounds=np.array([-100, -100, -100]),
        upper_bounds=np.array([100, 100, 100]),
        **algo_options,
    )
    aaae(res["solution_x"], np.zeros(3), decimal=6)
