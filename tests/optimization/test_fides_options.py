"""Test the different options of fides."""
import numpy as np
import pytest
from estimagic.config import IS_FIDES_INSTALLED
from estimagic.optimization.fides_optimizers import fides
from fides.hessian_approximation import Broyden
from fides.hessian_approximation import FX
from fides.hessian_approximation import SR1
from numpy.testing import assert_allclose

test_cases_no_contribs_needed = [
    {},
    {"hessian_update_strategy": "bfgs"},
    {"hessian_update_strategy": "BFGS"},
    {"hessian_update_strategy": SR1()},
    {"hessian_update_strategy": Broyden(phi=0.5)},
    {"hessian_update_strategy": "sr1"},
    {"hessian_update_strategy": "DFP"},
    {"hessian_update_strategy": "bb"},
    {"hessian_update_strategy": "bg"},
    {"convergence_absolute_criterion_tolerance": 1e-6},
    {"convergence_relative_criterion_tolerance": 1e-6},
    {"convergence_absolute_params_tolerance": 1e-6},
    {"convergence_absolute_gradient_tolerance": 1e-6},
    {"convergence_relative_gradient_tolerance": 1e-6},
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
@pytest.mark.parametrize("algo_options", test_cases_no_contribs_needed)
def test_fides_correct_algo_options(algo_options):
    res = fides(
        criterion_and_derivative=criterion_and_derivative,
        x=np.array([1, -5, 3]),
        lower_bounds=np.array([-10, -10, -10]),
        upper_bounds=np.array([10, 10, 10]),
        **algo_options,
    )
    assert_allclose(res["solution_x"], np.zeros(3), atol=5e-4)


test_cases_needing_contribs = [
    {"hessian_update_strategy": FX()},
    {"hessian_update_strategy": "ssm"},
    {"hessian_update_strategy": "TSSM"},
    {"hessian_update_strategy": "gnsbfgs"},
]


@pytest.mark.skipif(not IS_FIDES_INSTALLED, reason="fides not installed.")
@pytest.mark.parametrize("algo_options", test_cases_needing_contribs)
def test_fides_unimplemented_algo_options(algo_options):
    with pytest.raises(NotImplementedError):
        fides(
            criterion_and_derivative=criterion_and_derivative,
            x=np.array([1, -5, 3]),
            lower_bounds=np.array([-10, -10, -10]),
            upper_bounds=np.array([10, 10, 10]),
            **algo_options,
        )


@pytest.mark.skipif(not IS_FIDES_INSTALLED, reason="fides not installed.")
def test_fides_with_super_high_convergence_criteria():
    with pytest.raises(AssertionError):
        res = fides(
            criterion_and_derivative=criterion_and_derivative,
            x=np.array([1, -5, 3]),
            lower_bounds=np.array([-10, -10, -10]),
            upper_bounds=np.array([10, 10, 10]),
            convergence_absolute_criterion_tolerance=10,
            convergence_relative_criterion_tolerance=10,
            convergence_absolute_params_tolerance=10,
            convergence_absolute_gradient_tolerance=10,
            convergence_relative_gradient_tolerance=10,
        )
        assert_allclose(res["solution_x"], np.zeros(3), atol=5e-4)
