import pytest

from optimagic.optimization.create_optimization_problem import (
    pre_process_user_algorithm,
)
from optimagic.optimizers.scipy_optimizers import ScipyLBFGSB


def test_pre_process_user_algorithm_valid_string():
    got = pre_process_user_algorithm("scipy_lbfgsb")
    assert isinstance(got, ScipyLBFGSB)


def test_pre_process_user_algorithm_invalid_string():
    with pytest.raises(ValueError):
        pre_process_user_algorithm("not_an_algorithm")


def test_pre_process_user_algorithm_valid_instance():
    got = pre_process_user_algorithm(ScipyLBFGSB())
    assert isinstance(got, ScipyLBFGSB)


def test_pre_process_user_algorithm_valid_class():
    got = pre_process_user_algorithm(ScipyLBFGSB)
    assert isinstance(got, ScipyLBFGSB)
