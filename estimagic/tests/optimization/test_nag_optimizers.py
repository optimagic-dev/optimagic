import pytest

from estimagic.optimization.nag_optimizers import _change_evals_per_point_interface
from estimagic.optimization.nag_optimizers import (
    _perturb_jacobian_or_trust_region_step_from_user_value,
)


def test_change_evals_per_point_interface_none():
    res = _change_evals_per_point_interface(None)
    assert res is None


def test_change_evals_per_point_interface_func():
    def return_args(trust_region_radius, min_trust_region, n_iterations, n_restarts):
        return trust_region_radius, min_trust_region, n_iterations, n_restarts

    func = _change_evals_per_point_interface(return_args)
    res = func(delta=0, rho=1, iter=2, nrestarts=3)
    expected = (0, 1, 2, 3)
    assert res == expected


def test_perturb_jacobian_or_trust_region_step_from_user_value_none():
    res = _perturb_jacobian_or_trust_region_step_from_user_value(None)
    assert res == (None, None)


def test_perturb_jacobian_or_trust_region_step_from_user_value_jacobian():
    res = _perturb_jacobian_or_trust_region_step_from_user_value("Jacobian")
    assert res == (True, False)


def test_perturb_jacobian_or_trust_region_step_from_user_value_trust():
    res = _perturb_jacobian_or_trust_region_step_from_user_value("trust_region_step")
    assert res == (False, True)


def test_perturb_jacobian_or_trust_region_step_from_user_value_error():
    with pytest.raises(ValueError):
        _perturb_jacobian_or_trust_region_step_from_user_value("wrong_input")
