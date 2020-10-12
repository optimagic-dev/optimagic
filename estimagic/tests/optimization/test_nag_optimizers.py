import pytest

from estimagic.optimization.nag_optimizers import _build_options_dict
from estimagic.optimization.nag_optimizers import _change_evals_per_point_interface
from estimagic.optimization.nag_optimizers import _get_fast_start_method_from_user_value


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


def test_get_fast_start_method_from_user_value_auto():
    res = _get_fast_start_method_from_user_value("auto")
    assert res == (None, None)


def test_get_fast_start_method_from_user_value_jacobian():
    res = _get_fast_start_method_from_user_value("jacobian")
    assert res == (True, False)


def test_get_fast_start_method_from_user_value_trust():
    res = _get_fast_start_method_from_user_value("trust_region")
    assert res == (False, True)


def test_get_fast_start_method_from_user_value_error():
    with pytest.raises(ValueError):
        _get_fast_start_method_from_user_value("wrong_input")


def test_build_options_dict_none():
    default = {"a": 1, "b": 2}
    assert default == _build_options_dict(None, default)


def test_build_options_dict_override():
    default = {"a": 1, "b": 2}
    user_input = {"a": 0}
    res = _build_options_dict(user_input, default)
    expected = {"a": 0, "b": 2}
    assert res == expected


def test_build_options_dict_invalid_key():
    default = {"a": 1, "b": 2}
    user_input = {"other_key": 0}
    with pytest.raises(ValueError):
        _build_options_dict(user_input, default)
