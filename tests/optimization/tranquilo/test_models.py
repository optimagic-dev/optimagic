from estimagic.optimization.tranquilo.models import n_interactions
from estimagic.optimization.tranquilo.models import n_second_order_terms


def test_n_free_params_name_quadratic():
    pass


def test_n_free_params_name_diagonal():
    pass


def test_n_free_params_name_invalid():
    pass


def test_n_free_params_info_linear():
    pass


def test_n_free_params_info_diagonal():
    pass


def test_n_free_params_info_quadratic():
    pass


def test_n_free_params_invalid():
    pass


def test_n_second_order_terms():
    assert n_second_order_terms(3) == 6


def test_n_interactions():
    assert n_interactions(3) == 3


def test_is_second_order_model_info():
    pass


def test_is_second_order_model_model():
    pass


def test_is_second_order_model_invalid():
    pass
