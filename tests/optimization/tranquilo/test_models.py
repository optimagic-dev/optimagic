import numpy as np
import pytest
from estimagic.optimization.tranquilo.models import (
    ModelInfo,
    ScalarModel,
    is_second_order_model,
    n_free_params,
    n_interactions,
    n_second_order_terms,
)


def test_n_free_params_name_quadratic():
    assert n_free_params(dim=2, info_or_name="quadratic") == 1 + 2 + 3
    assert n_free_params(dim=3, info_or_name="quadratic") == 1 + 3 + 6
    assert n_free_params(dim=9, info_or_name="quadratic") == 1 + 9 + 45


def test_n_free_params_name_diagonal():
    assert n_free_params(dim=2, info_or_name="diagonal") == 1 + 2 + 2
    assert n_free_params(dim=3, info_or_name="diagonal") == 1 + 3 + 3
    assert n_free_params(dim=9, info_or_name="diagonal") == 1 + 9 + 9


def test_n_free_params_name_invalid():
    with pytest.raises(ValueError):
        assert n_free_params(dim=3, info_or_name="invalid")


@pytest.mark.parametrize("dim", [2, 3, 9])
def test_n_free_params_info_linear(dim):
    info = ModelInfo(has_squares=False, has_interactions=False)
    assert n_free_params(dim, info) == 1 + dim


@pytest.mark.parametrize("dim", [2, 3, 9])
def test_n_free_params_info_diagonal(dim):
    info = ModelInfo(has_squares=True, has_interactions=False)
    assert n_free_params(dim, info) == 1 + dim + dim


@pytest.mark.parametrize("dim", [2, 3, 9])
def test_n_free_params_info_quadratic(dim):
    info = ModelInfo(has_squares=True, has_interactions=True)
    assert n_free_params(dim, info) == 1 + dim + dim + (dim * (dim - 1) // 2)


def test_n_free_params_invalid():
    model = ScalarModel(intercept=1.0, linear_terms=np.ones(1), square_terms=np.ones(1))
    with pytest.raises(ValueError):
        n_free_params(dim=1, info_or_name=model)


def test_n_second_order_terms():
    assert n_second_order_terms(3) == 6


def test_n_interactions():
    assert n_interactions(3) == 3


@pytest.mark.parametrize("has_squares", [True, False])
@pytest.mark.parametrize("has_interactions", [True, False])
def test_is_second_order_model_info(has_squares, has_interactions):
    model_info = ModelInfo(has_squares=has_squares, has_interactions=has_interactions)
    assert is_second_order_model(model_info) == has_squares or has_interactions


def test_is_second_order_model_model():
    model = ScalarModel(intercept=1.0, linear_terms=np.ones(1))
    assert is_second_order_model(model) is False

    model = ScalarModel(intercept=1.0, linear_terms=np.ones(1), square_terms=np.ones(1))
    assert is_second_order_model(model) is True


def test_is_second_order_model_invalid():
    model = np.linalg.lstsq
    with pytest.raises(TypeError):
        is_second_order_model(model)
