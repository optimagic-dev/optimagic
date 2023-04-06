"""Tests for the process_arguments function and subfunctions.

When testing process_arguments we should only test the values of outputs that somehow
depend on the inputs, not the values with static defaults.

"""
import pytest
import numpy as np
from estimagic.optimization.tranquilo.process_arguments import (
    process_arguments,
    _process_batch_size,
    _process_sample_size,
    _process_model_type,
    _process_search_radius_factor,
    _process_acceptance_decider,
    _process_model_fitter,
    _process_residualize,
    _process_n_evals_at_start,
    ceil_to_multiple,
)


def test_process_arguments_scalar_deterministic():
    res = process_arguments(
        functype="scalar",
        criterion=lambda x: x @ x,
        x=np.array([-3, 1, 2]),
        radius_options={"initial_radius": 1.0},
    )
    assert res["radius_options"].initial_radius == 1.0


def test_process_batch_size():
    assert _process_batch_size(batch_size=2, n_cores=2) == 2
    assert _process_batch_size(batch_size=None, n_cores=3) == 3


def test_process_batch_size_invalid():
    with pytest.raises(ValueError, match="batch_size must be"):
        _process_batch_size(batch_size=1, n_cores=2)


def test_process_sample_size():
    x = np.arange(3)
    assert _process_sample_size(sample_size=None, model_type="linear", x=x) == 4
    assert _process_sample_size(sample_size=None, model_type="quadratic", x=x) == 7
    assert _process_sample_size(10, None, None) == 10


def test_process_sample_size_callable():
    x = np.arange(3)
    sample_size = lambda x, model_type: len(x) ** 2
    assert _process_sample_size(sample_size=sample_size, model_type="linear", x=x) == 9


def test_process_model_type():
    assert _process_model_type(model_type="linear", functype="scalar") == "linear"
    assert _process_model_type(model_type=None, functype="scalar") == "quadratic"
    assert _process_model_type(model_type=None, functype="least_squares") == "linear"
    assert _process_model_type(model_type=None, functype="likelihood") == "linear"


def test_process_model_type_invalid():
    with pytest.raises(ValueError, match="model_type must be"):
        _process_model_type(model_type="invalid", functype="scalar")


def test_process_search_radius_factor():
    assert _process_search_radius_factor(search_radius_factor=1.1, functype=None) == 1.1
    assert (
        _process_search_radius_factor(search_radius_factor=None, functype="scalar")
        == 4.25
    )
    assert (
        _process_search_radius_factor(
            search_radius_factor=None, functype="least_squares"
        )
        == 5.0
    )


def test_process_search_radius_factor_negative():
    with pytest.raises(ValueError, match="search_radius_factor must be"):
        _process_search_radius_factor(-1, "scalar")


def test_process_acceptance_decider():
    assert _process_acceptance_decider(acceptance_decider=None, noisy=True) == "noisy"
    assert (
        _process_acceptance_decider(acceptance_decider=None, noisy=False) == "classic"
    )
    assert (
        _process_acceptance_decider(acceptance_decider="classic", noisy=None)
        == "classic"
    )


def test_process_model_fitter():
    assert (
        _process_model_fitter(
            model_fitter=None, model_type="quadratic", sample_size=3, x=np.arange(3)
        )
        == "tranquilo"
    )
    assert (
        _process_model_fitter(
            model_fitter=None, model_type="linear", sample_size=4, x=np.arange(3)
        )
        == "ols"
    )
    assert (
        _process_model_fitter(
            model_fitter="xyz", model_type=None, sample_size=None, x=None
        )
        == "xyz"
    )


def test_process_residualize():
    assert _process_residualize(residualize=None, model_fitter="tranquilo") is True
    assert _process_residualize(residualize=None, model_fitter="ols") is False
    assert _process_residualize(residualize=False, model_fitter="custom") is False


def test_process_residualize_invalid():
    with pytest.raises(ValueError, match="residualize must be a boolean."):
        _process_residualize(residualize="invalid", model_fitter=None)


def test_process_n_evals_at_start():
    assert _process_n_evals_at_start(n_evals=None, noisy=True, batch_size=1) == 5
    assert _process_n_evals_at_start(n_evals=None, noisy=False, batch_size=1) == 1
    assert _process_n_evals_at_start(n_evals=10, noisy=None, batch_size=1) == 10
    assert _process_n_evals_at_start(n_evals=None, noisy=True, batch_size=3) == 6
    assert _process_n_evals_at_start(n_evals=None, noisy=False, batch_size=3) == 3
    assert _process_n_evals_at_start(n_evals=10, noisy=None, batch_size=3) == 12


def test_process_n_evals_at_start_negative():
    with pytest.raises(ValueError, match="n_initial_acceptance_evals must be"):
        _process_n_evals_at_start(n_evals=-1, noisy=None, batch_size=1)


def test_roundup_to_next_multiple_of_batch_size():
    assert ceil_to_multiple(1, 1) == 1
    assert ceil_to_multiple(456, 456) == 456
    assert ceil_to_multiple(123123, 1) == 123123
    assert ceil_to_multiple(1, 123123) == 123123
    assert ceil_to_multiple(4, 10) == 10
