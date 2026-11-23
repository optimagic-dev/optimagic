"""Unit tests for Bayesian optimizer helper functions."""

import numpy as np
import pytest

from optimagic.config import IS_BAYESOPT_INSTALLED_AND_VERSION_NEWER_THAN_2
from optimagic.optimization.internal_optimization_problem import InternalBounds

pytestmark = [
    pytest.mark.bayesopt,
    pytest.mark.skipif(
        not IS_BAYESOPT_INSTALLED_AND_VERSION_NEWER_THAN_2,
        reason="bayes_opt is not installed in a recent enough version >= 2.0.0.",
    ),
]

if IS_BAYESOPT_INSTALLED_AND_VERSION_NEWER_THAN_2:
    from bayes_opt import acquisition

    from optimagic.optimizers.bayesian_optimizer import (
        _extract_params_from_kwargs,
        _process_acquisition_function,
        _process_bounds,
    )


def test_extract_params_from_kwargs():
    """Test basic parameter extraction from kwargs dictionary."""
    params_dict = {"param0": 1.0, "param1": 2.0, "param2": 3.0}
    result = _extract_params_from_kwargs(params_dict)
    np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0]))


def test_process_bounds_valid():
    """Test processing valid bounds for Bayesian optimization."""
    bounds = InternalBounds(lower=np.array([-1.0, 0.0]), upper=np.array([1.0, 2.0]))
    result = _process_bounds(bounds)
    expected = {"param0": (-1.0, 1.0), "param1": (0.0, 2.0)}
    assert result == expected


def test_process_bounds_none():
    """Test processing bounds with None values."""
    bounds = InternalBounds(lower=None, upper=np.array([1.0, 2.0]))
    with pytest.raises(
        ValueError, match="Bayesian optimization requires finite bounds"
    ):
        _process_bounds(bounds)


def test_process_bounds_infinite():
    """Test processing bounds with infinite values."""
    bounds = InternalBounds(lower=np.array([-1.0, 0.0]), upper=np.array([1.0, np.inf]))
    with pytest.raises(
        ValueError, match="Bayesian optimization requires finite bounds"
    ):
        _process_bounds(bounds)


def test_process_acquisition_function_none():
    """Test processing None acquisition function."""
    result = _process_acquisition_function(
        acquisition_function=None,
        kappa=2.576,
        xi=0.01,
        exploration_decay=None,
        exploration_decay_delay=None,
        random_seed=None,
    )
    assert result is None


@pytest.mark.parametrize(
    "acq_name, expected_class_name",
    [
        ("ucb", "UpperConfidenceBound"),
        ("upper_confidence_bound", "UpperConfidenceBound"),
        ("ei", "ExpectedImprovement"),
        ("expected_improvement", "ExpectedImprovement"),
        ("poi", "ProbabilityOfImprovement"),
        ("probability_of_improvement", "ProbabilityOfImprovement"),
    ],
)
def test_process_acquisition_function_string(acq_name, expected_class_name):
    """Test processing string acquisition function."""
    expected_class = getattr(acquisition, expected_class_name)
    result = _process_acquisition_function(
        acquisition_function=acq_name,
        kappa=2.576,
        xi=0.01,
        exploration_decay=None,
        exploration_decay_delay=None,
        random_seed=None,
    )
    assert isinstance(result, expected_class)


def test_process_acquisition_function_invalid_string():
    """Test processing invalid string acquisition function."""
    with pytest.raises(ValueError, match="Invalid acquisition_function string"):
        _process_acquisition_function(
            acquisition_function="acq",
            kappa=2.576,
            xi=0.01,
            exploration_decay=None,
            exploration_decay_delay=None,
            random_seed=None,
        )


def test_process_acquisition_function_instance():
    """Test processing acquisition function instance."""
    acq_instance = acquisition.UpperConfidenceBound()
    result = _process_acquisition_function(
        acquisition_function=acq_instance,
        kappa=2.576,
        xi=0.01,
        exploration_decay=None,
        exploration_decay_delay=None,
        random_seed=None,
    )
    assert result is acq_instance


def test_process_acquisition_function_class():
    """Test processing acquisition function class."""
    result = _process_acquisition_function(
        acquisition_function=acquisition.UpperConfidenceBound,
        kappa=2.576,
        xi=0.01,
        exploration_decay=None,
        exploration_decay_delay=None,
        random_seed=None,
    )
    assert isinstance(result, acquisition.UpperConfidenceBound)


def test_process_acquisition_function_invalid_type():
    """Test processing invalid acquisition function type."""
    with pytest.raises(TypeError, match="acquisition_function must be None, a string"):
        _process_acquisition_function(
            acquisition_function=123,  # ty:ignore[invalid-argument-type]
            kappa=2.576,
            xi=0.01,
            exploration_decay=None,
            exploration_decay_delay=None,
            random_seed=None,
        )
