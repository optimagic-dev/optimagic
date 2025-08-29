"""Test helper functions in PySwarms optimizers."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from optimagic.config import IS_PYSWARMS_INSTALLED
from optimagic.optimization.internal_optimization_problem import InternalBounds
from optimagic.optimizers.pyswarms_optimizers import (
    PSOOptions,
    PyramidTopology,
    RandomTopology,
    RingTopology,
    StarTopology,
    VonNeumannTopology,
    _build_velocity_clamp,
    _create_initial_positions,
    _get_pyswarms_bounds,
    _pso_options_to_dict,
    _resolve_topology_config,
)

RNG = np.random.default_rng(12345)


# Test _pso_options_to_dict
def test_pso_options_to_dict_default():
    """Test PSO options conversion with default values."""
    options = PSOOptions()
    result = _pso_options_to_dict(options)

    expected = {
        "c1": 0.5,
        "c2": 0.3,
        "w": 0.9,
    }
    assert result == expected


def test_pso_options_to_dict_custom():
    """Test PSO options conversion with custom values."""
    options = PSOOptions(
        cognitive_parameter=1.5,
        social_parameter=2.0,
        inertia_weight=0.7,
    )
    result = _pso_options_to_dict(options)

    expected = {
        "c1": 1.5,
        "c2": 2.0,
        "w": 0.7,
    }
    assert result == expected


# Test _build_velocity_clamp
def test_build_velocity_clamp_both_values():
    """Test velocity clamp with both min and max values."""
    result = _build_velocity_clamp(-1.0, 1.0)
    assert result == (-1.0, 1.0)


def test_build_velocity_clamp_partial_values():
    """Test velocity clamp with only one value provided."""
    result = _build_velocity_clamp(-1.0, None)
    assert result is None

    result = _build_velocity_clamp(None, 1.0)
    assert result is None


def test_build_velocity_clamp_none_values():
    """Test velocity clamp with None values."""
    result = _build_velocity_clamp(None, None)
    assert result is None


# Test _get_pyswarms_bounds
def test_get_pyswarms_bounds_with_both():
    """Test bounds conversion when both lower and upper bounds are provided."""
    bounds = InternalBounds(lower=np.array([-2.0, -3.0]), upper=np.array([5.0, 4.0]))

    result = _get_pyswarms_bounds(bounds)

    assert result is not None
    lower, upper = result
    assert_array_equal(lower, np.array([-2.0, -3.0]))
    assert_array_equal(upper, np.array([5.0, 4.0]))


def test_get_pyswarms_bounds_with_none():
    """Test bounds conversion when no bounds are provided."""
    bounds = InternalBounds(lower=None, upper=None)

    result = _get_pyswarms_bounds(bounds)
    assert result is None


def test_get_pyswarms_bounds_partial_bounds():
    """Test bounds conversion with only one bound provided."""
    # Only lower bounds
    bounds = InternalBounds(lower=np.array([1.0, 2.0]), upper=None)
    result = _get_pyswarms_bounds(bounds)
    assert result is None

    # Only upper bounds
    bounds = InternalBounds(lower=None, upper=np.array([3.0, 4.0]))
    result = _get_pyswarms_bounds(bounds)
    assert result is None


def test_get_pyswarms_bounds_with_infinite():
    """Test that infinite bounds raise ValueError."""
    bounds = InternalBounds(
        lower=np.array([-np.inf, -1.0]), upper=np.array([1.0, np.inf])
    )

    with pytest.raises(ValueError, match="PySwarms does not support infinite bounds"):
        _get_pyswarms_bounds(bounds)


# Test _create_initial_positions
@pytest.mark.parametrize("center", [0.5, 1.0, 2.0])
def test_create_initial_positions_basic(center):
    """Test basic initial positions creation."""
    x0 = np.array([1.0, 2.0])
    n_particles = 5
    bounds = (np.array([-5.0, -5.0]), np.array([5.0, 5.0]))

    init_pos = _create_initial_positions(
        x0=x0, n_particles=n_particles, bounds=bounds, center=center, rng=RNG
    )

    assert init_pos.shape == (5, 2)

    assert_array_equal(init_pos[0], x0)

    # Check all particles are within bounds
    assert np.all(init_pos >= bounds[0])
    assert np.all(init_pos <= bounds[1])


def test_create_initial_positions_no_bounds():
    """Test initial positions creation with no bounds."""
    x0 = np.array([0.5, 1.5])
    n_particles = 3
    bounds = None

    init_pos = _create_initial_positions(
        x0=x0, n_particles=n_particles, bounds=bounds, center=1.0, rng=RNG
    )

    assert init_pos.shape == (3, 2)

    expected_x0 = np.array([0.5, 1.0])
    assert_array_equal(init_pos[0], expected_x0)

    assert np.all(init_pos >= 0.0)
    assert np.all(init_pos <= 1.0)


@pytest.mark.skipif(not IS_PYSWARMS_INSTALLED, reason="PySwarms not installed")
@pytest.mark.parametrize(
    ("topology_string", "expected_class_name", "expected_options"),
    [
        ("star", "Star", {}),
        ("ring", "Ring", {"k": 3, "p": 2}),
        ("vonneumann", "VonNeumann", {"p": 2, "r": 1}),
        ("random", "Random", {"k": 3}),
        ("pyramid", "Pyramid", {}),
    ],
)
def test_resolve_topology_config_by_string(
    topology_string, expected_class_name, expected_options
):
    """Test topology resolution with string names."""
    topology, options = _resolve_topology_config(topology_string)

    assert topology.__class__.__name__ == expected_class_name
    assert options == expected_options


@pytest.mark.skipif(not IS_PYSWARMS_INSTALLED, reason="PySwarms not installed")
@pytest.mark.parametrize(
    ("config_instance", "expected_class_name", "expected_options"),
    [
        (StarTopology(), "Star", {}),
        (RingTopology(k_neighbors=5, p_norm=1, static=True), "Ring", {"k": 5, "p": 1}),
        (
            VonNeumannTopology(p_norm=1, range_param=2),
            "VonNeumann",
            {"p": 1, "r": 2},
        ),
        (RandomTopology(k_neighbors=4, static=False), "Random", {"k": 4}),
        (PyramidTopology(static=True), "Pyramid", {}),
    ],
)
def test_resolve_topology_config_by_instance(
    config_instance, expected_class_name, expected_options
):
    """Test topology resolution with instances."""
    topology, options = _resolve_topology_config(config_instance)

    # Check the class name and options
    assert topology.__class__.__name__ == expected_class_name
    assert options == expected_options

    if hasattr(config_instance, "static"):
        assert topology.static == config_instance.static


@pytest.mark.skipif(not IS_PYSWARMS_INSTALLED, reason="PySwarms not installed")
def test_resolve_topology_config_invalid_string():
    """Test topology resolution with invalid string."""
    with pytest.raises(ValueError, match="Unknown topology string: 'invalid'"):
        _resolve_topology_config("invalid")


@pytest.mark.skipif(not IS_PYSWARMS_INSTALLED, reason="PySwarms not installed")
def test_resolve_topology_config_invalid_type():
    """Test topology resolution with invalid type."""
    with pytest.raises(TypeError, match="Unsupported topology configuration type"):
        _resolve_topology_config(123)
