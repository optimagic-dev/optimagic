import numpy as np
import pytest
from estimagic.optimization.tranquilo.adjust_radius import adjust_radius
from estimagic.optimization.tranquilo.options import RadiusOptions


@pytest.fixture()
def options():
    return RadiusOptions(initial_radius=0.1)


def test_increase(options):
    calculated = adjust_radius(
        radius=1,
        rho=1.5,
        step_length=np.linalg.norm(np.ones(2)),
        options=options,
    )

    expected = 2

    assert calculated == expected


def test_increase_blocked_by_small_step(options):
    calculated = adjust_radius(
        radius=1,
        rho=1.5,
        step_length=np.linalg.norm(np.array([0.1, 0.1])),
        options=options,
    )

    expected = 1

    assert calculated == expected


def test_decrease(options):
    calculated = adjust_radius(
        radius=1,
        rho=0.05,
        step_length=np.linalg.norm(np.ones(2)),
        options=options,
    )

    expected = 0.5

    assert calculated == expected


def test_max_radius_is_not_violated(options):
    calculated = adjust_radius(
        radius=750_000,
        rho=1.5,
        step_length=np.linalg.norm(np.array([750_000])),
        options=options,
    )

    expected = 1e6

    assert calculated == expected


def test_min_radius_is_not_violated(options):
    calculated = adjust_radius(
        radius=1e-09,
        rho=0.05,
        step_length=np.linalg.norm(np.ones(2)),
        options=options,
    )

    expected = 1e-06

    assert calculated == expected


def test_constant_radius():
    options = RadiusOptions(rho_increase=1.6, initial_radius=0.1)

    calculated = adjust_radius(
        radius=1,
        rho=1.5,
        step_length=np.linalg.norm(np.ones(2)),
        options=options,
    )

    expected = 1

    assert calculated == expected


def test_max_radius_to_step_ratio_is_not_violated():
    options = RadiusOptions(max_radius_to_step_ratio=2, initial_radius=0.1)

    calculated = adjust_radius(
        radius=1,
        rho=1.5,
        step_length=np.linalg.norm(np.array([0.75])),
        options=options,
    )

    expected = 1.5

    assert calculated == expected
