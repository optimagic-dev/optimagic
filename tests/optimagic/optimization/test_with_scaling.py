import numpy as np
from optimagic.optimization.optimize import maximize, minimize
from optimagic.parameters.scaling import ScalingOptions


def test_minimize_with_scaling_options():
    minimize(
        lambda x: x @ x,
        np.arange(3),
        algorithm="scipy_lbfgsb",
        scaling=ScalingOptions(method="start_values", magnitude=1),
    )


def test_minimize_with_scaling_options_dict():
    minimize(
        lambda x: x @ x,
        np.arange(3),
        algorithm="scipy_lbfgsb",
        scaling={"method": "start_values", "magnitude": 1},
    )


def test_minimize_with_default_scaling_options():
    minimize(
        lambda x: x @ x,
        np.arange(3),
        algorithm="scipy_lbfgsb",
        scaling=True,
    )


def test_maximize_with_scaling_options():
    maximize(
        lambda x: -x @ x,
        np.arange(3),
        algorithm="scipy_lbfgsb",
        scaling=ScalingOptions(method="start_values", magnitude=1),
    )


def test_maximize_with_scaling_options_dict():
    maximize(
        lambda x: -x @ x,
        np.arange(3),
        algorithm="scipy_lbfgsb",
        scaling={"method": "start_values", "magnitude": 1},
    )


def test_maximize_with_default_scaling_options():
    maximize(
        lambda x: -x @ x,
        np.arange(3),
        algorithm="scipy_lbfgsb",
        scaling=True,
    )
