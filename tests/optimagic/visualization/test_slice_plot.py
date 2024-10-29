import numpy as np
import pytest

from optimagic import mark
from optimagic.parameters.bounds import Bounds
from optimagic.visualization.slice_plot import slice_plot


@pytest.fixture()
def fixed_inputs():
    params = {"alpha": 0, "beta": 0, "gamma": 0, "delta": 0}
    bounds = Bounds(
        lower={name: -5 for name in params},
        upper={name: i + 2 for i, name in enumerate(params)},
    )

    out = {
        "params": params,
        "bounds": bounds,
    }
    return out


@mark.likelihood
def sphere_loglike(params):
    x = np.array(list(params.values()))
    return x**2


def sphere(params):
    x = np.array(list(params.values()))
    return x @ x


KWARGS = [
    {},
    {"plots_per_row": 4},
    {"selector": lambda x: [x["alpha"], x["beta"]]},
    {"param_names": {"alpha": "Alpha", "beta": "Beta"}},
    {"share_x": True},
    {"share_y": False},
    {"return_dict": True},
]
parametrization = [
    (func, kwargs) for func in [sphere_loglike, sphere] for kwargs in KWARGS
]


@pytest.mark.parametrize("func, kwargs", parametrization)
def test_slice_plot(fixed_inputs, func, kwargs):
    slice_plot(
        func=func,
        **fixed_inputs,
        **kwargs,
    )
