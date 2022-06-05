import numpy as np
import pytest
from estimagic.visualization.slice_plot import slice_plot


@pytest.fixture
def fixed_inputs():
    def sphere(params):
        x = np.array(list(params.values()))
        return x @ x

    params = {"alpha": 0, "beta": 0, "gamma": 0, "delta": 0}
    lower_bounds = {name: -5 for name in params}
    upper_bounds = {name: i + 2 for i, name in enumerate(params)}

    out = {
        "func": sphere,
        "params": params,
        "lower_bounds": lower_bounds,
        "upper_bounds": upper_bounds,
    }
    return out


KWARGS = [
    {},
    {"plots_per_row": 4},
    {"selector": lambda x: [x["alpha"], x["beta"]]},
    {"param_names": {"alpha": "Alpha", "beta": "Beta"}},
    {"share_x": True},
    {"share_y": False},
    {"return_dict": True},
]


@pytest.mark.parametrize("kwargs", KWARGS)
def test_slice_plot(fixed_inputs, kwargs):

    slice_plot(
        **fixed_inputs,
        **kwargs,
    )
