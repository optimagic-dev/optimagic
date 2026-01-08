import numpy as np
import pytest

from optimagic import mark
from optimagic.parameters.bounds import Bounds
from optimagic.visualization.backends import BACKEND_AVAILABILITY_AND_LINE_PLOT_FUNCTION
from optimagic.visualization.plotting_utilities import LineData, MarkerData
from optimagic.visualization.slice_plot import (
    _extract_slice_plot_lines_and_labels,
    _get_plot_data,
    _get_processed_func_and_func_eval,
    slice_plot,
)


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
    {"title": "Slice Plot"},
]
parametrization = [
    (func, kwargs) for func in [sphere_loglike, sphere] for kwargs in KWARGS
]


@pytest.mark.parametrize("backend", BACKEND_AVAILABILITY_AND_LINE_PLOT_FUNCTION.keys())
@pytest.mark.parametrize("func, kwargs", parametrization)
def test_slice_plot(fixed_inputs, func, backend, kwargs, close_mpl_figures):
    slice_plot(
        func=func,
        backend=backend,
        **fixed_inputs,
        **kwargs,
    )


def test_extract_slice_plot_lines(fixed_inputs):
    params, bounds = fixed_inputs["params"], fixed_inputs["bounds"]

    func, func_eval = _get_processed_func_and_func_eval(
        sphere, func_kwargs=None, params=params
    )

    plot_data, internal_params = _get_plot_data(
        func=func,
        params=params,
        bounds=bounds,
        func_eval=func_eval,
        selector=None,
        n_gridpoints=10,
        batch_evaluator="joblib",
        n_cores=1,
    )

    lines_list, marker_list, xlabels, ylabels = _extract_slice_plot_lines_and_labels(
        plot_data=plot_data,
        internal_params=internal_params,
        func_eval=func_eval,
        param_names={"alpha": "Alpha"},
        color=None,
    )

    assert isinstance(lines_list, list) and len(lines_list) == len(params)
    assert all(
        isinstance(subplot_lines, list)
        and len(subplot_lines) == 1
        and isinstance(subplot_lines[0], LineData)
        for subplot_lines in lines_list
    )

    assert isinstance(marker_list, list) and len(marker_list) == len(params)
    assert all(isinstance(marker, MarkerData) for marker in marker_list)
    for i, k in enumerate(params):
        assert marker_list[i].x == params[k]

    assert isinstance(xlabels, list)
    assert xlabels == ["Alpha", "beta", "gamma", "delta"]

    assert isinstance(ylabels, list)
    assert all(ylabel == "Function Value" for ylabel in ylabels)
