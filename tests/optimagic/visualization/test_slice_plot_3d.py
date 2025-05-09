import numpy as np
import plotly.io as pio
import pytest

from optimagic import mark
from optimagic.parameters.bounds import Bounds
from optimagic.visualization.slice_plot_3d import slice_plot_3d

pio.renderers.default = "browser"


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


KWARGS_3D = [
    {},
    {"projection": "contour"},
    {"projection": "surface"},
    {"selector": lambda x: [x["alpha"], x["beta"]]},
    {"param_names": {"alpha": "Alpha", "beta": "Beta"}},
    {"layout_kwargs": {"width": 800, "height": 600, "title": "Custom Layout"}},
    {
        "projection": "surface",
        "plot_kwargs": {"surface_plot": {"colorscale": "Viridis", "opacity": 0.9}},
    },
    {
        "projection": "contour",
        "plot_kwargs": {"contour_plot": {"colorscale": "Cividis", "showscale": True}},
    },
    {
        "selector": lambda x: [x["alpha"], x["beta"], x["gamma"]],
        "make_subplot_kwargs": {"rows": 1, "cols": 3, "horizontal_spacing": 0.1},
    },
    {"n_gridpoints": 100},
    # {"return_dict": True},
    # {"batch_evaluator": "sequential"},
    {
        "layout_kwargs": {
            "template": "plotly_dark",
            "xaxis_showgrid": True,
            "yaxis_showgrid": True,
        }
    },
]

KWARGS = [
    {"projection": "contour"},
    {"projection": "surface"},
    {"selector": lambda x: [x["alpha"], x["beta"]], "projection": "surface"},
    {
        "param_names": {"alpha": "α", "beta": "β"},
        "projection": "contour",
        "layout_kwargs": {"width": 700, "height": 500, "title": "Contour of α vs β"},
    },
    {
        "n_gridpoints": 50,
        "projection": "surface",
        "plot_kwargs": {"surface_plot": {"colorscale": "Inferno", "opacity": 0.7}},
    },
    {"batch_evaluator": "sequential", "return_dict": True, "projection": "contour"},
    {
        "make_subplot_kwargs": {
            "rows": 1,
            "cols": 3,
            "horizontal_spacing": 0.05,
            "start_cell": "bottom-left",
        },
        "layout_kwargs": {"width": 900, "height": 300, "template": "ggplot2"},
        "projection": "slice",
    },
    {
        "expand_yrange": 0.1,
        "plot_kwargs": {
            "line_plot": {"color_discrete_sequence": ["#FF5733"], "markers": True}
        },
        "projection": "slice",
    },
    {
        "plot_kwargs": {
            "contour_plot": {"line_smoothing": 0.4, "colorscale": "Electric"}
        },
        "selector": lambda x: [x["gamma"], x["delta"]],
        "projection": "contour",
    },
    {
        "plot_kwargs": {
            "surface_plot": {"colorscale": "Cividis", "showscale": True, "opacity": 0.6}
        },
        "layout_kwargs": {"template": "plotly_dark"},
        "projection": "surface",
    },
    {
        "layout_kwargs": {
            "xaxis_showgrid": True,
            "yaxis_showgrid": True,
            "width": 650,
            "height": 450,
        },
        "projection": "slice",
    },
    {
        "n_gridpoints": 40,
        "param_names": {"theta": "Θ", "phi": "Φ"},
        "selector": lambda x: [x["theta"], x["phi"]],
        "projection": "surface",
    },
    {
        "projection": "contour",
        "n_gridpoints": 35,
        "plot_kwargs": {"contour_plot": {"colorscale": "Hot", "showscale": True}},
        "layout_kwargs": {"template": "simple_white"},
    },
    {
        "projection": "surface",
        "n_gridpoints": 60,
        "selector": lambda x: [x["x1"], x["x2"]],
        "param_names": {"x1": "X₁", "x2": "X₂"},
        "make_subplot_kwargs": {"rows": 1, "cols": 2},
        "layout_kwargs": {"template": "presentation", "width": 800},
        "plot_kwargs": {
            "surface_plot": {"colorscale": "Blues", "opacity": 0.9, "showscale": True}
        },
        "return_dict": True,
        "batch_evaluator": "sequential",
    },
]
parametrization = [(func, kwargs) for func in [sphere] for kwargs in KWARGS_3D]


@pytest.mark.parametrize("func, kwargs", parametrization)
def test_slice_plot_3d(fixed_inputs, func, kwargs):
    print(func, kwargs)
    fig = slice_plot_3d(
        func=func,
        **fixed_inputs,
        **kwargs,
    )
    fig.show()
