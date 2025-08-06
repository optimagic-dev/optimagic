import numpy as np
import pytest

from optimagic import mark
from optimagic.parameters.bounds import Bounds
from optimagic.parameters.conversion import get_converter
from optimagic.visualization.slice_plot_3d import (
    Projection,
    generate_evaluation_points,
    plot_data_cache,
    slice_plot_3d,
)


@pytest.fixture()
def fixed_inputs():
    params = {"alpha": 0, "beta": 0, "gamma": 0, "delta": 0}
    bounds = Bounds(
        lower={name: -5 for name in params},
        upper={name: i for i, name in enumerate(params)},
    )
    return {"params": params, "bounds": bounds}


@mark.likelihood
def sphere_loglike(params):
    x = np.array(list(params.values()))
    return x**2


def sphere(params):
    x = np.array(list(params.values()))
    return x @ x


kwargs_slice_plot_3d = [
    {},
    {"projection": "contour"},
    {"projection": "surface"},
    {"projection": "surface", "n_gridpoints": 100},
    {"projection": {"lower": "contour", "upper": "contour"}},
    {"projection": {"lower": "surface", "upper": "contour"}},
    {
        "projection": {"lower": "contour", "upper": "surface"},
        "selector": lambda x: [x["alpha"], x["beta"], x["delta"]],
    },
    {"selector": lambda x: [x["alpha"], x["beta"]]},
    {"param_names": {"alpha": "Alpha", "beta": "Beta"}},
    {"layout_kwargs": {"width": 800, "height": 600, "title": "Custom Layout"}},
    {
        "projection": "surface",
        "selector": lambda x: [x["alpha"], x["gamma"]],
    },
    {
        "projection": "contour",
        "selector": lambda x: [x["alpha"], x["delta"]],
    },
    {
        "projection": "surface",
        "plot_kwargs": {"surface_plot": {"colorscale": "Viridis", "opacity": 0.9}},
    },
    {
        "projection": "contour",
        "plot_kwargs": {"contour_plot": {"colorscale": "Viridis", "showscale": True}},
    },
    {
        "selector": lambda x: [x["alpha"], x["beta"], x["gamma"]],
        "make_subplot_kwargs": {"rows": 1, "cols": 3, "horizontal_spacing": 0.01},
    },
    {
        "param_names": {"alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ"},
        "n_gridpoints": 10,
        "expand_yrange": 2,
    },
    {
        "layout_kwargs": {
            "template": "plotly_dark",
            "xaxis_showgrid": True,
            "yaxis_showgrid": True,
        }
    },
    {
        "plot_kwargs": {
            "scatter_plot": None,
            "line_plot": {"color_discrete_sequence": ["red"], "markers": True},
        }
    },
    {"return_dict": True},
    {
        "return_dict": True,
        "layout_kwargs": {
            "template": "plotly_dark",
            "xaxis_showgrid": True,
            "yaxis_showgrid": True,
        },
        "plot_kwargs": {
            "scatter_plot": None,
            "line_plot": {"color_discrete_sequence": ["red"], "markers": True},
        },
    },
]

parametrized_slice_plot_3d = [
    (func, kwarg) for func in [sphere, sphere_loglike] for kwarg in kwargs_slice_plot_3d
]


@pytest.mark.parametrize("func, kwargs", parametrized_slice_plot_3d)
def test_slice_plot_3d(fixed_inputs, func, kwargs):
    fig = slice_plot_3d(func=func, **fixed_inputs, **kwargs)
    if isinstance(fig, dict):
        print(fig)
    else:
        fig.show()


kwargs_generate_evaluation_points = [
    (
        sphere,
        5,
        ["alpha"],
        "univariate",
        False,
        [
            [-5.0, 0.0, 0.0, 0.0],
            [-3.75, 0.0, 0.0, 0.0],
            [-2.5, 0.0, 0.0, 0.0],
            [-1.25, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
    ),
    (
        sphere,
        3,
        ["alpha", "gamma"],
        "contour",
        False,
        [
            [-5.0, 0.0, 0.0, 0.0],
            [-2.5, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -5.0, 0.0],
            [0.0, 0.0, -1.5, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [-5.0, 0.0, -5.0, 0.0],
            [-2.5, 0.0, -5.0, 0.0],
            [0.0, 0.0, -5.0, 0.0],
            [-5.0, 0.0, -1.5, 0.0],
            [-2.5, 0.0, -1.5, 0.0],
            [0.0, 0.0, -1.5, 0.0],
            [-5.0, 0.0, 2.0, 0.0],
            [-2.5, 0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [-5.0, 0.0, -5.0, 0.0],
            [-5.0, 0.0, -1.5, 0.0],
            [-5.0, 0.0, 2.0, 0.0],
            [-2.5, 0.0, -5.0, 0.0],
            [-2.5, 0.0, -1.5, 0.0],
            [-2.5, 0.0, 2.0, 0.0],
            [0.0, 0.0, -5.0, 0.0],
            [0.0, 0.0, -1.5, 0.0],
            [0.0, 0.0, 2.0, 0.0],
        ],
    ),
    (
        sphere,
        5,
        ["beta", "delta"],
        "surface",
        True,
        [
            [0.0, -5.0, 0.0, 0.0],
            [0.0, -3.5, 0.0, 0.0],
            [0.0, -2.0, 0.0, 0.0],
            [0.0, -0.5, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, -5.0],
            [0.0, 0.0, 0.0, -3.0],
            [0.0, 0.0, 0.0, -1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 3.0],
            [0.0, -5.0, 0.0, -5.0],
            [0.0, -3.5, 0.0, -5.0],
            [0.0, -2.0, 0.0, -5.0],
            [0.0, -0.5, 0.0, -5.0],
            [0.0, 1.0, 0.0, -5.0],
            [0.0, -5.0, 0.0, -3.0],
            [0.0, -3.5, 0.0, -3.0],
            [0.0, -2.0, 0.0, -3.0],
            [0.0, -0.5, 0.0, -3.0],
            [0.0, 1.0, 0.0, -3.0],
            [0.0, -5.0, 0.0, -1.0],
            [0.0, -3.5, 0.0, -1.0],
            [0.0, -2.0, 0.0, -1.0],
            [0.0, -0.5, 0.0, -1.0],
            [0.0, 1.0, 0.0, -1.0],
            [0.0, -5.0, 0.0, 1.0],
            [0.0, -3.5, 0.0, 1.0],
            [0.0, -2.0, 0.0, 1.0],
            [0.0, -0.5, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, -5.0, 0.0, 3.0],
            [0.0, -3.5, 0.0, 3.0],
            [0.0, -2.0, 0.0, 3.0],
            [0.0, -0.5, 0.0, 3.0],
            [0.0, 1.0, 0.0, 3.0],
            [0.0, -5.0, 0.0, -5.0],
            [0.0, -5.0, 0.0, -3.0],
            [0.0, -5.0, 0.0, -1.0],
            [0.0, -5.0, 0.0, 1.0],
            [0.0, -5.0, 0.0, 3.0],
            [0.0, -3.5, 0.0, -5.0],
            [0.0, -3.5, 0.0, -3.0],
            [0.0, -3.5, 0.0, -1.0],
            [0.0, -3.5, 0.0, 1.0],
            [0.0, -3.5, 0.0, 3.0],
            [0.0, -2.0, 0.0, -5.0],
            [0.0, -2.0, 0.0, -3.0],
            [0.0, -2.0, 0.0, -1.0],
            [0.0, -2.0, 0.0, 1.0],
            [0.0, -2.0, 0.0, 3.0],
            [0.0, -0.5, 0.0, -5.0],
            [0.0, -0.5, 0.0, -3.0],
            [0.0, -0.5, 0.0, -1.0],
            [0.0, -0.5, 0.0, 1.0],
            [0.0, -0.5, 0.0, 3.0],
            [0.0, 1.0, 0.0, -5.0],
            [0.0, 1.0, 0.0, -3.0],
            [0.0, 1.0, 0.0, -1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 3.0],
        ],
    ),
]


@pytest.mark.parametrize(
    "func, n_points, selected_params, projection, grid_univariate, expected_points",
    kwargs_generate_evaluation_points,
)
def test_generate_evaluation_points(
    fixed_inputs,
    func,
    n_points,
    selected_params,
    projection,
    grid_univariate,
    expected_points,
):
    projection = Projection(projection)
    params = fixed_inputs["params"]
    func_eval = func(params)

    converter, internal_params = get_converter(
        params=params,
        constraints=None,
        bounds=fixed_inputs["bounds"],
        func_eval=func_eval,
        solver_type="value",
    )

    params_data = {
        name: np.linspace(
            internal_params.lower_bounds[internal_params.names.index(name)],
            internal_params.upper_bounds[internal_params.names.index(name)],
            n_points,
        )
        for name in selected_params
    }

    selected_indices = [list(params.keys()).index(param) for param in selected_params]
    points = generate_evaluation_points(
        projection,
        selected_indices,
        internal_params,
        params_data,
        converter,
    )

    points = [[point[key] for key in internal_params.names] for point in points]
    print([[float(va) for d in points for va in d]])
    np.testing.assert_allclose(points, expected_points, rtol=0.2)


kwargs_plot_data_cache = [
    (
        sphere,
        5,
        [0],
        "univariate",
        [25, 14.0, 6.25, 1.5, 0],
        {("alpha",): [25, 14.0, 6.25, 1.5, 0]},
    ),
    (
        sphere,
        3,
        [0, 2],
        "contour",
        [
            25,
            6.25,
            0,
            25,
            2.25,
            4,
            50,
            31.25,
            25,
            27.25,
            8.5,
            2.25,
            29,
            10.25,
            4,
            50,
            27.25,
            29,
            31.25,
            8.5,
            10.25,
            25,
            2.25,
            4,
        ],
        {
            ("alpha",): [25, 6.25, 0],
            ("gamma",): [25, 2.25, 4],
            ("alpha", "gamma"): [50, 27.25, 29, 31.25, 8.5, 10.25, 25, 2.25, 4],
        },
    ),
]


@pytest.mark.parametrize(
    "func, n_points, selected_indices, projection, func_values, expected_values",
    kwargs_plot_data_cache,
)
def test_evaluate_function_values(
    fixed_inputs,
    func,
    n_points,
    projection,
    selected_indices,
    func_values,
    expected_values,
):
    projection = Projection(projection)

    params = fixed_inputs["params"]
    func_eval = func(params)

    converter, internal_params = get_converter(
        params=params,
        constraints=None,
        bounds=fixed_inputs["bounds"],
        func_eval=func_eval,
        solver_type="value",
    )
    plot_data = plot_data_cache(
        projection, selected_indices, internal_params, func_values, n_points
    )
    assert plot_data == expected_values
