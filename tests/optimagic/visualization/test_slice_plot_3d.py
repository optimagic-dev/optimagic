import numpy as np
import pytest

from optimagic import mark
from optimagic.optimization.fun_value import enforce_return_type
from optimagic.parameters.bounds import Bounds
from optimagic.parameters.conversion import get_converter
from optimagic.shared.process_user_function import infer_aggregation_level
from optimagic.visualization.slice_plot_3d import (
    Projection,
    evaluate_function_values,
    generate_evaluation_points,
    slice_plot_3d,
)


@pytest.fixture()
def fixed_inputs():
    params = {"alpha": 0, "beta": 0, "gamma": 0, "delta": 0}
    bounds = Bounds(
        lower={name: -5 for name in params},
        upper={name: i + 2 for i, name in enumerate(params)},
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
        "n_gridpoints": 10,  # Reduced for faster testing if needed
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
    slice_plot_3d(func=func, **fixed_inputs, **kwargs)


kwargs_generate_evaluation_points = [
    (
        sphere,
        5,
        ["alpha"],
        "univariate",
        False,
        [
            [-5.0, 0.0, 0.0, 0.0],
            [-3.25, 0.0, 0.0, 0.0],
            [-1.5, 0.0, 0.0, 0.0],
            [0.25, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
        ],
    ),
    (
        sphere,
        3,
        ["alpha", "gamma"],
        "contour",
        False,
        [
            [-5.0, 0.0, -5.0, 0.0],
            [-1.5, 0.0, -5.0, 0.0],
            [2.0, 0.0, -5.0, 0.0],
            [-5.0, 0.0, -0.5, 0.0],
            [-1.5, 0.0, -0.5, 0.0],
            [2.0, 0.0, -0.5, 0.0],
            [-5.0, 0.0, 4.0, 0.0],
            [-1.5, 0.0, 4.0, 0.0],
            [2.0, 0.0, 4.0, 0.0],
        ],
    ),
    (
        sphere,
        5,
        ["beta"],
        "surface",
        True,
        [
            [0.0, -5.0, 0.0, 0.0],
            [0.0, -3.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
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

    if len(selected_params) == 1:
        selected_params = selected_params[0]

    points = generate_evaluation_points(
        params_data,
        internal_params,
        converter,
        selected_params,
        grid_univariate,
        projection,
    )

    points = [[point[key] for key in internal_params.names] for point in points]
    np.testing.assert_allclose(points, expected_points, rtol=1e-3)


@pytest.mark.parametrize(
    "func, points, param, expected_values",
    [
        (sphere, points, selected_params, expected_values)
        for (_, _, selected_params, _, _, points), expected_values in zip(
            kwargs_generate_evaluation_points,
            [
                [25.0, 10.5625, 2.25, 0.0625, 4.0],
                [50.0, 27.25, 29.0, 25.25, 2.5, 4.25, 41.0, 18.25, 20.0],
                [25.0, 9.0, 1.0, 1.0, 9.0],
            ],
            strict=False,
        )
    ],
)
def test_evaluate_function_values(fixed_inputs, func, points, param, expected_values):
    params = fixed_inputs["params"]
    func_eval = func(params)
    func = enforce_return_type(infer_aggregation_level(func))(func)

    converter, _ = get_converter(
        params=params,
        constraints=None,
        bounds=fixed_inputs["bounds"],
        func_eval=func_eval,
        solver_type="value",
    )

    converted = [converter.params_from_internal(np.array(p)) for p in points]
    result = evaluate_function_values(
        points=converted,
        func=func,
        batch_evaluator="joblib",
        n_cores=1,
    )
    np.testing.assert_allclose(result, expected_values, equal_nan=True)
