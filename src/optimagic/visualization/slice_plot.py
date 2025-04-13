# mypy: disable-error-code="attr-defined"

import numpy as np
import plotly.express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from optimagic.config import DEFAULT_N_CORES, PLOTLY_TEMPLATE
from optimagic.parameters.conversion import get_converter
from optimagic.typing import AggregationLevel
from optimagic.visualization.plot_data import (
    evaluate_func,
    evaluate_function_values,
    generate_eval_points,
    generate_grid_data,
    process_bounds,
    select_parameter_indices,
)
from optimagic.visualization.plotting_utilities import combine_plots, get_layout_kwargs


def _plot_slice(
    selected,
    internal_params,
    converter,
    grid_data,
    func,
    func_eval,
    batch_evaluator,
    n_cores,
    param_names,
    expand_yrange,
    color,
    template,
    title,
    plots_per_row,
    share_y,
    share_x,
    return_dict,
    make_subplot_kwargs,
):
    """Slice plot implementation."""

    plots_dict = {}
    title_kwargs = {"text": title} if title else None
    layout_kwargs = get_layout_kwargs(None, None, title_kwargs, template, False)

    for idx in selected:
        param_name = internal_params.names[idx]
        fixed_vars = {
            name: internal_params.values[internal_params.names.index(name)]
            for name in internal_params.names
            if name != param_name
        }
        x, evaluation_points = generate_eval_points(
            grid_data, internal_params, param_name, fixed_vars, converter, "slice"
        )
        func_values = evaluate_function_values(
            func, evaluation_points, batch_evaluator, n_cores
        )

        y_min, y_max = np.min(func_values), np.max(func_values)
        y_range = y_max - y_min
        yaxis_range = [y_min - y_range * expand_yrange, y_max + y_range * expand_yrange]

        display_name = (
            param_names.get(param_name, param_name) if param_names else param_name
        )
        fig = px.line(
            x=grid_data[param_name].values,
            y=func_values,
            color_discrete_sequence=[color],
        )
        fig.add_trace(
            go.Scatter(
                x=[internal_params.values[idx]],
                y=[func_eval.internal_value(AggregationLevel.SCALAR)],
                marker={"color": color},
            )
        )
        fig.update_layout(**layout_kwargs)
        fig.update_xaxes(title={"text": display_name})
        fig.update_yaxes(
            title={"text": "Function Value"}, range=yaxis_range if share_y else None
        )
        plots_dict[display_name] = fig

    if return_dict:
        return plots_dict
    return combine_plots(
        plots=list(plots_dict.values()),
        plots_per_row=plots_per_row,
        sharex=share_x,
        sharey=share_y,
        share_yrange_all=share_y,
        share_xrange_all=share_x,
        expand_yrange=expand_yrange,
        make_subplot_kwargs=make_subplot_kwargs,
        showlegend=False,
        template=template,
        clean_legend=True,
        layout_kwargs=layout_kwargs,
        legend_kwargs={},
        title_kwargs=title_kwargs,
    )


def _plot_pairwise(
    selected,
    internal_params,
    converter,
    grid_data,
    func,
    batch_evaluator,
    n_cores,
    projection,
    title,
    template,
    n_gridpoints,
    projection_kwargs,
):
    """Projection plot implementation."""
    selected_param_names = [internal_params.names[i] for i in selected]

    # Extract plotting params from projection_kwargs
    width = projection_kwargs.get("width", 700)
    height = projection_kwargs.get("height", 600)
    scene_camera_eye = projection_kwargs.get("scene_camera_eye", dict(x=1, y=2, z=0.5))
    colormap = projection_kwargs.get("colormap", "Viridis")
    showscale = projection_kwargs.get("showscale", True)
    line_smoothing = projection_kwargs.get("line_smoothing", 0.85)
    diagonal_show = projection_kwargs.get("diagonal_show", False)

    len_selected = len(selected)
    if len_selected == 2:
        name_x, name_y = selected_param_names
        selected_names = [name_x, name_y]

        fixed_vars = {
            name: grid_data.iloc[0][name]
            for name in internal_params.names
            if name not in selected_names
        }

        x, y, evaluation_points = generate_eval_points(
            grid_data,
            internal_params,
            selected_names,
            fixed_vars,
            converter,
            projection,
        )
        func_values = evaluate_function_values(
            func, evaluation_points, batch_evaluator, n_cores
        )
        z = np.reshape(func_values, (n_gridpoints, n_gridpoints))

        if projection == "3d":
            trace = go.Surface(
                z=z,
                x=x,
                y=y,
                colorscale=colormap,
                showscale=showscale,
            )
        else:
            trace = go.Contour(
                z=z,
                x=x[0],
                y=y[:, 0],
                contours_coloring="heatmap",
                line_smoothing=line_smoothing,
                colorscale=colormap,
                showscale=showscale,
            )

        fig = go.Figure(data=[trace])
        layout_kwargs = dict(
            title=title or f"{projection} plot: {name_x} vs {name_y}",
            template=template,
            width=width,
            height=height,
        )
        if projection == "3d":
            layout_kwargs["scene"] = dict(
                xaxis_title=name_x,
                yaxis_title=name_y,
                zaxis_title="Function Value",
                camera=dict(eye=scene_camera_eye),
            )
        fig.update_layout(**layout_kwargs)
        return fig
    else:
        # Pairwise plot for multiple plots ("3d" or "Contour" projections)
        specs = [
            [
                {"type": "scene"} if projection == "3d" else {"type": "xy"}
                for _ in range(len_selected)
            ]
            for _ in range(len_selected)
        ]
        subplot_titles = [
            f"{selected_param_names[j]} vs {selected_param_names[i]}" if i != j else ""
            for i in range(len_selected)
            for j in range(len_selected)
        ]
        fig = make_subplots(
            rows=len_selected,
            cols=len_selected,
            specs=specs,
            subplot_titles=subplot_titles,
        )

        for i, name_i in enumerate(selected_param_names):
            for j, name_j in enumerate(selected_param_names):
                if i == j and not diagonal_show:
                    continue

                selected_names = [name_j, name_i]
                fixed_vars = {
                    name: grid_data.iloc[0][name]
                    for name in internal_params.names
                    if name not in selected_names
                }

                x, y, evaluation_points = generate_eval_points(
                    grid_data,
                    internal_params,
                    selected_names,
                    fixed_vars,
                    converter,
                    projection,
                )
                func_values = evaluate_function_values(
                    func, evaluation_points, batch_evaluator, n_cores
                )
                z = np.reshape(func_values, (n_gridpoints, n_gridpoints))

                if projection == "3d":
                    trace = go.Surface(
                        z=z, x=x, y=y, showscale=showscale, colorscale=colormap
                    )
                else:
                    trace = go.Contour(
                        z=z,
                        x=x[0],
                        y=y[:, 0],
                        contours_coloring="heatmap",
                        line_smoothing=line_smoothing,
                        showscale=showscale,
                        colorscale=colormap,
                    )

                fig.add_trace(trace, row=i + 1, col=j + 1)

        layout_kwargs = dict(
            title=title or f"{projection} plot",
            template=template,
            width=width,
            height=height,
        )

        # update layout kwargs
        if projection == "3d":
            eye_layouts = {}
            for i in range(1, (len_selected * len_selected) + 1):
                scene_id = "scene" if i == 1 else f"scene{i}"
                eye_layouts[f"{scene_id}_camera"] = dict(eye=scene_camera_eye)

            layout_kwargs = dict(**layout_kwargs, **eye_layouts)

        fig.update_layout(**layout_kwargs)
        return fig


def slice_plot(
    func,
    params,
    bounds=None,
    func_kwargs=None,
    selector=None,
    n_cores=DEFAULT_N_CORES,
    n_gridpoints=20,
    plots_per_row=2,
    param_names=None,
    share_y=True,
    expand_yrange=0.02,
    share_x=False,
    color="#497ea7",
    template=PLOTLY_TEMPLATE,
    title=None,
    return_dict=False,
    make_subplot_kwargs=None,
    batch_evaluator="joblib",
    projection="slice",
    projection_kwargs=None,
    # deprecated
    lower_bounds=None,
    upper_bounds=None,
):
    """Plot criterion along coordinates at given and random values.

    Generates slice or pairwise plots for selected parameters using 1D slices or 2D
    projections such as contour and 3D surface plots. Optionally combines individual
    plots into a subplot layout or returns them separately.

    # TODO: Use soft bounds to create the grid (if available).
    # TODO: Don't do a function evaluation outside the batch evaluator.

    Args:
        func (callable): Criterion function that takes params and returns a scalar,
            PyTree or FunctionValue object.
        params (pytree): A pytree with parameters.
        bounds: Lower and upper bounds on the parameters. The bounds are used to create
            a grid over which slice plots are drawn. The most general and preferred
            way to specify bounds is an `optimagic.Bounds` object that collects lower,
            upper, soft_lower and soft_upper bounds. The soft bounds are not used for
            slice_plots. Each bound type mirrors the structure of params. Check our
            how-to guide on bounds for examples. If params is a flat numpy array, you
            can also provide bounds via any format that is supported by
            scipy.optimize.minimize.
        func_kwargs (dict or NoneType): Optional dictionary of additional arguments
            passed to the criterion function.
        selector (callable): Function that takes params and returns a subset
            of params for which we actually want to generate the plot.
        n_cores (int): Number of cores used for parallel batch evaluation.
        n_gridpoints (int): Number of gridpoints on which the criterion function is
            evaluated. This is the number per plotted line or grid axis.
        plots_per_row (int): Number of plots per row for combined subplot output.
        param_names (dict or NoneType): Dictionary mapping original parameter names
            to new display names.
        share_y (bool): If True, the individual plots share the y-axis scale and
            plots in the same row actually share the y axis.
        expand_yrange (float): The ratio by which to expand the range of the
            (shared) y axis, so that the axis is not cropped at the exact min/max
            of the criterion value.
        share_x (bool): If True, set the same range of x axis for all plots and share
            the x axis for all plots in one column.
        color (str): The line color for slice plots.
        template (str): The Plotly template used for plot styling.
        title (str or NoneType): The figure title.
        return_dict (bool): If True, return a dictionary with individual plots for
            each parameter or pair, else combine into a single figure.
        make_subplot_kwargs (dict or NoneType): Dictionary of keyword arguments used
            to instantiate a Plotly figure with multiple subplots, e.g., controlling
            horizontal or vertical spacing.
        batch_evaluator (str or callable): See :ref:`batch_evaluators` for options.
        projection (str): Type of plot to generate. Must be one of:
            - "slice": 1D slice plots along each selected parameter
            - "3d": 3D surface plots for each pair of selected parameters
            - "contour": 2D contour plots for each pair of selected parameters
        projection_kwargs (dict or NoneType): Dictionary of keyword arguments for
            customizing 2D/3D projections. Only applies when `projection` is "3d" or
            "contour". Supported keys include:
            - "width" (int): Width of each projection figure.
            - "height" (int): Height of each projection figure.
            - "scene_camera_eye" (dict): Camera position for 3D view.
            - "colormap" (str): Colormap name used for surface or contour.
            - "showscale" (bool): Whether to display the color scale.
            - "line_smoothing" (float): Smoothing factor for contours.
            - "diagonal_show" (bool): Whether to show diagonal subplots in pairwise
            view.

        lower_bounds (deprecated): Use `bounds` instead.
        upper_bounds (deprecated): Use `bounds` instead.

    Returns:
        out (dict or plotly.Figure): If `return_dict` is True, returns a dictionary
            with individual plots for each parameter or parameter pair. Otherwise,
            returns a Plotly figure with all plots combined into subplots.

    Raises:
        ValueError: If projection type is not one of {"slice", "3d", "contour"}.
        ValueError: If selected parameters have non-finite lower or upper bounds.

    """

    projection_kwargs = projection_kwargs or {}

    # Validate projection mode
    if projection not in {"slice", "3d", "contour"}:
        raise ValueError(
            f"Invalid projection '{projection}'. "
            f"Must be one of 'slice', '3d', or 'contour'."
        )

    # Preprocess function and bounds
    func, func_eval = evaluate_func(params, func, func_kwargs)
    bounds = process_bounds(bounds, lower_bounds, upper_bounds)

    # Generate internal parameters and converter for basic structure
    converter, internal_params = get_converter(
        params=params,
        constraints=None,
        bounds=bounds,
        func_eval=func_eval,
        solver_type="value",
    )
    n_params = len(internal_params.values)

    # Subset of parameters which is selected
    selected = select_parameter_indices(converter, selector, n_params)
    if not np.isfinite(internal_params.lower_bounds[selected]).all():
        raise ValueError("All selected parameters must have finite lower bounds.")
    if not np.isfinite(internal_params.upper_bounds[selected]).all():
        raise ValueError("All selected parameters must have finite upper bounds.")

    # Validate sufficient number of parameters for "3d" and "contour" projections
    if projection in {"3d", "contour"} and len(selected) < 2:
        raise ValueError(
            f"{projection!r} projection requires at least two parameters. "
            f"Got {len(selected)}. Please revise the `selector`."
        )

    # Create data grid
    grid_data = generate_grid_data(internal_params, selected, n_gridpoints)

    if projection == "slice":
        return _plot_slice(
            selected,
            internal_params,
            converter,
            grid_data,
            func,
            func_eval,
            batch_evaluator,
            n_cores,
            param_names,
            expand_yrange,
            color,
            template,
            title,
            plots_per_row,
            share_y,
            share_x,
            return_dict,
            make_subplot_kwargs,
        )

    # Apply for '3d' or 'contour' projections only
    return _plot_pairwise(
        selected,
        internal_params,
        converter,
        grid_data,
        func,
        batch_evaluator,
        n_cores,
        projection,
        title,
        template,
        n_gridpoints,
        projection_kwargs,
    )
