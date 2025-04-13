import numpy as np
import plotly.express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from optimagic.config import DEFAULT_N_CORES, PLOTLY_TEMPLATE
from optimagic.typing import AggregationLevel
from optimagic.visualization.plot_data import (
    evaluate_func,
    evaluate_function_values,
    generate_eval_points,
    generate_grid_data,
    generate_internal_params,
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
        X, evaluation_points = generate_eval_points(
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
):
    selected_param_names = [internal_params.names[i] for i in selected]

    if len(selected) == 2:
        # Only two parameters selected: make a single plot
        name_x, name_y = selected_param_names
        selected_names = [name_x, name_y]

        fixed_vars = {
            name: grid_data.iloc[0][name]
            for name in internal_params.names
            if name not in selected_names
        }

        X, Y, evaluation_points = generate_eval_points(
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
        Z = np.reshape(func_values, (n_gridpoints, n_gridpoints))

        trace = (
            go.Surface(
                z=Z,
                x=X,
                y=Y,
                showscale=True,
            )
            if projection == "3d"
            else go.Contour(
                z=Z,
                x=X[0],
                y=Y[:, 0],
                contours_coloring="heatmap",
                line_smoothing=0.85,
            )
        )

        fig = go.Figure(data=[trace])
        fig.update_layout(
            title=title or f"{projection} plot: {name_x} vs {name_y}",
            template=template,
            scene=dict(
                xaxis_title=name_x,
                yaxis_title=name_y,
                zaxis_title="Function Value",
                camera=dict(eye=dict(x=1, y=2, z=0.5)),
            )
            if projection == "3d"
            else None,
            width=700,
            height=600,
        )
        return fig

    else:
        # General pairwise plot logic (len(selected) > 2)
        specs = [
            [
                {"type": "scene"} if projection == "3d" else {"type": "xy"}
                for _ in range(len(selected))
            ]
            for _ in range(len(selected))
        ]

        subplot_titles = [
            f"{selected_param_names[j]} vs {selected_param_names[i]}" if i != j else ""
            for i in range(len(selected))
            for j in range(len(selected))
        ]

        fig = make_subplots(
            rows=len(selected),
            cols=len(selected),
            specs=specs,
            subplot_titles=subplot_titles,
        )

        for i, name_i in enumerate(selected_param_names):
            for j, name_j in enumerate(selected_param_names):
                if i == j:
                    continue

                selected_names = [name_j, name_i]
                fixed_vars = {
                    name: grid_data.iloc[0][name]
                    for name in internal_params.names
                    if name not in selected_names
                }

                X, Y, evaluation_points = generate_eval_points(
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
                Z = np.reshape(func_values, (n_gridpoints, n_gridpoints))

                trace = (
                    go.Surface(z=Z, x=X, y=Y, showscale=False)
                    if projection == "3d"
                    else go.Contour(
                        z=Z,
                        x=X[0],
                        y=Y[:, 0],
                        contours_coloring="heatmap",
                        line_smoothing=0.85,
                    )
                )

                fig.add_trace(trace, row=i + 1, col=j + 1)

        fig.update_layout(
            title=title or f"({projection}) Pairwise Plot",
            template=template,
            width=800,
            height=800,
            scene_camera_eye=dict(x=2, y=2, z=0.1),
            margin=dict(t=30, r=0, l=20, b=10),
        )
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
    # deprecated
    lower_bounds=None,
    upper_bounds=None,
):
    # Preprocess function and bounds
    func, func_eval = evaluate_func(params, func, func_kwargs)
    bounds = process_bounds(bounds, lower_bounds, upper_bounds)

    # Generate internal parameter representation
    converter, internal_params = generate_internal_params(params, bounds, func_eval)
    n_params = len(internal_params.values)

    # Select parameters for plotting
    selected = select_parameter_indices(converter, selector, n_params)
    if not np.isfinite(internal_params.lower_bounds[selected]).all():
        raise ValueError("All selected parameters must have finite lower bounds.")
    if not np.isfinite(internal_params.upper_bounds[selected]).all():
        raise ValueError("All selected parameters must have finite upper bounds.")

    # Create grid data
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
    if projection in {"3d", "contour"}:
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
        )

    return None
