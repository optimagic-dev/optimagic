import warnings
from copy import deepcopy
from enum import Enum
from functools import partial

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from numpy.typing import NDArray
from plotly.subplots import make_subplots
from pybaum import tree_just_flatten

from optimagic import deprecations
from optimagic.batch_evaluators import process_batch_evaluator
from optimagic.config import DEFAULT_N_CORES, PLOTLY_TEMPLATE
from optimagic.deprecations import replace_and_warn_about_deprecated_bounds
from optimagic.optimization.fun_value import (
    convert_fun_output_to_function_value,
    enforce_return_type,
)
from optimagic.parameters.bounds import pre_process_bounds
from optimagic.parameters.conversion import get_converter
from optimagic.parameters.tree_registry import get_registry
from optimagic.shared.process_user_function import infer_aggregation_level
from optimagic.typing import AggregationLevel


def slice_plot_3d(  # type: ignore[no-untyped-def]
    func,
    params,
    bounds=None,
    func_kwargs=None,
    selector=None,
    n_gridpoints: int = 20,
    projection="univariate",
    make_subplot_kwargs=None,
    layout_kwargs=None,
    plot_kwargs=None,
    param_names: dict[str, str] | None = None,
    expand_yrange: float = 0.02,
    batch_evaluator="joblib",
    n_cores: int = DEFAULT_N_CORES,
    return_dict: bool = False,
    lower_bounds=None,
    upper_bounds=None,
) -> go.Figure | dict[tuple[int, int], go.Figure]:
    """Generate interactive slice, contour or surface plots of a function.

    This function produces plots of a user-supplied criterion function evaluated on a
    grid of its parameters. It can generate:
    - 2D univariate slice plots (each parameter vs. function value).
    - 2D contour plots (two parameters vs. function value).
    - 3D surface plots (two parameters vs. function value).

    Plots can be returned as a dictionary of individual figures or combined into a
    single Plotly figure with subplots.

    Args:
        func (callable): The criterion function. It takes `params` and returns a
            scalar, PyTree, or `FunctionValue` object.
        params (pytree): A pytree of parameters.
        bounds (optimagic.Bounds or sequence or None): An `optimagic.Bounds` object
            or other supported format specifying the lower and upper bounds for
            parameters. These bounds define the grid for the plots.
        func_kwargs (dict or None): Additional keyword arguments for `func`.
        selector (callable): A function that takes `params` and returns a subset
            of them to be plotted. If None, all parameters are plotted.
        n_gridpoints (int): The number of points per parameter used to create the
            evaluation grid. For a 2D plot, this means `n_gridpoints`**2
            evaluations.
        projection (str or dict): The type of plot. Can be `"univariate"`,
            `"contour"`, `"surface"`, or a dictionary like `{"lower": "contour",
            "upper": "surface"}` to create a grid of mixed plot types.
        make_subplot_kwargs (dict or None): Keyword arguments for
            `plotly.subplots.make_subplots`.
        layout_kwargs (dict or None): Keyword arguments for the figure's
            `update_layout` method.
        plot_kwargs (dict or None): A nested dictionary of keyword arguments to
            customize traces, e.g., `{"line_plot": {"color": "blue"}}`.
        param_names (dict or NoneType): A dictionary mapping internal parameter
            names to display names.
        expand_yrange (float): The factor by which to expand the function value
            axis range. This only applies to the z-axis of **surface plots** to
            prevent the plot from feeling cramped. It does not affect line or
            contour plots.
        batch_evaluator (str or callable): The batch evaluator to parallelize
            function evaluations. See :ref:`batch_evaluators`.
        n_cores (int): The number of cores to use for parallelization.
        return_dict (bool): If `True`, returns a dictionary of `go.Figure`
            objects keyed by `(row, col)`. If `False`, returns a single combined
            `go.Figure`.
        lower_bounds (sequence or None): Deprecated. Use `bounds` instead.
        upper_bounds (sequence or None): Deprecated. Use `bounds` instead.

    Returns:
        plotly.Figure | dict: A single combined Plotly figure or a dictionary of
        individual figures.

    """
    bounds = replace_and_warn_about_deprecated_bounds(
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        bounds=bounds,
    )
    bounds = pre_process_bounds(bounds)

    if func_kwargs is not None:
        func = partial(func, **func_kwargs)

    func_eval = func(params)

    # ==================================================================================
    # handle deprecated function output
    # ==================================================================================
    if deprecations.is_dict_output(func_eval):
        msg = (
            "Functions that return dictionaries are deprecated in slice_plot and will "
            "raise an error in version 0.6.0. Please pass a function that returns a "
            "FunctionValue object instead and use the `mark` decorators to specify "
            "whether it is a scalar, least-squares or likelihood function."
        )
        warnings.warn(msg, FutureWarning)
        func_eval = deprecations.convert_dict_to_function_value(func_eval)
        func = deprecations.replace_dict_output(func)

    # ==================================================================================
    # Infer the function type and enforce the return type
    # ==================================================================================

    if deprecations.is_dict_output(func_eval):
        problem_type = deprecations.infer_problem_type_from_dict_output(func_eval)
    else:
        problem_type = infer_aggregation_level(func)

    func_eval = convert_fun_output_to_function_value(func_eval, problem_type)

    func = enforce_return_type(problem_type)(func)

    # ==================================================================================

    converter, internal_params = get_converter(
        params=params,
        constraints=None,
        bounds=bounds,
        func_eval=func_eval,
        solver_type="value",
    )

    n_params = len(internal_params.values)
    selected = np.arange(n_params, dtype=int)
    if selector is not None:
        helper = converter.params_from_internal(selected)
        registry = get_registry(extended=True)
        selected = np.array(
            tree_just_flatten(selector(helper), registry=registry), dtype=int
        ).reshape(-1)
    n_params = len(selected)
    if not np.isfinite(internal_params.lower_bounds[selected]).all():
        raise ValueError("All selected parameters must have finite lower bounds.")

    if not np.isfinite(internal_params.upper_bounds[selected]).all():
        raise ValueError("All selected parameters must have finite upper bounds.")

    # Projection configuration
    projection = Projection(projection)
    if not projection.is_univariate and n_params < 2:
        raise ValueError(
            f"{projection!r} requires at least two parameters. Got {n_params} params."
        )

    params_data, display_names = {}, {}

    for pos in selected:
        name = internal_params.names[pos]
        params_data[name] = np.linspace(
            internal_params.lower_bounds[pos],
            internal_params.upper_bounds[pos],
            n_gridpoints,
        )
        display_names[name] = param_names.get(name, name) if param_names else name

    # This is where
    evaluation_points = generate_evaluation_points(
        projection, selected, internal_params, params_data, converter
    )

    evaluator = process_batch_evaluator(batch_evaluator)

    raw_func_values = evaluator(
        func=func,
        arguments=evaluation_points,
        error_handling="continue",
        n_cores=n_cores,
    )

    # add NaNs where an evaluation failed
    func_values = np.array(
        [
            np.nan
            if isinstance(val, str)
            else val.internal_value(AggregationLevel.SCALAR)
            for val in raw_func_values
        ]
    )

    plot_data = plot_data_cache(
        projection, selected, internal_params, func_values, n_gridpoints
    )

    # Kwargs evaluation
    plot_kwargs = evaluate_plot_kwargs(plot_kwargs)
    make_subplot_kwargs = evaluate_make_subplot_kwargs(
        make_subplot_kwargs, n_params, projection, display_names
    )
    layout_kwargs = evaluate_layout_kwargs(
        layout_kwargs, projection, make_subplot_kwargs
    )

    plots = {}
    if projection.is_univariate:
        cols = make_subplot_kwargs.get("cols")
        for idx, param_pos in enumerate(selected):
            row, col = divmod(idx, cols)

            param_name = internal_params.names[param_pos]
            display_name = display_names[param_name]

            x = params_data[param_name].tolist()
            y = plot_data.get(
                tuple(
                    sorted(
                        [
                            param_name,
                        ]
                    )
                ),
                [],
            )

            y_range = compute_yaxis_range(
                y[~np.isnan(y)] if np.any(~np.isnan(y)) else [0, 1], expand_yrange
            )
            grid_univariate = False

            # Scatter plot point
            scatter_point = {
                "x": [internal_params.values[param_pos]],
                "y": [func_eval.internal_value(AggregationLevel.SCALAR)],
            }

            fig = plot_line(
                x,
                y,
                display_name,
                y_range,
                scatter_point,
                plot_kwargs,
                layout_kwargs,
                grid_univariate,
            )
            plots[(row, col)] = fig
    else:
        single_plot = True if n_params == 2 else False
        projection_config = projection.get_config()
        lower_projection = projection_config.get("lower")
        upper_projection = projection_config.get("upper")

        for i, x_selected in enumerate(selected):
            for j, y_selected in enumerate(selected):
                if x_selected == y_selected and single_plot:
                    x_pos, y_pos = selected
                else:
                    x_pos = x_selected
                    y_pos = y_selected

                # Diagonal plot are slice plots
                if i == j and not single_plot:
                    grid_univariate = True
                    param_name = internal_params.names[x_pos]
                    display_name = display_names[param_name]

                    x = params_data[param_name].tolist()
                    y = plot_data.get(
                        tuple(
                            sorted(
                                [
                                    param_name,
                                ]
                            )
                        ),
                        [],
                    )
                    y_range = compute_yaxis_range(y, expand_yrange)

                    # Scatter plot point
                    scatter_point = {
                        "x": [internal_params.values[x_pos]],
                        "y": [func_eval.internal_value(AggregationLevel.SCALAR)],
                    }

                    fig = plot_line(
                        x,
                        y,
                        display_name,
                        y_range,
                        scatter_point,
                        plot_kwargs,
                        layout_kwargs,
                        grid_univariate,
                    )

                else:
                    subplot_projection = None
                    if i < j and upper_projection is not None:
                        subplot_projection = upper_projection
                    elif i > j and lower_projection is not None:
                        subplot_projection = lower_projection
                    elif i == j and single_plot:
                        subplot_projection = lower_projection

                    if subplot_projection is not None:
                        x_name = internal_params.names[x_pos]
                        y_name = internal_params.names[y_pos]
                        current_param_names = [x_name, y_name]

                        x, y = np.meshgrid(params_data[x_name], params_data[y_name])
                        z = plot_data.get(tuple(sorted(current_param_names)), [])
                        z = np.reshape(z, (n_gridpoints, n_gridpoints))

                        # Scatter plot point
                        scatter_point = {
                            "x": [internal_params.values[x_pos]],
                            "y": [internal_params.values[y_pos]],
                            "z": [func_eval.internal_value(AggregationLevel.SCALAR)],
                        }

                        if subplot_projection.is_surface:
                            fig = plot_surface(
                                x, y, z, scatter_point, plot_kwargs, layout_kwargs
                            )
                        else:
                            fig = plot_contour(
                                x, y, z, scatter_point, plot_kwargs, layout_kwargs
                            )
                    else:
                        fig = go.Figure()
                plots[(i, j)] = fig
                if single_plot:
                    break
            if single_plot:
                break

    if return_dict:
        return plots
    return combine_plots(plots, make_subplot_kwargs, layout_kwargs, expand_yrange)


def generate_evaluation_points(  # type: ignore[no-untyped-def]
    projection, selected, internal_params, params_data, converter
):
    """Create the list of parameter sets for function evaluation.

    This function generates all the points (parameter sets) that need to be
    evaluated by the criterion function to create the plots. It generates points
    for both univariate slices and, if applicable, bivariate grids.

    Args:
        projection (Projection): The processed projection configuration object.
        selected (NDArray[int]): Array of integer positions for the selected
            parameters.
        internal_params (InternalParams): An object holding the internal parameter
            representation (values, names, bounds).
        params_data (dict): A dictionary mapping parameter names to their grid
            values (np.linspace array).
        converter (Converter): The parameter converter object.

    Returns:
        list: A list of parameter pytrees. Each element is a full parameter set
        ready to be passed to the user's criterion function.

    """
    evaluation_points = []
    default_point = dict(
        zip(internal_params.names, internal_params.values, strict=False)
    )
    for pos in selected:
        name = internal_params.names[pos]
        for value in params_data[name]:
            point = default_point.copy()
            point[name] = value
            values = np.array(list(point.values()))
            evaluation_points.append(converter.params_from_internal(values))
    if projection.is_dict:
        for x_pos in selected:
            for y_pos in selected:
                if x_pos == y_pos:
                    continue
                x_name = internal_params.names[x_pos]
                y_name = internal_params.names[y_pos]

                x_mesh, y_mesh = np.meshgrid(params_data[x_name], params_data[y_name])
                for x_val, y_val in zip(x_mesh.ravel(), y_mesh.ravel(), strict=False):
                    point = default_point.copy()
                    point[x_name] = x_val
                    point[y_name] = y_val
                    values = np.array(list(point.values()))
                    evaluation_points.append(converter.params_from_internal(values))
    return evaluation_points


def plot_data_cache(  # type: ignore[no-untyped-def]
    projection, selected, internal_params, func_values, n_gridpoints
):
    """Caches and maps evaluated function values to their parameters.

    This function takes the flat array of criterion function outputs and maps
    them back to the parameters that generated them. The result is a dictionary
    where keys are tuples of parameter names and values are the corresponding
    function values.

    Args:
        projection (Projection): The processed projection configuration object.
        selected (NDArray[int]): Array of integer positions for the selected
            parameters.
        internal_params (InternalParams): An object holding the internal parameter
            representation.
        func_values (NDArray[float]): A flat numpy array containing the results
            from the batch evaluator.
        n_gridpoints (int): The number of grid points per parameter.

    Returns:
        dict: A dictionary mapping parameter name tuples to numpy arrays of
        function values.
        - For univariate plots: `{(param_name,): array([...])}`
        - For bivariate plots: `{(param_a, param_b): array([...])}`

    """
    plot_data = {}
    func_values_idx = 0

    for pos in selected:
        key = tuple(
            sorted(
                [
                    internal_params.names[pos],
                ]
            )
        )
        y = func_values[func_values_idx : func_values_idx + n_gridpoints]
        plot_data[key] = y
        func_values_idx += n_gridpoints

    if projection.is_dict:
        for x_pos in selected:
            for y_pos in selected:
                if x_pos == y_pos:
                    continue
                key = tuple(
                    sorted([internal_params.names[x_pos], internal_params.names[y_pos]])
                )
                plot_data[key] = func_values[
                    func_values_idx : func_values_idx + (n_gridpoints**2)
                ]
                func_values_idx += n_gridpoints**2

    return plot_data


def plot_line(  # type: ignore[no-untyped-def]
    x: list[float],
    y: list[float],
    display_name: str,
    y_range: list[float],
    scatter_point,
    plot_kwargs,
    layout_kwargs,
    grid_univariate: bool,
) -> go.Figure:
    """Generate a 2D line plot with an overlayed scatter point.

    This function constructs a line plot for a univariate parameter slice and
    highlights the initial parameter's function value with a scatter marker.

    Args:
        x (list[float]): The parameter values for the x-axis.
        y (list[float]): The function values for the y-axis.
        display_name (str): The name of the parameter to be used as the x-axis
            title.
        y_range (list[float]): A list `[min, max]` defining the y-axis range.
        scatter_point (dict): A dictionary with "x" and "y" keys for the
            overlayed scatter marker.
        plot_kwargs (dict): A dictionary of trace-level customizations.
        layout_kwargs (dict): A dictionary of layout customizations.
        grid_univariate (bool): If `True`, this is a diagonal plot in a grid,
            and axis titles are omitted.

    Returns:
        go.Figure: A Plotly figure object containing the line plot.

    """
    fig = px.line(x=x, y=y, **plot_kwargs["line_plot"])
    if plot_kwargs["scatter_plot"] is not None:
        fig.add_trace(
            go.Scatter(
                x=scatter_point["x"],
                y=scatter_point["y"],
                **plot_kwargs["scatter_plot"],
            )
        )

    if layout_kwargs:
        fig.update_layout(**layout_kwargs)

    if not grid_univariate:
        fig.update_xaxes(title={"text": display_name})
        fig.update_yaxes(title={"text": "Function Value"}, range=y_range)
    else:
        fig.update_xaxes(title=None)
        fig.update_yaxes(title=None, range=y_range)
    return fig


def plot_surface(  # type: ignore[no-untyped-def]
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    z,
    scatter_point,
    plot_kwargs,
    layout_kwargs,
):
    """Create a 3D surface plot of the function over two parameters.

    This function constructs a 3D surface plot and highlights the initial
    parameter's function value with a 3D scatter marker.

    Args:
        x (NDArray[np.float64]): A meshgrid of x-axis parameter values.
        y (NDArray[np.float64]): A meshgrid of y-axis parameter values.
        z (NDArray[np.float64]): A 2D array of function values corresponding
            to the x-y grid.
        scatter_point (dict): A dictionary with "x", "y", and "z" keys for the
            overlayed 3D scatter marker.
        plot_kwargs (dict): A dictionary of trace-level customizations.
        layout_kwargs (dict): A dictionary of layout customizations.

    Returns:
        go.Figure: A Plotly figure object containing the surface plot.

    """
    trace = go.Surface(z=z, x=x, y=y, **plot_kwargs["surface_plot"])

    fig = go.Figure(data=[trace], layout=layout_kwargs)
    if plot_kwargs["scatter_plot"] is not None:
        fig.add_trace(
            go.Scatter3d(
                x=scatter_point["x"],
                y=scatter_point["y"],
                z=scatter_point["z"],
                **plot_kwargs["scatter_plot"],
            )
        )
    return fig


def plot_contour(  # type: ignore[no-untyped-def]
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    z: list[float],
    scatter_point,
    plot_kwargs,
    layout_kwargs,
):
    """Create a 2D contour plot for function values over a parameter grid.

    This function constructs a 2D contour plot and highlights the initial
    parameter's function value with a scatter marker.

    Args:
        x (NDArray[np.float64]): A meshgrid of x-axis parameter values.
        y (NDArray[np.float64]): A meshgrid of y-axis parameter values.
        z (list[float]): A list of function values corresponding to the grid.
        scatter_point (dict): A dictionary with "x" and "y" keys for the
            overlayed scatter marker.
        plot_kwargs (dict): A dictionary of trace-level customizations.
        layout_kwargs (dict): A dictionary of layout customizations.

    Returns:
        go.Figure: A Plotly figure object containing the contour plot.

    """
    trace = go.Contour(
        z=z, x=x[0], y=y[:, 0], coloraxis="coloraxis", **plot_kwargs["contour_plot"]
    )
    fig = go.Figure(data=[trace], layout=layout_kwargs)

    if plot_kwargs["scatter_plot"] is not None:
        fig.add_trace(
            go.Scatter(
                x=scatter_point["x"],
                y=scatter_point["y"],
                **plot_kwargs["scatter_plot"],
            )
        )
    return fig


class ProjectionConfig(str, Enum):
    """An Enum to validate and represent supported projection types."""

    UNIVARIATE = "univariate"
    CONTOUR = "contour"
    SURFACE = "surface"

    @classmethod
    def validate(cls, value):  # type: ignore[no-untyped-def]
        if value is None:
            return None
        if isinstance(value, str):
            value = value.lower()
            if value in cls._value2member_map_:
                return cls(value)
            raise ValueError(f"Invalid projection: '{value}'")
        raise TypeError(f"Expected str or None, got {type(value)}")

    @property
    def is_univariate(self) -> bool:
        return self == ProjectionConfig.UNIVARIATE

    @property
    def is_surface(self) -> bool:
        return self == ProjectionConfig.SURFACE

    @property
    def is_contour(self) -> bool:
        return self == ProjectionConfig.CONTOUR


class Projection:
    """A helper class to parse the `projection` argument.

    This class handles parsing the `projection` argument, which can be a simple
    string (e.g., "univariate") or a dictionary (e.g., `{"lower": "contour",
    "upper": "surface"}`) for creating mixed-grid plots.

    """

    def __init__(self, value):  # type: ignore[no-untyped-def]
        self._univariate = False
        self.lower = None
        self.upper = None

        self._parse(value)

    def _parse(self, value):  # type: ignore[no-untyped-def]
        if isinstance(value, str):
            value = value.lower()
            if value == ProjectionConfig.UNIVARIATE:
                self._univariate = True
            elif value in (ProjectionConfig.SURFACE, ProjectionConfig.CONTOUR):
                self.lower = ProjectionConfig.validate(value)
                self.upper = None
            else:
                raise ValueError(f"Invalid projection: '{value}'")
        elif isinstance(value, dict):
            self.lower = ProjectionConfig.validate(value.get("lower"))
            self.upper = ProjectionConfig.validate(value.get("upper"))
        else:
            raise TypeError(
                f"Invalid type for projection: {type(value)}. "
                "Must be a string or dict with 'lower' and 'upper' keys."
            )

    @property
    def is_univariate(self) -> bool:
        return self._univariate

    @property
    def is_dict(self) -> bool:
        return not self._univariate

    def get_config(self):  # type: ignore[no-untyped-def]
        if self._univariate:
            return ProjectionConfig.UNIVARIATE
        return {"lower": self.lower, "upper": self.upper}


def compute_yaxis_range(y: list[float], expand_yrange: float) -> list[float]:
    # Calculate expanded y-axis limits based on data range
    y_min, y_max = np.min(y), np.max(y)
    y_range = y_max - y_min
    return [y_min - expand_yrange * y_range, y_max + expand_yrange * y_range]


def combine_plots(  # type: ignore[no-untyped-def]
    plots: dict[tuple[int, int], go.Figure],
    make_subplot_kwargs,
    layout_kwargs,
    expand_yrange: float,
) -> go.Figure:
    """Combine individual Plotly figures into a single subplot layout.

    This function merges traces from a dictionary of individual plots into a
    single `go.Figure` with a subplot grid. It handles axis sharing, range
    adjustments, and overall layout formatting.

    Args:
        plots (dict): A dictionary mapping `(row, col)` tuples to `go.Figure`
            objects.
        make_subplot_kwargs (dict): Keyword arguments for `make_subplots`.
        layout_kwargs (dict): Keyword arguments for the final layout update.
        expand_yrange (float): The expansion factor to apply to any shared
            y-axes.

    Returns:
        go.Figure: A single, combined Plotly Figure object.

    """
    plots = deepcopy(plots)

    # --- NEW, SIMPLIFIED LOGIC FOR SINGLE PLOTS ---
    # If the plot grid is just 1x1, do not rebuild the figure.
    # Return the already correctly-scaled plot directly.
    if make_subplot_kwargs.get("rows") == 1 and make_subplot_kwargs.get("cols") == 1:
        # Extract the single figure from the plots dictionary.
        (row, col), fig = plots.popitem()

        # Apply final layout customizations like width and height.
        fig.update_layout(**layout_kwargs)

        # Get the correct titles for the x and y axes.
        # Note: A bug in title assignment is also fixed here.
        all_titles = make_subplot_kwargs.get("column_titles", ["", ""])
        x_title = all_titles[0]
        y_title = all_titles[1]

        # Assign titles correctly depending on whether it's a 3D or 2D plot.
        if hasattr(fig.layout, "scene") and fig.layout.scene:
            scene_key = next(key for key in fig.layout if key.startswith("scene"))
            fig.layout[scene_key].xaxis.title = x_title
            fig.layout[scene_key].yaxis.title = y_title
            fig.layout[scene_key].zaxis.title = "Function Value"
        else:
            fig.update_xaxes(title_text=x_title)
            fig.update_yaxes(title_text=y_title)

        return fig
    # --- END OF NEW LOGIC ---

    # --- Original logic for creating a grid of subplots (for len(plots) > 1) ---
    fig = make_subplots(**make_subplot_kwargs)
    fig.update_layout(**layout_kwargs)

    for ann in fig.layout.annotations:
        if abs(ann.y - 1) < 1e-3:
            ann.update(y=-0.18 / make_subplot_kwargs["cols"])
        elif abs(ann.x - 0.98) < 1e-3:
            ann.update(x=-0.18 / make_subplot_kwargs["rows"], textangle=270)

    shared_y = make_subplot_kwargs.get("shared_yaxes", False)
    shared_x = make_subplot_kwargs.get("shared_xaxes", False)
    all_y, all_x = [], []

    for (row_idx, col_idx), subfig in plots.items():
        for trace in subfig.data:
            fig.add_trace(trace, row=row_idx + 1, col=col_idx + 1)
            if shared_y and hasattr(trace, "y"):
                arr = np.array(trace.y)
                if arr.ndim > 0:
                    all_y.append(arr)
            if shared_x and hasattr(trace, "x"):
                arr = np.array(trace.x)
                if arr.ndim > 0:
                    all_x.append(arr)

        if hasattr(subfig.layout, "xaxis") and hasattr(subfig.layout.xaxis, "title"):
            fig.update_xaxes(
                title_text=subfig.layout.xaxis.title.text,
                row=row_idx + 1,
                col=col_idx + 1,
            )
        if hasattr(subfig.layout, "yaxis") and hasattr(subfig.layout.yaxis, "title"):
            if shared_y:
                if col_idx == 0:
                    fig.update_yaxes(
                        title_text=subfig.layout.yaxis.title.text,
                        row=row_idx + 1,
                        col=col_idx + 1,
                    )
            else:
                fig.update_yaxes(
                    title_text=subfig.layout.yaxis.title.text,
                    row=row_idx + 1,
                    col=col_idx + 1,
                )

    if shared_y and all_y:
        y_range = compute_yaxis_range(np.concatenate(all_y), expand_yrange)
        fig.update_yaxes(range=y_range)
    if shared_x and all_x:
        x_all = np.concatenate(all_x)
        fig.update_xaxes(range=[np.min(x_all), np.max(x_all)])

    return fig


def _get_subplot_spec(  # type: ignore[no-untyped-def]
    i: int, j: int, projection, n_selected: int
) -> dict[str | None, str | None]:
    # Determine subplot spec type (xy, scene, contour) for a given subplot position.
    if i == j and n_selected != 2:
        return {"type": "xy"}

    projection_config = projection.get_config()
    if n_selected == 2:
        sub_projection = projection_config["lower"]
    else:
        sub_projection = (
            projection_config["lower"] if i > j else projection_config["upper"]
        )

    if sub_projection:
        if sub_projection.is_surface:
            return {"type": "scene"}
        elif sub_projection.is_contour:
            return {"type": "contour"}

    return {}


def evaluate_plot_kwargs(plot_kwargs):  # type: ignore[no-untyped-def]
    # Set default styling for plots if not provided by the user.
    if plot_kwargs is None:
        plot_kwargs = {}

    plot_kwargs_defaults = {
        "line_plot": {
            "color_discrete_sequence": ["#497ea7"],
            "markers": False,
            "template": PLOTLY_TEMPLATE,
        },
        "scatter_plot": {
            "marker": {"color": "red", "size": 5},
        },
        "surface_plot": {
            "colorscale": "Aggrnyl",
            "showscale": False,
            "opacity": 0.8,
        },
        "contour_plot": {
            "colorscale": "Aggrnyl",
            "showscale": True,
            # "line_smoothing": 0.85,
        },
    }

    plot_kwargs_defaults.update(plot_kwargs)
    return plot_kwargs_defaults


def evaluate_make_subplot_kwargs(  # type: ignore[no-untyped-def]
    make_subplot_kwargs,
    n_selected: int,
    projection,
    titles: dict[str, str],
):
    # Set default parameters for make_subplots() if not provided by user.
    if make_subplot_kwargs is None:
        make_subplot_kwargs = {}

    if projection.is_dict and any(k in make_subplot_kwargs for k in ["rows", "cols"]):
        raise ValueError(
            f"`rows` and `cols` cannot be manually specified when projection is "
            f"{projection} is of grid type."
        )

    if projection.is_univariate:
        cols = make_subplot_kwargs.get("cols", 1 if n_selected == 1 else 2)
        rows = (n_selected + cols - 1) // cols
        make_subplot_defaults = {
            "rows": rows,
            "cols": cols,
            "shared_xaxes": True,
            "shared_yaxes": True,
        }
    else:
        rows = cols = n_selected if n_selected > 2 else 1

        specs = []
        for i in range(rows):
            specs_row = []
            for j in range(cols):
                specs_row.append(_get_subplot_spec(i, j, projection, n_selected))
            specs.append(specs_row)

        make_subplot_defaults = {
            "rows": rows,
            "cols": cols,
            "specs": specs,
            "row_titles": list(titles.values()),
            "column_titles": list(titles.values()),
        }

    make_subplot_defaults.update(
        {
            "horizontal_spacing": 1 / (make_subplot_defaults["cols"] * 5),
            "vertical_spacing": (1 / max(make_subplot_defaults["rows"] - 1, 1)) / 5,
        }
    )
    make_subplot_defaults.update(make_subplot_kwargs)
    return make_subplot_defaults


# mypy: disable-error-code="dict-item"
def evaluate_layout_kwargs(  # type: ignore[no-untyped-def]
    layout_kwargs,
    projection,
    subplot_config,
):
    # Set default parameters for update_layout() if not provided by user.

    # Default camera view
    default_scene_camera_view = dict(x=2, y=2, z=0.5)

    if layout_kwargs is None:
        layout_kwargs = {}
    layout_defaults = {}

    if subplot_config.get("rows", 0) > 1 or subplot_config.get("cols", 0) > 1:
        width = 300 * subplot_config.get("cols", 0)
        height = 300 * subplot_config.get("rows", 0)
    else:
        width = 450
        height = 450

    if projection.is_dict:
        scene_layout = {}
        scene_counter = 0

        template = "plotly"

        rows = subplot_config.get("rows", 0)
        cols = subplot_config.get("cols", 0)

        scene_layout["coloraxis"] = {"colorscale": "aggrnyl"}

        if "specs" in subplot_config:
            specs = subplot_config["specs"]
            for i in range(rows):
                for j in range(cols):
                    if "type" in specs[i][j] and specs[i][j]["type"] == "scene":
                        scene_counter += 1
                        scene_id = f"scene{scene_counter}"
                        scene_layout[f"{scene_id}"] = {
                            "camera": {"eye": default_scene_camera_view},
                            "xaxis": dict(title="", nticks=5),
                            "yaxis": dict(title="", nticks=5),
                            "zaxis": dict(title="", nticks=5),
                        }

            layout_defaults.update(scene_layout)
    else:
        template = PLOTLY_TEMPLATE

    layout_defaults.update(
        {
            "width": width,
            "height": height,
            "template": template,
            "showlegend": False,
        }
    )

    layout_defaults.update(layout_kwargs)
    return layout_defaults
