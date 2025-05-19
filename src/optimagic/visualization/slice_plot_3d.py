import warnings
from copy import deepcopy
from enum import Enum
from functools import partial
from typing import Any, Union

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


def slice_plot_3d(
    func: Any,
    params: Any,
    bounds: Any = None,
    func_kwargs: None | dict[Any, Any] = None,
    selector: Any = None,
    n_gridpoints: int = 20,
    projection: Any = "univariate",
    make_subplot_kwargs: Any = None,
    layout_kwargs: Any = None,
    plot_kwargs: Any = None,
    param_names: dict[str, str] | None = None,
    expand_yrange: float = 0.02,
    batch_evaluator: str = "joblib",
    n_cores: int = DEFAULT_N_CORES,
    return_dict: bool = False,
    lower_bounds: Any = None,
    upper_bounds: Any = None,
) -> go.Figure | dict[tuple[int, int], go.Figure]:
    """Generate interactive slice, contour or surface plots of a function over its
    parameters.

    Produces 2D univariate slice plots (each param vs function value), 2D contour plots
    (two params vs function value), or 3D surface plots (two params vs function value)
    of a user-supplied function evaluated on a grid defined by parameter bounds.
    Individual plots can be returned as a dict or combined into a single
    Plotly figure with subplots.

    Args:
        func (callable): criterion function that takes params and returns a scalar,
            PyTree, or FunctionValue object.
        params (pytree): A pytree with parameters.
        bounds (optimagic.Bounds or sequence or None): Lower and upper bounds on the
            parameters. The bounds are used to create
            a grid over which slice plots are drawn. The most general and preferred
            way to specify bounds is an `optimagic.Bounds` an object that collects
            lower, upper, soft_lower, and soft_upper bounds. The soft bounds are
            not used for slice_plots. Each bound type mirrors the structure of params.
            Check our how-to guide on bounds for examples. If params is a flat numpy
            array, you can also provide bounds via any format that is supported by
            scipy.optimize.minimize.
        func_kwargs (dict or None): Extra keywords to pass to `func` on each call.
            Default: None
        selector (callable): Function that takes params and returns a subset
            of params for which we actually want to generate the plot.
            Default: None
        n_gridpoints (int): Number of gridpoints on which the criterion function is
            evaluated. This is the number per plotted line.
            Default: 20
        projection (str or ProjectionConfig): Type of plot: `"univariate"` (2D slice),
            `"contour"` (2D contour), or `"surface"` (3D surface).
            Default: `"univariate"`
        make_subplot_kwargs (dict or None): kwargs for `plotly.subplots.make_subplots`
            Default: None.
            Internal defaults when None:
              - rows, cols computed from a number of parameters and projection
              - horizontal_spacing=1/(cols*5), vertical_spacing=(1/(max(rows-1,1)))/5
              - If projection is univariate, `shared_xaxis` and `shared_yaxis` are added
                 with default value as True.
              - If projection is contour or surface, `specs` grid matching types are
                 added. `row_titles` and `column_title` are added for grid reference.
        layout_kwargs (dict or None): kwargs for figure layout update. Default: None.
            Internal defaults when None:
              - width, height = 450 (single plot) or 300 × cols by 300 × rows
              - template = "plotly" (multivariate) or DEFAULT PLOTLY_TEMPLATE
              - `showlegend` is set to False
              - If projection is surface: scenes are added with the configuration,
                    - default camera eye view: dict(x=2, y=2, z=0.5)
                    - xaxis, yaxis and zaxis titles are None and nticks are 5
        plot_kwargs (dict or None): Nested dict of trace‐level kwargs. Default: None.
            Internal defaults when None:
              - line_plot: {'color_discrete_sequence':['#497ea7'], 'markers': False}
              - scatter_plot: {'marker':{'color':'red','size':5}}
                (Note: Setting scatter plot to None will remove points in the plots.)
              - surface_plot: {'colorscale':'Aggrnyl','showscale':False,'opacity':0.8}
              - contour_plot: {'colorscale':'Aggrnyl','showscale':False,
                                'line_smoothing':0.85}
        param_names (dict or NoneType): Dictionary mapping parameter names to new ones.
            Default: None
        expand_yrange (float): The ration by which to expand the range of the
            y-axis, such that the axis is not cropped at exactly the max of
            Criterion Value.
            Default: 0.02
        batch_evaluator (str or callable): See :ref:`batch_evaluators`.
            Default: "joblib"
        n_cores (int): Number of cores.
            Default: 1
        return_dict (bool): If True, return a dict of individual figures
            keyed by (row, col). If False, return a combined Plotly Figure.
            Default: False
        lower_bounds (sequence or None): Deprecated alias for bound lower limit.
            Default: None
        upper_bounds (sequence or None): Deprecated alias for bound upper limit.
            Default: None

    Returns:
        dict or plotly.Figure:
            If `return_dict=True`, a dict mapping subplot indices to
            Plotly Figure objects. Otherwise, a single combined Plotly Figure with
            shared axes and layout.

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

    if not np.isfinite(internal_params.lower_bounds[selected]).all():
        raise ValueError("All selected parameters must have finite lower bounds.")

    if not np.isfinite(internal_params.upper_bounds[selected]).all():
        raise ValueError("All selected parameters must have finite upper bounds.")

    params_data, display_names = {}, {}
    for pos in selected:
        name = internal_params.names[pos]
        params_data[name] = np.linspace(
            internal_params.lower_bounds[pos],
            internal_params.upper_bounds[pos],
            n_gridpoints,
        )
        display_names[name] = param_names.get(name, name) if param_names else name

    # Projection configuration
    projection = Projection(projection)
    if not projection.is_univariate and n_params < 2:
        raise ValueError(
            f"{projection!r} requires at least two parameters. Got {n_params} params."
        )

    n_params = len(selected)

    # Kwargs evaluation
    plot_kwargs = evaluate_plot_kwargs(plot_kwargs)
    make_subplot_kwargs = evaluate_make_subplot_kwargs(
        make_subplot_kwargs, n_params, projection, display_names
    )
    layout_kwargs = evaluate_layout_kwargs(
        layout_kwargs, projection, make_subplot_kwargs
    )

    plots = {}
    plot_data_cache = {}  # type: ignore
    if projection.is_univariate:
        cols = make_subplot_kwargs.get("cols")
        for idx, param_pos in enumerate(selected):
            row, col = divmod(idx, cols)

            param_name = internal_params.names[param_pos]
            display_name = display_names[param_name]

            grid_univariate = False

            fig = plot_univariate(
                param_pos,
                display_name,
                params_data,
                grid_univariate,
                internal_params,
                converter,
                func,
                func_eval,
                batch_evaluator,
                n_cores,
                plot_kwargs,
                layout_kwargs,
                expand_yrange,
                projection,
            )
            plots[(row, col)] = fig
    else:
        single_plot = True if n_params == 2 else False
        lower_projection = projection.get_config().get("lower")
        upper_projection = projection.get_config().get("upper")

        for i, x_selected in enumerate(selected):
            for j, y_selected in enumerate(selected):
                x_pos = x_selected
                y_pos = y_selected
                if x_pos == y_pos and single_plot:
                    x_pos, y_pos = selected

                # Diagonal plot are slice plots
                if i == j and not single_plot:
                    grid_univariate = True
                    param_name = internal_params.names[x_pos]
                    display_name = display_names[param_name]

                    fig = plot_univariate(
                        x_pos,
                        display_name,
                        params_data,
                        grid_univariate,
                        internal_params,
                        converter,
                        func,
                        func_eval,
                        batch_evaluator,
                        n_cores,
                        plot_kwargs,
                        layout_kwargs,
                        expand_yrange,
                        projection,
                    )
                else:
                    grid_univariate = False
                    subplot_projection = None
                    if i < j and upper_projection is not None:
                        subplot_projection = upper_projection
                    elif i > j and lower_projection is not None:
                        subplot_projection = lower_projection
                    elif i == j and single_plot:
                        subplot_projection = lower_projection
                    if subplot_projection is not None:
                        fig, plot_data_cache = plot_multivariate(
                            x_pos,
                            y_pos,
                            params_data,
                            grid_univariate,
                            internal_params,
                            converter,
                            n_gridpoints,
                            func,
                            func_eval,
                            batch_evaluator,
                            n_cores,
                            plot_kwargs,
                            layout_kwargs,
                            subplot_projection,
                            plot_data_cache,
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


# Helper functions
def evaluate_function_values(
    points: Any, func: Any, batch_evaluator: Any, n_cores: int
) -> list[float]:
    """Evaluate a function on a list of parameter points using a batch evaluator.

    Returns function values for each parameter point, using the specified batch
    evaluator and core count. Failed evaluations are returned as NaN.

    """
    batch_evaluator = process_batch_evaluator(batch_evaluator)
    results: Any = batch_evaluator(
        func=func,
        arguments=points,
        error_handling="continue",
        n_cores=n_cores,
    )
    results = [
        float("nan")
        if isinstance(val, str)
        else val.internal_value(AggregationLevel.SCALAR)
        for val in results
    ]
    return results


def generate_evaluation_points(
    params_data: dict[str, NDArray[np.float64]],
    internal: Any,
    converter: Any,
    params: Any,
    grid_univariate: bool,
    projection: Any,
) -> Any:
    """Generate parameter sets for evaluation based on the projection type.

    Creates a list of parameter vectors by sweeping over one or two parameters, used to
    evaluate function values for univariate or multivariate plots.

    """
    evaluation_points = []
    point = dict(zip(internal.names, internal.values, strict=False))

    if projection.is_univariate or grid_univariate:
        x = params_data[params]

        for p_value in x:
            # updating only the parameter of interest
            point[params] = p_value

            values = np.array(list(point.values()))
            evaluation_points.append(converter.params_from_internal(values))
    else:
        x_name, y_name = params[0], params[1]
        x_vals = params_data[x_name]
        y_vals = params_data[y_name]

        x, y = np.meshgrid(x_vals, y_vals)
        for a, b in zip(x.ravel(), y.ravel(), strict=False):
            point[x_name] = a
            point[y_name] = b
            values = np.array(list(point.values()))
            evaluation_points.append(converter.params_from_internal(values))
    return evaluation_points


# Plot Utils
def plot_univariate(
    param_pos: int,
    display_name: str,
    params_data: dict[str, NDArray[np.float64]],
    grid_univariate: bool,
    internal_params: Any,
    converter: Any,
    func: Any,
    func_eval: Any,
    batch_evaluator: Union[str, Any],
    n_cores: int,
    plot_kwargs: Any,
    layout_kwargs: Any,
    expand_yrange: float,
    projection: Any,
) -> go.Figure:
    """Create a line plot for a single parameter's slice of the function.

    Evaluates the function while varying one parameter and plots the result along with
    the current point using a line. Plot scatter point on initial params.

    """
    param_name = internal_params.names[param_pos]
    eval_points = generate_evaluation_points(
        params_data, internal_params, converter, param_name, grid_univariate, projection
    )

    # Line plot points
    x = params_data[param_name].tolist()
    y = evaluate_function_values(eval_points, func, batch_evaluator, n_cores)
    y_range = compute_yaxis_range(y, expand_yrange)

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
    return fig


def plot_multivariate(
    x_pos: int,
    y_pos: int,
    params_data: dict[str, NDArray[np.float64]],
    grid_univariate: bool,
    internal_params: Any,
    converter: Any,
    n_gridpoints: int,
    func: Any,
    func_eval: Any,
    batch_evaluator: Any,
    n_cores: int,
    plot_kwargs: Any,
    layout_kwargs: Any,
    projection: Any,
    plot_data_cache: Any,
) -> Any:
    """Plot a 3D surface or 2D contour showing function value over two parameters.

    Evaluates the function on a meshgrid over two parameters and visualizes the
    function's behavior using the chosen projection type (surface or contour). Plot
    scatter point on initial params.

    """
    x_name = internal_params.names[x_pos]
    y_name = internal_params.names[y_pos]
    param_names = [x_name, y_name]

    # Keys are sorted to avoid duplicates
    key = tuple(sorted(param_names))
    if key not in plot_data_cache.keys():
        evaluation_points = generate_evaluation_points(
            params_data,
            internal_params,
            converter,
            param_names,
            grid_univariate,
            projection,
        )

        # Line plot points
        x, y = np.meshgrid(params_data[x_name], params_data[y_name])
        z = evaluate_function_values(evaluation_points, func, batch_evaluator, n_cores)
        z = np.reshape(z, (n_gridpoints, n_gridpoints))  # type: ignore[assignment]

        plot_data_cache[key] = {"x": x.copy(), "y": y.copy(), "z": z.copy()}
    else:
        # Reuse plot data by accessing the symmetric counterpart from the cache (dict).
        # When visualizing the lower triangle of the grid (i.e., swapped axis order),
        # transpose the values (x, y, z) and swap X and Y to maintain correct alignment.
        x = plot_data_cache[key]["y"].T
        y = plot_data_cache[key]["x"].T
        z = plot_data_cache[key]["z"].T

    # Scatter plot point
    scatter_point = {
        "x": [internal_params.values[x_pos]],
        "y": [internal_params.values[y_pos]],
        "z": [func_eval.internal_value(AggregationLevel.SCALAR)],
    }

    if projection.is_surface:
        return (
            plot_surface(x, y, z, scatter_point, plot_kwargs, layout_kwargs),
            plot_data_cache,
        )
    else:
        return (
            plot_contour(x, y, z, scatter_point, plot_kwargs, layout_kwargs),
            plot_data_cache,
        )


def plot_line(
    x: list[float],
    y: list[float],
    display_name: str,
    y_range: list[float],
    scatter_point: Any,
    plot_kwargs: Any,
    layout_kwargs: Any,
    grid_univariate: bool,
) -> go.Figure:
    """Generate a 2D line plot with an overlayed scatter point.

    Constructs a line plot of the function values and highlights the current evaluation
    point using a scatter overlay.

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


def plot_surface(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    z: list[float],
    scatter_point: Any,
    plot_kwargs: Any,
    layout_kwargs: Any,
) -> go.Figure:
    """Create a 3D surface plot of the function over two parameters.

    Plots a 3D surface using Plotly and adds a scatter point for the initial parameter.

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


def plot_contour(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    z: list[float],
    scatter_point: Any,
    plot_kwargs: Any,
    layout_kwargs: Any,
) -> go.Figure:
    """Create a 2D contour plot for function values over a parameter grid.

    Uses Plotly to draw a filled contour plot and plots the initial evaluation point.

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
    UNIVARIATE = "univariate"
    CONTOUR = "contour"
    SURFACE = "surface"

    @classmethod
    def validate(cls, value: Any) -> Any:
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
    def __init__(self, value: Any):
        self._raw = value
        self._univariate = False
        self.lower = None
        self.upper = None

        self._parse(value)

    def _parse(self, value: Any) -> Any:
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

    def get_config(self) -> Any:
        if self._univariate:
            return ProjectionConfig.UNIVARIATE
        return {"lower": self.lower, "upper": self.upper}


def compute_yaxis_range(y: list[float], expand_yrange: float) -> list[float]:
    # Calculate expanded y-axis limits based on data range
    y_min, y_max = np.min(y), np.max(y)
    y_range = y_max - y_min
    return [y_min - expand_yrange * y_range, y_max + expand_yrange * y_range]


def combine_plots(
    plots: dict[tuple[int, int], go.Figure],
    make_subplot_kwargs: dict[str, Any],
    layout_kwargs: dict[str, Any] | None,
    expand_yrange: float,
) -> go.Figure:
    """Combine individual Plotly figures into a single subplot layout.

    Merges subplot traces, applies axis sharing and range adjustments, and formats
    layout to produce a unified figure from multiple slices or surfaces.

    """
    plots = deepcopy(plots)

    n_rows = make_subplot_kwargs["rows"]
    n_cols = make_subplot_kwargs["cols"]
    if "row_titles" in make_subplot_kwargs:
        titles = make_subplot_kwargs["row_titles"]
        if n_rows == 1 and n_cols == 1:
            make_subplot_kwargs["row_titles"] = [titles[0]]
            make_subplot_kwargs["column_titles"] = [titles[1]]

    # Create a subplots figure
    fig = make_subplots(**make_subplot_kwargs)
    fig.update_layout(**layout_kwargs)

    # Adjust subplot annotation positions (Grid titles)
    for ann in fig.layout.annotations:
        if abs(ann.y - 1) < 1e-3:
            ann.update(y=-0.18 / n_cols)
        elif abs(ann.x - 0.98) < 1e-3:
            ann.update(x=-0.18 / n_rows, textangle=270)

    shared_y = make_subplot_kwargs.get("shared_yaxes", False)
    shared_x = make_subplot_kwargs.get("shared_xaxes", False)

    all_y, all_x = [], []

    # Add traces
    for (row_idx, col_idx), subfig in plots.items():
        for trace in subfig.data:
            fig.add_trace(trace, row=row_idx + 1, col=col_idx + 1)

            if shared_y and hasattr(trace, "y"):
                all_y.append(np.array(trace.y))
            if shared_x and hasattr(trace, "x"):
                all_x.append(np.array(trace.x))

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

    # Apply shared y-axis range
    if shared_y and all_y:
        y_range = compute_yaxis_range(np.concatenate(all_y), expand_yrange)
        fig.update_yaxes(range=y_range)

    # Apply shared x-axis range
    if shared_x and all_x:
        x_all = np.concatenate(all_x)
        fig.update_xaxes(range=[np.min(x_all), np.max(x_all)])

    return fig


def _get_subplot_spec(
    i: int, j: int, projection: Any, n_selected: int
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


def evaluate_plot_kwargs(plot_kwargs: dict[str, Any] | None) -> dict[str, Any]:
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
            "line_smoothing": 0.85,
        },
    }

    plot_kwargs_defaults.update(plot_kwargs)
    return plot_kwargs_defaults


def evaluate_make_subplot_kwargs(
    make_subplot_kwargs: dict[str, Any] | None,
    n_selected: int,
    projection: Any,
    titles: dict[str, str],
) -> dict[str, Any]:
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
def evaluate_layout_kwargs(
    layout_kwargs: dict[str, Any] | None,
    projection: Any,
    subplot_config: dict[str, Any],
) -> dict[str, Any]:
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
