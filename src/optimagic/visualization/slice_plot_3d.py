import warnings
from copy import deepcopy
from enum import Enum, auto
from functools import partial

import numpy as np
import plotly.express as px
from plotly import graph_objects as go
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
    func,
    params,
    bounds=None,
    func_kwargs=None,
    selector=None,
    n_gridpoints=20,
    projection="slice",
    make_subplot_kwargs=None,
    layout_kwargs=None,
    plot_kwargs=None,
    param_names=None,
    expand_yrange=0.02,
    batch_evaluator="joblib",
    n_cores=DEFAULT_N_CORES,
    return_dict=False,
    lower_bounds=None,
    upper_bounds=None,
):
    """
    Generate interactive slice, contour or surface plots of a function
    over its parameters.

    Produces 2D slice plots (one parameter at a time), 2D contour plots
    (two parameters), or 3D surface plots (two parameters) of a user-supplied
    function evaluated on a grid defined by parameter bounds. Individual plots can
    be returned as a dict or combined into a single
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
        projection (str or Projection): Type of plot: `"slice"` (2D slice),
            `"contour"` (2D contour), or `"surface"` (3D surface).
            Default: `"slice"`
        make_subplot_kwargs (dict or None): kwargs for `plotly.subplots.make_subplots`
            Default: None.
            Internal defaults when None:
              • rows, cols computed from a number of parameters and projection
              • start_cell='top-left', print_grid=False
              • horizontal_spacing=1/(cols*5), vertical_spacing=(1/(max(rows-1,1)))/5
              • If projection is contour or surface, `specs` grid matching types are
                 added.
        layout_kwargs (dict or None): kwargs for figure layout update. Default: None.
            Internal defaults when None:
              • width, height = 450 (single plot) or 300 × cols by 300 × rows
              • template = "plotly" (multi‐parameter) or DEFAULT PLOTLY_TEMPLATE
              • xaxis_showgrid=False, yaxis_showgrid=False
              • title={'text':'Slice Plot'}
        plot_kwargs (dict or None): Nested dict of trace‐level kwargs. Default: None.
            Internal defaults when None:
              • line_plot: {'color_discrete_sequence':['#497ea7'], 'markers': False}
              • scatter_plot: {'marker':{'color':'#497ea7','size':4}}
              • surface_plot (if projection="surface"):
                    {'colorscale':'Aggrnyl','showscale':False,'opacity':0.8}
              • contour_plot (if projection="contour"):
                    {'colorscale':'Aggrnyl','showscale':False,'line_smoothing':0.85}
        param_names (dict or NoneType): Dictionary mapping old parameter names
            to new ones.
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
            keyed by (row,col). If False, return a combined Plotly Figure.
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
    projection = Projection.from_value(projection)
    template = "plotly" if projection.is_multiple() else PLOTLY_TEMPLATE

    # Initialize plot, subplot and layout kwargs
    plot_kwargs = plot_kwargs or {}
    make_subplot_kwargs = make_subplot_kwargs or {}
    layout_kwargs = layout_kwargs or {}

    # Preprocess function and bounds
    func, func_eval = evaluate_func(params, func, func_kwargs)
    bounds = process_bounds(bounds, lower_bounds, upper_bounds)

    # Setup converter and parameters
    converter, internal_params = get_converter(
        params=params,
        constraints=[],
        bounds=bounds,
        func_eval=func_eval,
        solver_type="value",
    )

    selected = select_parameter_indices(converter, internal_params, selector)
    selected_size = len(selected)

    if projection.is_multiple() and selected_size < 2:
        raise ValueError(
            f"{projection!r} requires at least two parameters. Got {selected_size}."
        )

    # Generate data for plotting
    param_data = generate_param_data(internal_params, selected, n_gridpoints)

    # Update plotting-related kwargs
    make_subplot_kwargs, layout_kwargs, plot_kwargs = evaluate_kwargs(
        projection,
        selected_size,
        make_subplot_kwargs,
        layout_kwargs,
        plot_kwargs,
        return_dict,
        template,
    )

    plots = {}
    if projection.is_single():
        cols = make_subplot_kwargs.get("cols", 1)
        for idx, pos in enumerate(selected):
            fig = plot_single_param(
                pos,
                param_data,
                param_names,
                internal_params,
                func,
                func_eval,
                converter,
                batch_evaluator,
                n_cores,
                expand_yrange,
                plot_kwargs,
                make_subplot_kwargs,
                layout_kwargs,
            )
            row, col = divmod(idx, cols)
            plots[(row, col)] = fig
    else:
        single_plot = selected_size == 2
        for i, pos_x in enumerate(selected):
            for j, pos_y in enumerate(selected):
                if single_plot:
                    pos_y += 1

                # Diagonal plot are slice plots
                if pos_x == pos_y and not single_plot:
                    fig = plot_single_param(
                        pos_x,
                        param_data,
                        param_names,
                        internal_params,
                        func,
                        func_eval,
                        converter,
                        batch_evaluator,
                        n_cores,
                        expand_yrange,
                        plot_kwargs,
                        make_subplot_kwargs,
                        layout_kwargs,
                    )
                else:
                    fig = plot_multiple_params(
                        pos_x,
                        pos_y,
                        param_data,
                        internal_params,
                        param_names,
                        func,
                        func_eval,
                        converter,
                        batch_evaluator,
                        n_cores,
                        projection,
                        n_gridpoints,
                        plot_kwargs,
                        layout_kwargs,
                    )
                plots[(i, j)] = fig
                if single_plot:
                    break
            if single_plot:
                break

    if return_dict:
        return plots
    return combine_plots(plots, make_subplot_kwargs, layout_kwargs, expand_yrange)


def plot_single_param(
    pos,
    param_data,
    param_names,
    internal_params,
    func,
    func_eval,
    converter,
    batch_evaluator,
    n_cores,
    expand_yrange,
    plot_kwargs,
    make_subplot_kwargs,
    layout_kwargs,
):
    """
    Generate a 2D slice plot for a single parameter index.

    1. Extracts the parameter name and display label.
    2. Builds arrays of parameter values across a predefined grid.
    3. Evaluates the target function at each grid point in batch mode,
       leveraging parallelism as configured by `batch_evaluator` and `n_cores`.
    4. Computes an expanded y-axis range based on `expand_yrange` to ensure
       adequate plot padding.
    5. Creates a Plotly line plot of the function values and overlays a
       scatter marker at the initial parameters.

    Returns:
        go.Figure: A Plotly figure with line and marker traces, ready for
        integration into subplots or standalone display.
    """
    param_name = internal_params.names[pos]
    display_name = (
        param_names.get(param_name, param_name) if param_names else param_name
    )
    x = param_data[param_name].tolist()
    y = evaluate_function_values(
        param_data,
        internal_params,
        param_name,
        func,
        converter,
        batch_evaluator,
        n_cores,
    )
    y_range = compute_yaxis_range(y, expand_yrange)
    fig = plot_slice(
        x=x,
        y=y,
        point={
            "x": [internal_params.values[pos]],
            "y": [func_eval.internal_value(AggregationLevel.SCALAR)],
        },
        display_name=display_name,
        y_range=y_range,
        plot_kwargs=plot_kwargs,
        make_subplot_kwargs=make_subplot_kwargs,
        layout_kwargs=layout_kwargs,
    )
    return fig


def plot_multiple_params(
    pos_x,
    pos_y,
    param_data,
    internal_params,
    param_names,
    func,
    func_eval,
    converter,
    batch_evaluator,
    n_cores,
    projection,
    n_gridpoints,
    plot_kwargs,
    layout_kwargs,
):
    """
    Generate a 2D contour or 3D surface plot for two parameters.

    # TODO: avoid redundant function evaluations to computational efficacy

    1. Maps the selected parameter indices (`pos_x`, `pos_y`) to their names.
    2. Constructs a meshgrid of x-y values over the bounds grid.
    3. Batch-evaluates the user function at each (x,y) pair, reshaping
       results into a matrix of size (n_gridpoints, n_gridpoints).
    4. Depending on `projection`, dispatches to either `plot_contour` or
      `plot_surface`, passing through customized `plot_kwargs`.

    Returns:
        go.Figure: A contour or surface figure, optionally annotated with
        the function value at the initial parameters.
    """
    x_name = internal_params.names[pos_x]
    y_name = internal_params.names[pos_y]

    x = param_data[x_name]
    y = param_data[y_name]

    z = evaluate_function_values(
        param_data,
        internal_params,
        [x_name, y_name],
        func,
        converter,
        batch_evaluator,
        n_cores,
        projection,
    )
    x, y = np.meshgrid(x, y)
    z = np.reshape(z, (n_gridpoints, n_gridpoints))
    display = {
        "x": param_names.get(x_name, x_name) if param_names else x_name,
        "y": param_names.get(y_name, y_name) if param_names else y_name,
    }
    if projection == Projection.SURFACE:
        return plot_surface(
            x,
            y,
            z,
            plot_kwargs=plot_kwargs,
            layout_kwargs=layout_kwargs,
            # Uncomment if the point is a necessary
            # point={"x": [internal_params.values[pos_x]],
            #        "y": [internal_params.values[pos_y]],
            #        "z": [func_eval.internal_value(
            #            AggregationLevel.SCALAR)]},
        )
    return plot_contour(
        x,
        y,
        z,
        plot_kwargs=plot_kwargs,
        layout_kwargs=layout_kwargs,
        # Uncomment if the point is a necessary
        # point={"x": [internal_params.values[pos_x]],
        #        "y": [internal_params.values[pos_y]]}
    )


def plot_slice(
    x,
    y,
    y_range=None,
    point=None,
    display_name=None,
    plot_kwargs=None,
    make_subplot_kwargs=None,
    layout_kwargs=None,
):
    """
    Create a 2D line plot with an initial parameter marker.

    1. Uses Plotly Express to draw a line of y vs. x with settings from
      `plot_kwargs['line_plot']`.
    2. If `point` is provided, overlays a scatter trace at the specified
      (x,y) coordinates using `plot_kwargs['scatter_plot']`.
    3. Applies axis titles: `display_name` on x-axis, "Function Value" on y-axis.
    4. Honors `shared_yaxes` from `make_subplot_kwargs` to set a common y-range.
    5. Updates figure layout from `layout_kwargs`.

    Returns:
        go.Figure: Configured 2D line and marker plot.
    """
    fig = px.line(x=x, y=y, **plot_kwargs["line_plot"])

    if point:
        fig.add_trace(
            go.Scatter(x=point["x"], y=point["y"], **plot_kwargs["scatter_plot"])
        )

    if layout_kwargs:
        fig.update_layout(**layout_kwargs)
    fig.update_xaxes(title={"text": display_name})
    fig.update_yaxes(
        title={"text": "Function Value"},
        range=y_range
        if "shared_yaxes" in make_subplot_kwargs and make_subplot_kwargs["shared_yaxes"]
        else None,
    )
    return fig


def plot_surface(x, y, z, layout_kwargs=None, plot_kwargs=None, point=None):
    """
    Construct a 3D surface plot of z = f(x,y,z).

    1. Creates a `go.Surface` trace with x, y, z matrices and
      styling from `plot_kwargs['surface_plot']`.
    2. Builds a `go.Figure` with the surface as its primary data.
    3. If `point` is specified, adds a 3D scatter marker at the
      current (x,y,z) location.
    4. Applies overall scene and layout configurations via `layout_kwargs`.

    Returns:
        go.Figure: A 3D surface visualization.
    """
    trace = go.Surface(z=z, x=x, y=y, **plot_kwargs["surface_plot"])

    fig = go.Figure(data=[trace], layout=layout_kwargs)

    if point:
        fig.add_trace(
            go.Scatter3d(
                x=point["x"], y=point["y"], z=point["z"], **plot_kwargs["scatter_plot"]
            )
        )
    return fig


def plot_contour(x, y, z, layout_kwargs=None, plot_kwargs=None, point=None):
    """
    Build a 2D contour plot of z = f(x,y,z).

    1. Uses `go.Contour` with flattened x and y axes and z data,
      styled via `plot_kwargs['contour_plot']`.
    2. Wraps the contour in a `go.Figure` and applies `layout_kwargs`.
    3. Optionally overlays a 3D scatter marker for the current parameter
      point (projected onto the x-y plane).

    Returns:
        go.Figure: A contour map with optional annotation.
    """
    trace = go.Contour(z=z, x=x[0], y=y[:, 0], **plot_kwargs["contour_plot"])

    fig = go.Figure(data=[trace], layout=layout_kwargs)
    if point:
        fig.add_trace(
            go.Scatter3d(x=point["x"], y=point["y"], **plot_kwargs["scatter_plot"])
        )

    return fig


# Plot utils
class Projection(Enum):
    SLICE = auto()
    CONTOUR = auto()
    SURFACE = auto()

    @classmethod
    def from_value(cls, value):
        if isinstance(value, str):
            try:
                return cls[value.upper()]
            except ValueError as err:
                raise ValueError(f"Invalid projection : '{value}'") from err
        elif isinstance(value, cls):
            return value
        else:
            raise TypeError(f"Expected str or Projection(Enum), got {type(value)}")

    def is_single(self):
        return self == Projection.SLICE

    def is_multiple(self):
        return self in (Projection.CONTOUR, Projection.SURFACE)

    def __str__(self):
        return f"Projection({self.name})"


def compute_yaxis_range(y, expand_yrange):
    # Calculate expanded y-axis limits based on data range
    y_min, y_max = np.min(y), np.max(y)
    y_range = y_max - y_min
    return [y_min - expand_yrange * y_range, y_max + expand_yrange * y_range]


def update_nested_dict(default, updates):
    # Recursively merge `updates` into `default`
    for k, v in updates.items():
        if isinstance(v, dict) and k in default and isinstance(default[k], dict):
            # If both are dicts, recurse
            default[k] = update_nested_dict(default[k], v)
        else:
            # Otherwise, replace value
            default[k] = v
    return default


def get_layout_kwargs(
    layout_kwargs,
    rows,
    cols,
    return_dict=False,
    template=PLOTLY_TEMPLATE,
    single_plot=False,
):
    """
    Generate a default set of Plotly layout kwargs for subplots.
    Merges user-supplied `layout_kwargs` if provided, overriding defaults.

    Returns:
        dict: kwargs for `Figure.update_layout()`
    """
    if return_dict or single_plot:
        width = 450
        height = 450
    else:
        width = 300 * cols
        height = 300 * rows
    layout_defaults = {
        "width": width,
        "height": height,
        "template": template,
        "xaxis_showgrid": False,
        "yaxis_showgrid": False,
        "title": {"text": "Slice Plot"},
        "legend": {},
    }

    if layout_kwargs:
        layout_defaults.update(layout_kwargs)

    return layout_defaults


def get_make_subplot_kwargs(
    make_subplot_kwargs, rows, cols, projection=None, single_plot=False
):
    """
    Assemble default kwargs for `plotly.subplots.make_subplots`.
    User-supplied `make_subplot_kwargs` override these defaults.

    Returns:
        dict: Kwargs for `make_subplots()`
    """
    make_subplot_defaults = {
        "rows": rows,
        "cols": cols,
        "start_cell": "top-left",
        "print_grid": False,
        "horizontal_spacing": 1 / (cols * 5),
        "vertical_spacing": (1 / max(rows - 1, 1)) / 5,
    }
    if make_subplot_kwargs:
        make_subplot_defaults.update(make_subplot_kwargs)
    if projection and projection.is_multiple():
        make_subplot_defaults["specs"] = [
            [
                {"type": "xy"}
                if row == col and not single_plot
                else {"type": "scene"}
                if projection == Projection.SURFACE
                else {"type": "xy"}
                for col in range(cols)
            ]
            for row in range(rows)
        ]
    return make_subplot_defaults


def get_plot_kwargs(projection, plot_kwargs):
    """
    Generate a default set of Plotly plot kwargs for individual plots.
    Merges user-supplied `plot_kwargs` if provided, overriding defaults.

    Returns:
        dict: kwargs for individual plots
    """
    line_plot_default_kwargs = {
        "color_discrete_sequence": ["#497ea7"],
        "markers": False,
    }
    scatter_plot_default_kwargs = {
        "marker": {
            "color": "#497ea7",
            "size": 4,
        }
    }

    # Update line plot kwargs if present
    if "line_plot" in plot_kwargs:
        plot_kwargs["line_plot"] = update_nested_dict(
            line_plot_default_kwargs, plot_kwargs["line_plot"]
        )
    else:
        plot_kwargs["line_plot"] = line_plot_default_kwargs

    # Update scatter plot kwargs if present
    if "scatter_plot" in plot_kwargs:
        plot_kwargs["scatter_plot"] = update_nested_dict(
            scatter_plot_default_kwargs, plot_kwargs["scatter_plot"]
        )
    else:
        plot_kwargs["scatter_plot"] = scatter_plot_default_kwargs

    if projection == Projection.SURFACE:
        surface_plot_default_kwargs = {
            "colorscale": "Aggrnyl",
            "showscale": False,
            "opacity": 0.8,
        }

        # Update surface plot kwargs if present
        if "surface_plot" in plot_kwargs:
            plot_kwargs["surface_plot"] = update_nested_dict(
                surface_plot_default_kwargs, plot_kwargs["surface_plot"]
            )
        else:
            plot_kwargs["surface_plot"] = surface_plot_default_kwargs

    elif projection == Projection.CONTOUR:
        contour_plot_default_kwargs = {
            "colorscale": "Aggrnyl",
            "showscale": False,
            "line_smoothing": 0.85,
        }

        if "contour_plot" in plot_kwargs:
            plot_kwargs["contour_plot"] = update_nested_dict(
                contour_plot_default_kwargs, plot_kwargs["contour_plot"]
            )
        else:
            plot_kwargs["contour_plot"] = contour_plot_default_kwargs
    return plot_kwargs


def evaluate_kwargs(
    projection,
    size,
    make_subplot_kwargs,
    layout_kwargs,
    plot_kwargs,
    return_dict=False,
    template=None,
):
    """
    Prepare and merge subplot, layout, and plot kwargs based on projection
    type and plot count.
    """
    plot_kwargs = get_plot_kwargs(projection, plot_kwargs)

    if projection.is_single():
        cols = make_subplot_kwargs.get("cols", 1 if size == 1 else 2)
        rows = (size + cols - 1) // cols

        shared_xaxes = make_subplot_kwargs.get("shared_xaxes", True)
        shared_yaxes = make_subplot_kwargs.get("shared_yaxes", True)

        layout_kwargs = get_layout_kwargs(
            layout_kwargs, rows, cols, return_dict=return_dict
        )
        make_subplot_kwargs = get_make_subplot_kwargs(make_subplot_kwargs, rows, cols)
        make_subplot_kwargs["shared_xaxes"] = shared_xaxes
        make_subplot_kwargs["shared_yaxes"] = shared_yaxes
    else:
        if make_subplot_kwargs:
            for key in make_subplot_kwargs.keys():
                if key in ["rows", "cols", "shared_yaxes", "shared_xaxes"]:
                    raise ValueError(
                        f"{key} param is not allowed in plot_kwargs when "
                        f"the projection is {projection.value}."
                    )

        cols = size if size > 2 else 1
        rows = size if size > 2 else 1

        layout_kwargs = get_layout_kwargs(
            layout_kwargs,
            rows,
            cols,
            return_dict=return_dict,
            template=template,
            single_plot=size == 2,
        )
        make_subplot_kwargs = get_make_subplot_kwargs(
            make_subplot_kwargs,
            rows,
            cols,
            projection=projection,
            single_plot=size == 2,
        )

    return make_subplot_kwargs, layout_kwargs, plot_kwargs


def _clean_legend_duplicates(fig):
    """Remove duplicate legend entries from a combined Plotly figure."""
    trace_names = set()

    def disable_legend_if_duplicate(trace):
        if trace.name in trace_names:
            # in this case the legend is a duplicate
            trace.update(showlegend=False)
        else:
            trace_names.add(trace.name)

    fig.for_each_trace(disable_legend_if_duplicate)
    return fig


def combine_plots(plots, make_subplot_kwargs, layout_kwargs, expand_yrange):
    """
    Combine individual subplot figures into one Plotly Figure,
    sharing axes and layout.
    """
    plots = deepcopy(plots)

    # Create a subplot figure
    fig = make_subplots(**make_subplot_kwargs)
    fig.update_layout(**layout_kwargs)

    # Determine rows and cols from make_subplot_kwargs
    # rows = make_subplot_kwargs.get("rows")
    # cols = make_subplot_kwargs.get("cols")

    # Add traces
    for (row_idx, col_idx), subfig in plots.items():
        for trace in subfig.data:
            fig.add_trace(trace, row=row_idx + 1, col=col_idx + 1)

        if hasattr(subfig.layout, "xaxis") and hasattr(subfig.layout.xaxis, "title"):
            fig.update_xaxes(
                title_text=subfig.layout.xaxis.title.text,
                row=row_idx + 1,
                col=col_idx + 1,
            )
        if hasattr(subfig.layout, "yaxis") and hasattr(subfig.layout.yaxis, "title"):
            if (
                "shared_yaxes" in make_subplot_kwargs
                and make_subplot_kwargs["shared_yaxes"]
            ):
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

    # Share y-axis ranges if needed
    if "shared_yaxes" in make_subplot_kwargs and make_subplot_kwargs["shared_yaxes"]:
        all_y = []
        for subfig in plots.values():
            for trace in subfig.data:
                if "y" in trace:
                    all_y.append(np.array(trace["y"]))
        if all_y:
            y_all = np.concatenate(all_y)
            lb, ub = np.min(y_all), np.max(y_all)
            y_range = ub - lb
            y_lower = lb - expand_yrange * y_range
            y_upper = ub + expand_yrange * y_range
            fig.update_yaxes(range=[y_lower, y_upper])

    # Share x-axis ranges if needed
    if "shared_xaxes" in make_subplot_kwargs and make_subplot_kwargs["shared_xaxes"]:
        all_x = []
        for subfig in plots.values():
            for trace in subfig.data:
                if "x" in trace:
                    all_x.append(np.array(trace["x"]))
        if all_x:
            x_all = np.concatenate(all_x)
            lb, ub = np.min(x_all), np.max(x_all)
            fig.update_xaxes(range=[lb, ub])

    # Clean duplicate legends if needed
    fig = _clean_legend_duplicates(fig)

    return fig


# Plot Data
def evaluate_func(params, func, func_kwargs):
    """
    Wrap user function to handle func_kwargs, deprecated dict outputs,
    and enforce return types.
    """
    if func_kwargs:
        func = partial(func, **func_kwargs)

    func_eval = func(params)

    if deprecations.is_dict_output(func_eval):
        warnings.warn(
            "Functions that return dictionaries are deprecated and will "
            "raise an error in future versions.",
            FutureWarning,
        )
        func_eval = deprecations.convert_dict_to_function_value(func_eval)
        func = deprecations.replace_dict_output(func)

    problem_type = (
        deprecations.infer_problem_type_from_dict_output(func_eval)
        if deprecations.is_dict_output(func_eval)
        else infer_aggregation_level(func)
    )
    func_eval = convert_fun_output_to_function_value(func_eval, problem_type)
    func = enforce_return_type(problem_type)(func)
    return func, func_eval


def process_bounds(bounds, lower_bounds, upper_bounds):
    """
    Normalize bounds input, handling deprecated `lower_bounds`/`upper_bounds`
    signatures.
    """
    bounds = replace_and_warn_about_deprecated_bounds(
        bounds=bounds, lower_bounds=lower_bounds, upper_bounds=upper_bounds
    )
    return pre_process_bounds(bounds)


def select_parameter_indices(converter, internal_params, selector):
    """
    Determine which parameter indices to plot, either all or those
    returned by `selector`
    """
    n_params = len(internal_params.values)
    if selector is None:
        return np.arange(n_params, dtype=int)

    helper = converter.params_from_internal(np.arange(n_params))
    registry = get_registry(extended=True)
    selected = np.array(
        tree_just_flatten(selector(helper), registry=registry), dtype=int
    )

    if not np.isfinite(internal_params.lower_bounds[selected]).all():
        raise ValueError("All selected parameters must have finite lower bounds.")
    if not np.isfinite(internal_params.upper_bounds[selected]).all():
        raise ValueError("All selected parameters must have finite upper bounds.")

    return selected


def generate_param_data(internal_params, selected, n_gridpoints):
    """
    Generate a dictionary of parameter values for each selected
    index over `n_gridpoints`
    """
    #
    metadata = {}
    for pos in selected:
        metadata[internal_params.names[pos]] = np.linspace(
            internal_params.lower_bounds[pos],
            internal_params.upper_bounds[pos],
            n_gridpoints,
        )
    return metadata


def evaluate_function_values(
    data, internal, params, func, converter, batch_evaluator, n_cores, projection=None
):
    """
    Batch-evaluate the user function at grid points, returning a flat list
    of scalar values
    """
    if not projection:
        projection = Projection.SLICE

    evaluation_points = generate_evaluation_points(
        data, internal, converter, params, projection
    )
    batch_evaluator = process_batch_evaluator(batch_evaluator)
    results = batch_evaluator(
        func=func,
        arguments=evaluation_points,
        error_handling="continue",
        n_cores=n_cores,
    )
    return [
        float("nan")
        if isinstance(val, str)
        else val.internal_value(AggregationLevel.SCALAR)
        for val in results
    ]


def generate_evaluation_points(data, internal, converter, p_names, projection):
    """
    Build the list of internal parameter vectors to pass to the batch evaluator
    """
    evaluation_points = []
    point = dict(zip(internal.names, internal.values, strict=False))

    if projection.is_single():
        x = data[p_names]

        for p_value in x:
            # updating only the parameter of interest
            point[p_names] = p_value

            values = np.array(list(point.values()))
            evaluation_points.append(converter.params_from_internal(values))
    elif projection.is_multiple():
        x_name, y_name = p_names[0], p_names[1]
        x_vals = data[x_name]
        y_vals = data[y_name]

        x, y = np.meshgrid(x_vals, y_vals)
        x_ravel = x.ravel()
        y_ravel = y.ravel()
        for a, b in zip(x_ravel, y_ravel, strict=False):
            point[x_name] = a
            point[y_name] = b
            values = np.array(list(point.values()))
            evaluation_points.append(converter.params_from_internal(values))
    return evaluation_points
