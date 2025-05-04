import warnings
from functools import partial

import numpy as np
from pybaum import tree_just_flatten

from optimagic import deprecations
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
from optimagic.visualization.plot_data import *
from optimagic.visualization.plot_utils import *


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
    """Generate interactive slice, contour or surface plots of a function over its
    parameters.

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
              - rows, cols computed from a number of parameters and projection
              - start_cell='top-left', print_grid=False
              - horizontal_spacing=1/(cols*5), vertical_spacing=(1/(max(rows-1,1)))/5
              - If projection is contour or surface, `specs` grid matching types are
                 added.
        layout_kwargs (dict or None): kwargs for figure layout update. Default: None.
            Internal defaults when None:
              - width, height = 450 (single plot) or 300 × cols by 300 × rows
              - template = "plotly" (multi‐parameter) or DEFAULT PLOTLY_TEMPLATE
              - xaxis_showgrid=False, yaxis_showgrid=False
        plot_kwargs (dict or None): Nested dict of trace‐level kwargs. Default: None.
            Internal defaults when None:
              - line_plot: {'color_discrete_sequence':['#497ea7'], 'markers': False}
              - scatter_plot: {'marker':{'color':'#497ea7','size':4}}
              - surface_plot (if projection="surface"):
                    {'colorscale':'Aggrnyl','showscale':False,'opacity':0.8}
              - contour_plot (if projection="contour"):
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
        )

    if not np.isfinite(internal_params.lower_bounds[selected]).all():
        raise ValueError("All selected parameters must have finite lower bounds.")

    if not np.isfinite(internal_params.upper_bounds[selected]).all():
        raise ValueError("All selected parameters must have finite upper bounds.")

    param_data = {}
    for pos in selected:
        param_data[internal_params.names[pos]] = np.linspace(
            internal_params.lower_bounds[pos],
            internal_params.upper_bounds[pos],
            n_gridpoints,
        )

    projection = Projection(projection)
    template = "plotly" if not projection.is_univariate() else PLOTLY_TEMPLATE

    # Initialize plot, subplot and layout kwargs
    if plot_kwargs is None:
        plot_kwargs = {}
    if make_subplot_kwargs is None:
        make_subplot_kwargs = {}
    if layout_kwargs is None:
        layout_kwargs = {}

    selected_size = len(selected)

    if not projection.is_univariate() and selected_size < 2:
        raise ValueError(
            f"{projection!r} requires at least two parameters. Got {selected_size}."
        )

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
    if projection.is_univariate():
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
