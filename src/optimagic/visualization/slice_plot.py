import warnings
from functools import partial
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pybaum import tree_just_flatten

import optimagic as om
from optimagic import deprecations
from optimagic.batch_evaluators import (
    BatchEvaluator,
    BatchEvaluatorLiteral,
    process_batch_evaluator,
)
from optimagic.config import DEFAULT_N_CORES, DEFAULT_PALETTE
from optimagic.deprecations import replace_and_warn_about_deprecated_bounds
from optimagic.optimization.fun_value import (
    SpecificFunctionValue,
    convert_fun_output_to_function_value,
    enforce_return_type,
)
from optimagic.parameters.bounds import pre_process_bounds
from optimagic.parameters.conversion import get_converter
from optimagic.parameters.space_conversion import InternalParams
from optimagic.parameters.tree_registry import get_registry
from optimagic.shared.process_user_function import infer_aggregation_level
from optimagic.typing import AggregationLevel, PyTree
from optimagic.visualization.backends import grid_line_plot, line_plot
from optimagic.visualization.plotting_utilities import LineData, MarkerData


def slice_plot(
    func: Callable,
    params: PyTree,
    bounds: om.Bounds | None = None,
    func_kwargs: dict | None = None,
    selector: Callable[[PyTree], PyTree] | None = None,
    n_cores: int = DEFAULT_N_CORES,
    n_gridpoints: int = 20,
    plots_per_row: int = 2,
    param_names: dict[str, str] | None = None,
    share_y: bool = True,
    expand_yrange: float = 0.02,
    share_x: bool = False,
    backend: Literal["plotly", "matplotlib", "bokeh", "altair"] = "plotly",
    template: str | None = None,
    color: str | None = DEFAULT_PALETTE[0],
    title: str | None = None,
    return_dict: bool = False,
    batch_evaluator: BatchEvaluatorLiteral | BatchEvaluator = "joblib",
    # deprecated
    make_subplot_kwargs: dict | None = None,
    lower_bounds: None = None,
    upper_bounds: None = None,
) -> Any:
    """Plot criterion along coordinates at given and random values.

    Generates plots for each parameter and optionally combines them into a figure
    with subplots.

    # TODO: Use soft bounds to create the grid (if available).
    # TODO: Don't do a function evaluation outside the batch evaluator.

    Args:
        func: criterion function that takes params and returns scalar, PyTree or
            FunctionValue object.
        params: A pytree with parameters.
        bounds: Lower and upper bounds on the parameters. The bounds are used to create
            a grid over which slice plots are drawn. The most general and preferred
            way to specify bounds is an `optimagic.Bounds` object that collects lower,
            upper, soft_lower and soft_upper bounds. The soft bounds are not used for
            slice_plots. Each bound type mirrors the structure of params. Check our
            how-to guide on bounds for examples. If params is a flat numpy array, you
            can also provide bounds via any format that is supported by
            scipy.optimize.minimize.
        func_kwargs: Additional keyword arguments passed to func.
        selector: Function that takes params and returns a subset of params for which we
            actually want to generate the plot.
        n_cores: Number of cores.
        n_gridpoints: Number of gridpoints on which the criterion function is evaluated.
            This is the number per plotted line.
        plots_per_row: Number of plots per row.
        param_names: Dictionary mapping old parameter names to new ones.
        share_y: If True, the individual plots share the scale on the yaxis and plots in
            one row actually share the y axis.
        expand_yrange: The ratio by which to expand the range of the (shared) y axis,
            such that the axis is not cropped at exactly max of Criterion Value.
        share_x: If True, set the same range of x axis for all plots and share the
            x axis for all plots in one column.
        backend: The backend to use for plotting. Default is "plotly".
        template: The template for the figure. If not specified, the default template of
            the backend is used. For the 'bokeh' backend, this changes the global theme,
            which affects all Bokeh plots in the session.
        color: The line color.
        title: The figure title. This is not used for the `bokeh` backend, as it does
            not support title for grid plot.
        return_dict: If True, return dictionary with individual plots of each parameter,
            else, combine individual plots into a figure with subplots.
        batch_evaluator: See :ref:`batch_evaluators`.

    Returns:
        The figure object containing the slice plot if `return_dict` is False.
            Otherwise, a dictionary with individual slice plots for each parameter.

    """
    # ==================================================================================
    # Process inputs

    bounds = replace_and_warn_about_deprecated_bounds(
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        bounds=bounds,
    )
    bounds = pre_process_bounds(bounds)

    func, func_eval = _get_processed_func_and_func_eval(func, func_kwargs, params)

    if make_subplot_kwargs is not None:
        deprecations.throw_make_subplot_kwargs_in_slice_plot_future_warning()

    # ==================================================================================
    # Extract backend-agnostic plotting data from results

    plot_data, internal_params = _get_plot_data(
        func=func,
        params=params,
        bounds=bounds,
        func_eval=func_eval,
        selector=selector,
        n_gridpoints=n_gridpoints,
        batch_evaluator=batch_evaluator,
        n_cores=n_cores,
    )

    lines_list, marker_list, xlabels, ylabels = _extract_slice_plot_lines_and_labels(
        plot_data=plot_data,
        internal_params=internal_params,
        func_eval=func_eval,
        param_names=param_names,
        color=color,
    )

    # ==================================================================================
    # Generate the figure

    xrange, yrange = _get_axis_limits(
        plot_data, share_y=share_y, share_x=share_x, expand_yrange=expand_yrange
    )

    if return_dict:
        fig_dict = {}

        for i in range(len(lines_list)):
            fig = line_plot(
                lines=lines_list[i],
                marker=marker_list[i],
                backend=backend,
                xlabel=xlabels[i],
                ylabel=ylabels[i],
                template=template,
            )

            fig_dict[xlabels[i]] = fig

        return fig_dict
    else:
        n_rows = int(np.ceil(len(lines_list) / plots_per_row))

        if share_y:
            ylabels = [
                ylabel if i % plots_per_row == 0 else ""
                for i, ylabel in enumerate(ylabels)
            ]

        fig = grid_line_plot(
            lines_list=lines_list,
            marker_list=marker_list,
            backend=backend,
            n_rows=n_rows,
            n_cols=plots_per_row,
            xlabels=xlabels,
            xrange=xrange,
            share_x=share_x,
            ylabels=ylabels,
            yrange=yrange,
            share_y=share_y,
            template=template,
            height=300 * n_rows,
            width=400 * plots_per_row,
            plot_title=title,
            make_subplot_kwargs=make_subplot_kwargs,
        )
        return fig


def _get_processed_func_and_func_eval(
    func: Callable, func_kwargs: dict | None, params: PyTree
) -> tuple[Callable, SpecificFunctionValue]:
    if func_kwargs is not None:
        func = partial(func, **func_kwargs)
    func_eval = func(params)

    # handle deprecated function output
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

    # Infer the function type and enforce the return type
    if deprecations.is_dict_output(func_eval):
        problem_type = deprecations.infer_problem_type_from_dict_output(func_eval)
    else:
        problem_type = infer_aggregation_level(func)

    func_eval = convert_fun_output_to_function_value(func_eval, problem_type)
    func = enforce_return_type(problem_type)(func)

    return func, func_eval


def _get_plot_data(
    func: Callable,
    params: PyTree,
    bounds: om.Bounds | None,
    func_eval: SpecificFunctionValue,
    selector: Callable[[PyTree], PyTree] | None,
    n_gridpoints: int,
    batch_evaluator: BatchEvaluatorLiteral | BatchEvaluator,
    n_cores: int,
) -> tuple[pd.DataFrame, InternalParams]:
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
        ).ravel()  # Ensure the result is a 1D array

    if not np.isfinite(internal_params.lower_bounds[selected]).all():
        raise ValueError("All selected parameters must have finite lower bounds.")

    if not np.isfinite(internal_params.upper_bounds[selected]).all():
        raise ValueError("All selected parameters must have finite upper bounds.")

    evaluation_points, metadata = [], []
    for pos in selected:
        lb = internal_params.lower_bounds[pos]
        ub = internal_params.upper_bounds[pos]
        grid = np.linspace(lb, ub, n_gridpoints)
        name = internal_params.names[pos]
        for param_value in grid:
            if param_value != internal_params.values[pos]:
                meta = {
                    "name": name,
                    "Parameter Value": param_value,
                }

                x = internal_params.values.copy()
                x[pos] = param_value
                point = converter.params_from_internal(x)
                evaluation_points.append(point)
                metadata.append(meta)

    func_values = _retrieve_func_values(
        func, evaluation_points, batch_evaluator, n_cores
    )
    func_values += [func_eval.internal_value(AggregationLevel.SCALAR)] * len(selected)

    for pos in selected:
        meta = {
            "name": internal_params.names[pos],
            "Parameter Value": internal_params.values[pos],
        }
        metadata.append(meta)

    plot_data = pd.DataFrame(metadata)
    plot_data["Function Value"] = func_values  # type: ignore[assignment]

    return plot_data, internal_params


def _retrieve_func_values(
    func: Callable,
    evaluation_points: list[PyTree],
    batch_evaluator: BatchEvaluatorLiteral | BatchEvaluator,
    n_cores: int,
) -> list[float | NDArray[np.float64]]:
    """Retrieve function values at given evaluation points using batch evaluator."""
    batch_evaluator = process_batch_evaluator(batch_evaluator)

    func_values = batch_evaluator(
        func=func,
        arguments=evaluation_points,
        error_handling="continue",
        n_cores=n_cores,
    )

    # add NaNs where an evaluation failed
    func_values = [
        np.nan if isinstance(val, str) else val.internal_value(AggregationLevel.SCALAR)
        for val in func_values
    ]

    return func_values


def _extract_slice_plot_lines_and_labels(
    plot_data: pd.DataFrame,
    internal_params: InternalParams,
    func_eval: SpecificFunctionValue,
    param_names: dict[str, str] | None,
    color: str | None,
) -> tuple[list[list[LineData]], list[MarkerData], list[str], list[str]]:
    """Extract lines, markers and labels for slice plots."""
    lines_list = []
    marker_list = []
    xlabels = []
    ylabels = []

    for _par_name, _data in plot_data.groupby("name", sort=False):
        df = _data.sort_values("Parameter Value")

        par_name = str(_par_name)
        if param_names is not None and par_name in param_names:
            par_name = param_names[par_name]

        subplot_line = LineData(
            x=df["Parameter Value"].to_numpy(),
            y=df["Function Value"].to_numpy(),
            color=color,
            name=par_name,
            show_in_legend=False,
        )
        lines_list.append([subplot_line])

        if internal_params.names is not None:
            pos = internal_params.names.index(_par_name)
            marker_data = MarkerData(
                x=float(internal_params.values[pos]),
                y=float(func_eval.internal_value(AggregationLevel.SCALAR)),
                color=color,
                show_in_legend=False,
            )
            marker_list.append(marker_data)

        xlabels.append(par_name)
        ylabels.append("Function Value")

    return lines_list, marker_list, xlabels, ylabels


def _get_axis_limits(
    plot_data: pd.DataFrame, share_y: bool, share_x: bool, expand_yrange: float
) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    if share_y:
        lb = plot_data["Function Value"].min()
        ub = plot_data["Function Value"].max()
        y_range = ub - lb
        ub += y_range * expand_yrange
        lb -= y_range * expand_yrange
        yrange = (lb, ub)
    else:
        yrange = None

    if share_x:
        lb = plot_data["Parameter Value"].min()
        ub = plot_data["Parameter Value"].max()
        xrange = (lb, ub)
    else:
        xrange = None

    return xrange, yrange
