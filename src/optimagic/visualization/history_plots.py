import inspect
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import plotly.graph_objects as go
from pybaum import leaf_names, tree_flatten, tree_just_flatten, tree_unflatten

from optimagic.config import PLOTLY_PALETTE, PLOTLY_TEMPLATE
from optimagic.logging.logger import LogReader, SQLiteLogOptions
from optimagic.optimization.algorithm import Algorithm
from optimagic.optimization.history import History
from optimagic.optimization.optimize_result import OptimizeResult
from optimagic.parameters.tree_registry import get_registry
from optimagic.typing import Direction, IterationHistory


def criterion_plot(
    results: list[OptimizeResult | str | Path] | dict[str, OptimizeResult | str | Path],
    names: list[str] | str | None = None,
    max_evaluations: int | None = None,
    template: str = PLOTLY_TEMPLATE,
    palette: list[str] | str = PLOTLY_PALETTE,
    stack_multistart: bool = False,
    monotone: bool = False,
    show_exploration: bool = False,
) -> go.Figure:
    """Plot the criterion history of an optimization.

    Args:
        results: A (list or dict of) optimization results with collected history.
            If dict, then the key is used as the name in a legend.
        names: Names corresponding to res or entries in res.
        max_evaluations: Clip the criterion history after that many entries.
        template: The template for the figure. Default is "plotly_white".
        palette: The coloring palette for traces. Default is "qualitative.Set2".
        stack_multistart: Whether to combine multistart histories into a single history.
            Default is False.
        monotone: If True, the criterion plot becomes monotone in the sense that at each
            iteration the current best criterion value is displayed. Default is False.
        show_exploration: If True, exploration samples of a multistart optimization are
            visualized. Default is False.

    Returns:
        plotly.graph_objs._figure.Figure: The figure.

    """
    # ==================================================================================
    # Process inputs

    if not isinstance(palette, list):
        palette = [palette]
    palette_cycle = itertools.cycle(palette)

    dict_of_optimize_results_or_paths = _harmonize_inputs_to_dict(results, names)

    # ==================================================================================
    # Extract backend-agnostic plotting data from results

    list_of_optimize_data = _retrieve_optimization_data(
        results=dict_of_optimize_results_or_paths,
        stack_multistart=stack_multistart,
        show_exploration=show_exploration,
    )

    plot_data = _extract_criterion_plot_data(
        data=list_of_optimize_data,
        max_evaluations=max_evaluations,
        palette_cycle=palette_cycle,
        stack_multistart=stack_multistart,
        monotone=monotone,
    )

    # ==================================================================================
    # Generate the plotly figure

    plot_config = PlotConfig(
        template=template,
        legend={"yanchor": "top", "xanchor": "right", "y": 0.95, "x": 0.95},
    )

    fig = _plotly_criterion_plot(plot_data, plot_config)
    return fig


def _harmonize_inputs_to_dict(
    results: list[OptimizeResult | str | Path] | dict[str, OptimizeResult | str | Path],
    names: list[str] | str | None,
) -> dict[str, OptimizeResult | str | Path]:
    """Convert all valid inputs for results and names to dict[str, OptimizeResult]."""
    # convert scalar case to list case
    if not isinstance(names, list) and names is not None:
        names = [names]

    if isinstance(results, (OptimizeResult, str, Path)):
        results = [results]

    if names is not None and len(names) != len(results):
        raise ValueError("len(results) needs to be equal to len(names).")

    # handle dict case
    if isinstance(results, dict):
        if names is not None:
            results_dict = dict(zip(names, list(results.values()), strict=False))
        else:
            results_dict = results

    # unlabeled iterable of results
    else:
        if names is None:
            names = list(str(i) for i in range(len(results)))
        results_dict = dict(zip(names, results, strict=False))

    # convert keys to strings
    results_dict = {_convert_key_to_str(k): v for k, v in results_dict.items()}

    return results_dict


def _convert_key_to_str(key: Any) -> str:
    if inspect.isclass(key) and issubclass(key, Algorithm):
        out = str(key.name)
    elif isinstance(key, Algorithm):
        out = str(key.name)
    else:
        out = str(key)
    return out


def params_plot(
    result,
    selector=None,
    max_evaluations=None,
    template=PLOTLY_TEMPLATE,
    show_exploration=False,
):
    """Plot the params history of an optimization.

    Args:
        result (Union[OptimizeResult, pathlib.Path, str]): An optimization results with
            collected history. If dict, then the key is used as the name in a legend.
        selector (callable): A callable that takes params and returns a subset
            of params. If provided, only the selected subset of params is plotted.
        max_evaluations (int): Clip the criterion history after that many entries.
        template (str): The template for the figure. Default is "plotly_white".
        show_exploration (bool): If True, exploration samples of a multistart
            optimization are visualized. Default is False.

    Returns:
        plotly.graph_objs._figure.Figure: The figure.

    """
    # ==================================================================================
    # Process inputs
    # ==================================================================================

    if isinstance(result, OptimizeResult):
        data = _retrieve_optimization_data_from_results_object(
            result,
            stack_multistart=True,
            show_exploration=show_exploration,
            plot_name="params_plot",
        )
        start_params = result.start_params
    elif isinstance(result, (str, Path)):
        data = _retrieve_optimization_data_from_database(
            result,
            stack_multistart=True,
            show_exploration=show_exploration,
        )
        start_params = data.start_params
    else:
        raise TypeError("result must be an OptimizeResult or a path to a log file.")

    if data.stacked_local_histories is not None:
        history = data.stacked_local_histories.params
    else:
        history = data.history.params

    # ==================================================================================
    # Create figure
    # ==================================================================================

    fig = go.Figure()

    registry = get_registry(extended=True)

    hist_arr = np.array([tree_just_flatten(p, registry=registry) for p in history]).T
    names = leaf_names(start_params, registry=registry)

    if selector is not None:
        flat, treedef = tree_flatten(start_params, registry=registry)
        helper = tree_unflatten(treedef, list(range(len(flat))), registry=registry)
        selected = np.array(tree_just_flatten(selector(helper), registry=registry))
        names = [names[i] for i in selected]
        hist_arr = hist_arr[selected]

    for name, data in zip(names, hist_arr, strict=False):
        if max_evaluations is not None and len(data) > max_evaluations:
            plot_data = data[:max_evaluations]
        else:
            plot_data = data

        trace = go.Scatter(
            x=np.arange(len(plot_data)),
            y=plot_data,
            mode="lines",
            name=name,
        )
        fig.add_trace(trace)

    fig.update_layout(
        template=template,
        xaxis_title_text="No. of criterion evaluations",
        yaxis_title_text="Parameter value",
        legend={"yanchor": "top", "xanchor": "right", "y": 0.95, "x": 0.95},
    )

    return fig


def _retrieve_optimization_data(
    results: dict[str, OptimizeResult | str | Path],
    stack_multistart: bool,
    show_exploration: bool,
) -> list["_OptimizeData"]:
    """Retrieve data for criterion plot from results (OptimizeResult or database).

    Args:
        results: A dict of optimization results with collected history.
            The key is used as the name in a legend.
        stack_multistart: Whether to combine multistart histories into a single history.
            Default is False.
        show_exploration: If True, exploration samples of a multistart optimization are
            visualized. Default is False.

    Returns:
        list[_OptimizeData]: A list of _OptimizeData objects containing the history,
            direction, multistart information, and local histories.

    """
    data = []
    for name, res in results.items():
        if isinstance(res, OptimizeResult):
            _data = _retrieve_optimization_data_from_results_object(
                res=res,
                stack_multistart=stack_multistart,
                show_exploration=show_exploration,
                plot_name="criterion_plot",
            )
        elif isinstance(res, (str, Path)):
            _data = _retrieve_optimization_data_from_database(
                res=res,
                stack_multistart=stack_multistart,
                show_exploration=show_exploration,
            )
        else:
            msg = "results must be (or contain) an OptimizeResult or a path to a log"
            f"file, but is type {type(res)}."
            raise TypeError(msg)

        _data.name = name
        data.append(_data)

    return data


def _retrieve_optimization_data_from_results_object(
    res: OptimizeResult,
    stack_multistart: bool,
    show_exploration: bool,
    plot_name: str,
) -> "_OptimizeData":
    """Retrieve optimization data from results object.

    Args:
        res: An optimization results object.
        stack_multistart: Whether to combine multistart histories into a single history.
            Default is False.
        show_exploration: If True, exploration samples of a multistart optimization are
            visualized. Default is False.
        plot_name: Name of the plotting function that calls this function. Used for
            raising errors.

    Returns:
        _OptimizeData: A data object containing the history, direction, multistart
            information, and local histories.

    """
    if res.history is None:
        msg = f"{plot_name} requires an optimize result with history. Enable history "
        "collection by setting collect_history=True when calling maximize or minimize."
        raise ValueError(msg)

    is_multistart = res.multistart_info is not None

    if res.multistart_info:
        local_histories = [
            opt.history
            for opt in res.multistart_info.local_optima
            if opt.history is not None
        ]
    else:
        local_histories = None

    if stack_multistart and local_histories is not None and res.multistart_info:
        stacked = _get_stacked_local_histories(local_histories, res.direction)
        if show_exploration:
            fun = res.multistart_info.exploration_results.tolist()[::-1] + stacked.fun
            params = res.multistart_info.exploration_sample[::-1] + stacked.params

            stacked = History(
                direction=stacked.direction,
                fun=fun,
                params=params,
                # TODO: This needs to be fixed
                start_time=len(fun) * [None],  # type: ignore
                stop_time=len(fun) * [None],  # type: ignore
                batches=len(fun) * [None],  # type: ignore
                task=len(fun) * [None],  # type: ignore
            )
    else:
        stacked = None

    data = _OptimizeData(
        history=res.history,
        direction=Direction(res.direction),
        is_multistart=is_multistart,
        local_histories=local_histories,
        stacked_local_histories=stacked,
    )
    return data


def _retrieve_optimization_data_from_database(
    res: str | Path,
    stack_multistart: bool,
    show_exploration: bool,
) -> "_OptimizeData":
    """Retrieve optimization data from a database.

    Args:
        res: A path to an optimization database.
        stack_multistart: Whether to combine multistart histories into a single history.
            Default is False.
        show_exploration: If True, exploration samples of a multistart optimization are
            visualized. Default is False.

    Returns:
        _OptimizeData: A data object containing the history, direction, multistart
            information, and local histories.

    """
    reader: LogReader = LogReader.from_options(SQLiteLogOptions(res))
    _problem_table = reader.problem_df

    direction = _problem_table["direction"].tolist()[-1]

    multistart_history = reader.read_multistart_history(direction)
    _history = multistart_history.history
    local_histories = multistart_history.local_histories
    exploration = multistart_history.exploration

    if stack_multistart and local_histories is not None:
        stacked = _get_stacked_local_histories(local_histories, direction, _history)
        if show_exploration:
            stacked["params"] = exploration["params"][::-1] + stacked["params"]  # type: ignore
            stacked["criterion"] = exploration["criterion"][::-1] + stacked["criterion"]  # type: ignore
    else:
        stacked = None

    history = History(
        direction=direction,
        fun=_history["fun"],
        params=_history["params"],
        start_time=_history["time"],
        # TODO (@janosg): Retrieve `stop_time` from `hist` once it is available.
        # https://github.com/optimagic-dev/optimagic/pull/553
        stop_time=len(_history["fun"]) * [None],  # type: ignore
        task=len(_history["fun"]) * [None],  # type: ignore
        batches=list(range(len(_history["fun"]))),
    )

    data = _OptimizeData(
        history=history,
        direction=direction,
        is_multistart=local_histories is not None,
        local_histories=local_histories,
        stacked_local_histories=stacked,
        start_params=reader.read_start_params(),
    )
    return data


def _get_stacked_local_histories(
    local_histories: list[History] | list[IterationHistory],
    direction: Any,
    history: History | IterationHistory | None = None,
) -> History:
    """Stack local histories.

    Local histories is a list of dictionaries, each of the same structure. We transform
    this to a dictionary of lists. Finally, when the data is read from the database we
    append the best history at the end.

    """
    stacked: dict[str, list[Any]] = {"criterion": [], "params": [], "runtime": []}
    for hist in local_histories:
        stacked["criterion"].extend(hist.fun)
        stacked["params"].extend(hist.params)
        stacked["runtime"].extend(hist.time)

    # append additional history is necessary
    if history is not None:
        stacked["criterion"].extend(history.fun)
        stacked["params"].extend(history.params)
        stacked["runtime"].extend(history.time)

    return History(
        direction=direction,
        fun=stacked["criterion"],
        params=stacked["params"],
        start_time=stacked["runtime"],
        # TODO (@janosg): Retrieve `stop_time` from `hist` once it is available for the
        # IterationHistory.
        # https://github.com/optimagic-dev/optimagic/pull/553
        stop_time=len(stacked["criterion"]) * [None],  # type: ignore
        task=len(stacked["criterion"]) * [None],  # type: ignore
        batches=list(range(len(stacked["criterion"]))),
    )


def _extract_criterion_plot_data(
    data: list["_OptimizeData"],
    max_evaluations: int | None,
    palette_cycle: Iterator[str],
    stack_multistart: bool,
    monotone: bool,
) -> "CriterionPlotData":
    """Extract lines for criterion plot from data.

    Args:
        data: Data retrieved from results or database.
        max_evaluations: Clip the criterion history after that many entries.
        palette_cycle: Cycle of colors for plotting.
        stack_multistart: Whether to combine multistart histories into a single
            history. Default is False.
        monotone: If True, the criterion plot becomes monotone in the sense that at each
            iteration the current best criterion value is displayed.

    Returns:
        CriterionPlotData: A data object containing the lines for the plot.

    """
    fun_or_monotone_fun = "monotone_fun" if monotone else "fun"

    # Collect multistart optimization paths
    multistart_lines: list[LineData] = []

    plot_multistart = len(data) == 1 and data[0].is_multistart and not stack_multistart

    if plot_multistart and data[0].local_histories:
        for i, local_history in enumerate(data[0].local_histories):
            history = getattr(local_history, fun_or_monotone_fun)

            if max_evaluations is not None and len(history) > max_evaluations:
                history = history[:max_evaluations]

            line_data = LineData(
                x=np.arange(len(history)),
                y=history,
                color="#bab0ac",
                name=str(i),
                show_in_legend=False,
            )
            multistart_lines.append(line_data)

    # Collect main optimization paths
    lines: list[LineData] = []

    for _data in data:
        if stack_multistart and _data.stacked_local_histories is not None:
            _history = _data.stacked_local_histories
        else:
            _history = _data.history

        history = getattr(_history, fun_or_monotone_fun)

        if max_evaluations is not None and len(history) > max_evaluations:
            history = history[:max_evaluations]

        _color = next(palette_cycle)
        if not isinstance(_color, str):
            msg = "highlight_palette needs to be a string or list of strings, but its "
            f"entry is of type {type(_color)}."
            raise TypeError(msg)

        line_data = LineData(
            x=np.arange(len(history)),
            y=history,
            color=_color,
            name="best result" if plot_multistart else _data.name,
            show_in_legend=not plot_multistart,
        )
        lines.append(line_data)

    plot_data = CriterionPlotData(
        lines=lines,
        multistart_lines=multistart_lines,
    )
    return plot_data


@dataclass()
class _OptimizeData:
    history: History
    direction: Direction
    is_multistart: bool
    local_histories: list[History] | list[IterationHistory] | None
    stacked_local_histories: History | None
    start_params: list[Any] | None = None
    name: str | None = None


@dataclass(frozen=True)
class LineData:
    """Data of a single line.

    Attributes:
        x: The x-coordinates of the points.
        y: The y-coordinates of the points.
        color: The color of the line. Default is None.
        name: The name of the line. Default is None.
        show_in_legend: Whether to show the line in the legend. Default is True.

    """

    x: np.ndarray
    y: np.ndarray
    color: str | None = None
    name: str | None = None
    show_in_legend: bool = True


@dataclass(frozen=True)
class CriterionPlotData:
    """Backend agnostic data for criterion plot.

    Attributes:
        lines: Main optimization paths.
        multistart_lines: Multistart optimization paths, if applicable.

    """

    lines: list[LineData]
    multistart_lines: list[LineData]


@dataclass(frozen=True)
class PlotConfig:
    """Configuration settings for figure.

    Attributes:
        template: The template for the figure.
        legend: Configuration for the legend.

    """

    template: str
    legend: dict[str, Any]


def _plotly_criterion_plot(
    plot_data: CriterionPlotData, plot_config: PlotConfig
) -> go.Figure:
    """Create a plotly figure from the plot data and configuration."""

    fig = go.Figure()

    for line in plot_data.multistart_lines + plot_data.lines:
        trace = go.Scatter(
            x=line.x,
            y=line.y,
            name=line.name,
            mode="lines",
            line_color=line.color,
            showlegend=line.show_in_legend,
            connectgaps=True,
        )
        fig.add_trace(trace)

    fig.update_layout(
        template=plot_config.template,
        xaxis_title_text="No. of criterion evaluations",
        yaxis_title_text="Criterion value",
        legend=plot_config.legend,
    )

    return fig
