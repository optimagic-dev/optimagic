import inspect
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
from pybaum import leaf_names, tree_flatten, tree_just_flatten, tree_unflatten

from optimagic.config import PLOTLY_TEMPLATE
from optimagic.logging.logger import LogReader, SQLiteLogOptions
from optimagic.optimization.algorithm import Algorithm
from optimagic.optimization.history import History
from optimagic.optimization.optimize_result import OptimizeResult
from optimagic.parameters.tree_registry import get_registry
from optimagic.typing import IterationHistory, PyTree
from optimagic.visualization.backends import get_plot_backend_class
from optimagic.visualization.plotting_utilities import LineData, get_palette_cycle

BACKEND_TO_CRITERION_PLOT_LEGEND_PROPERTIES: dict[str, dict[str, Any]] = {
    "plotly": {
        "yanchor": "top",
        "xanchor": "right",
        "y": 0.95,
        "x": 0.95,
    },
    "matplotlib": {
        "loc": "upper right",
    },
}


ResultOrPath = OptimizeResult | str | Path


def criterion_plot(
    results: ResultOrPath | list[ResultOrPath] | dict[str, ResultOrPath],
    names: list[str] | str | None = None,
    max_evaluations: int | None = None,
    backend: str = "plotly",
    template: str | None = None,
    palette: list[str] | str | None = None,
    stack_multistart: bool = False,
    monotone: bool = False,
    show_exploration: bool = False,
) -> Any:
    """Plot the criterion history of an optimization.

    Args:
        results: An optimization result (or list of, or dict of results) with collected
            history, or path(s) to it. If dict, then the key is used as the name in the
            legend.
        max_evaluations: Clip the criterion history after that many entries.
        backend: The backend to use for plotting. Default is "plotly".
        template: The template for the figure. Default is "plotly_white".
        palette: The coloring palette for traces. Default is "qualitative.Set2".
        stack_multistart: Whether to combine multistart histories into a single history.
            Default is False.
        monotone: If True, the criterion plot becomes monotone in the sense that at each
            iteration the current best criterion value is displayed. Default is False.
        show_exploration: If True, exploration samples of a multistart optimization are
            visualized. Default is False.

    Returns:
        The figure object containing the criterion plot.

    """
    # ==================================================================================
    # Get Plot Backend class

    plot_cls = get_plot_backend_class(backend)

    # ==================================================================================
    # Process inputs

    if palette is None:
        palette = plot_cls.default_palette
    palette_cycle = get_palette_cycle(palette)

    dict_of_optimize_results_or_paths = _harmonize_inputs_to_dict(results, names)

    # ==================================================================================
    # Extract backend-agnostic plotting data from results

    list_of_optimize_data = _retrieve_optimization_data(
        results=dict_of_optimize_results_or_paths,
        stack_multistart=stack_multistart,
        show_exploration=show_exploration,
    )

    lines, multistart_lines = _extract_criterion_plot_lines(
        data=list_of_optimize_data,
        max_evaluations=max_evaluations,
        palette_cycle=palette_cycle,
        stack_multistart=stack_multistart,
        monotone=monotone,
    )

    # ==================================================================================
    # Generate the figure

    plot = plot_cls(template)

    plot.add_lines(lines + multistart_lines)
    plot.set_labels(xlabel="No. of criterion evaluations", ylabel="Criterion value")
    plot.set_legend_properties(BACKEND_TO_CRITERION_PLOT_LEGEND_PROPERTIES[backend])

    return plot.figure


def _harmonize_inputs_to_dict(
    results: ResultOrPath | list[ResultOrPath] | dict[str, ResultOrPath],
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
            names = [str(i) for i in range(len(results))]
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


@dataclass(frozen=True)
class _PlottingMultistartHistory:
    """Data container for an optimization history and metadata. Contains local histories
    in case of multistart optimization.

    This dataclass is only used internally.

    """

    history: History
    name: str | None
    start_params: PyTree
    is_multistart: bool
    local_histories: list[History] | list[IterationHistory] | None
    stacked_local_histories: History | None


def _retrieve_optimization_data(
    results: dict[str, OptimizeResult | str | Path],
    stack_multistart: bool,
    show_exploration: bool,
) -> list[_PlottingMultistartHistory]:
    """Retrieve data for criterion plot from results (OptimizeResult or database).

    Args:
        results: A dict of optimization results with collected history.
            The key is used as the name in a legend.
        stack_multistart: Whether to combine multistart histories into a single history.
            Default is False.
        show_exploration: If True, exploration samples of a multistart optimization are
            visualized. Default is False.

    Returns:
        A list of objects containing the history, metadata, and local histories of each
            optimization result.

    """
    data = []
    for name, res in results.items():
        if isinstance(res, OptimizeResult):
            _data = _retrieve_optimization_data_from_results_object(
                res=res,
                stack_multistart=stack_multistart,
                show_exploration=show_exploration,
                plot_name="criterion_plot",
                res_name=name,
            )
        elif isinstance(res, (str, Path)):
            _data = _retrieve_optimization_data_from_database(
                res=res,
                stack_multistart=stack_multistart,
                show_exploration=show_exploration,
                res_name=name,
            )
        else:
            msg = (
                "results must be (or contain) an OptimizeResult or a path to a log "
                f"file, but is type {type(res)}."
            )
            raise TypeError(msg)

        data.append(_data)

    return data


def _retrieve_optimization_data_from_results_object(
    res: OptimizeResult,
    stack_multistart: bool,
    show_exploration: bool,
    plot_name: str,
    res_name: str | None = None,
) -> _PlottingMultistartHistory:
    """Retrieve optimization data from results object.

    Args:
        res: An optimization results object.
        stack_multistart: Whether to combine multistart histories into a single history.
            Default is False.
        show_exploration: If True, exploration samples of a multistart optimization are
            visualized. Default is False.
        plot_name: Name of the plotting function that calls this function. Used for
            raising errors.
        res_name: Name of the result.

    Returns:
        A data object containing the history, metadata, and local histories of the
            optimization result.

    """
    if res.history is None:
        msg = f"{plot_name} requires an optimize result with history. Enable history "
        "collection by setting collect_history=True when calling maximize or minimize."
        raise ValueError(msg)

    if res.multistart_info:
        local_histories = [
            opt.history
            for opt in res.multistart_info.local_optima
            if opt.history is not None
        ]

        if stack_multistart:
            stacked = _get_stacked_local_histories(local_histories, res.direction)
            if show_exploration:
                fun = res.multistart_info.exploration_results[::-1] + stacked.fun
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
    else:
        local_histories = None
        stacked = None

    data = _PlottingMultistartHistory(
        history=res.history,
        name=res_name,
        start_params=res.start_params,
        is_multistart=res.multistart_info is not None,
        local_histories=local_histories,
        stacked_local_histories=stacked,
    )
    return data


def _retrieve_optimization_data_from_database(
    res: str | Path,
    stack_multistart: bool,
    show_exploration: bool,
    res_name: str | None = None,
) -> _PlottingMultistartHistory:
    """Retrieve optimization data from a database.

    Args:
        res: A path to an optimization database.
        stack_multistart: Whether to combine multistart histories into a single history.
            Default is False.
        show_exploration: If True, exploration samples of a multistart optimization are
            visualized. Default is False.
        res_name: Name of the result.

    Returns:
        A data object containing the history, metadata, and local histories of the
            optimization result.

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

    data = _PlottingMultistartHistory(
        history=history,
        name=res_name,
        start_params=reader.read_start_params(),
        is_multistart=local_histories is not None,
        local_histories=local_histories,
        stacked_local_histories=stacked,
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


def _extract_criterion_plot_lines(
    data: list[_PlottingMultistartHistory],
    max_evaluations: int | None,
    palette_cycle: "itertools.cycle[str]",
    stack_multistart: bool,
    monotone: bool,
) -> tuple[list[LineData], list[LineData]]:
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
        Tuple containing
            - lines: Main optimization paths.
            - multistart_lines: Multistart optimization paths.

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

        line_data = LineData(
            x=np.arange(len(history)),
            y=history,
            color=next(palette_cycle),
            name="best result" if plot_multistart else _data.name,
            show_in_legend=not plot_multistart,
        )
        lines.append(line_data)

    return lines, multistart_lines
