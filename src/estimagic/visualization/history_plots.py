import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from estimagic.config import PLOTLY_TEMPLATE
from estimagic.logging.read_log import OptimizeLogReader
from estimagic.logging.read_log import read_optimization_problem_table
from estimagic.optimization.history_tools import get_history_arrays
from estimagic.optimization.optimize_result import OptimizeResult
from estimagic.parameters.tree_registry import get_registry
from pybaum import leaf_names
from pybaum import tree_flatten
from pybaum import tree_just_flatten
from pybaum import tree_unflatten


@dataclass
class PlottingData:
    history: dict
    direction: str
    is_multistart: bool
    local_histories: dict = None
    stacked_local_histories: dict = None


def criterion_plot(
    res,
    names=None,
    max_evaluations=None,
    template=PLOTLY_TEMPLATE,
    highlight_palette=px.colors.qualitative.T10,
    base_color="#bab0ac",
    stack_multistart=False,
    monotone=False,
):
    """Plot the criterion history of an optimization.

    Args:
        res (Union[List, Dict][Union[OptimizeResult, pathlib.Path, str]): A (list or
            dict of) optimization results with collected history. If dict, then the
            key is used as the name in a legend.
        names (Union[List[str], str]): Names corresponding to res or entries in res.
        max_evaluations (int): Clip the criterion history after that many entries.
        template (str): A plotly template.
        highlight_palette (Union[List[str], str]): Hex codes of the line color.
        base_color (str): Hex code of the line color for local optimizations in a
            multistart optimization.
        stack_multistart (bool): Whether to combine multistart histories into a single
            history. Default False.
        monotone (bool): If True, the criterion plot becomes monotone in the sense
            that only that at each iteration the current best criterion value is
            displayed.

    Returns:
        plotly.graph_objs._figure.Figure: The figure.

    """
    # ==================================================================================
    # Process inputs
    # ==================================================================================

    if stack_multistart and monotone:
        raise ValueError("If stack_multistart is True, monotone needs to be False.")

    if not isinstance(res, dict):
        if isinstance(res, list):
            names = range(len(res)) if names is None else names
            if len(names) != len(res):
                raise ValueError("len(res) needs to be equal to len(names).")
            res = dict(zip(range(len(res)), res))
        else:
            name = 0 if names is None else names
            if isinstance(name, list):
                if len(name) > 1:
                    raise ValueError("len(res) needs to be equal to len(names).")
                else:
                    name = name[0]
            res = {name: res}

    if not isinstance(highlight_palette, list):
        highlight_palette = [highlight_palette]
    highlight_palette = itertools.cycle(highlight_palette)

    key = "monotone_criterion" if monotone else "criterion"

    # ==================================================================================
    # Extract plotting data from results objects / data base
    # ==================================================================================

    data = {}
    for name, _res in res.items():

        if isinstance(_res, OptimizeResult):
            _data = _extract_plotting_data_from_results_object(_res)
        elif isinstance(_res, (str, Path)):
            _data = _extract_plotting_data_from_data_base(_res)
        else:
            msg = "res must be an OptimizeResult or a path to a log file, but is type "
            f" {type(_res)}."
            raise ValueError(msg)

        data[name] = _data

    # ==================================================================================
    # Create figure
    # ==================================================================================

    fig = go.Figure()

    plot_multistart = (
        len(data) == 1 and list(data.values())[0].is_multistart and not stack_multistart
    )

    # ==================================================================================
    # Plot multistart paths

    if plot_multistart:

        scatter_kws = {
            "connectgaps": True,
            "showlegend": False,
        }

        line_kws = {
            "color": base_color,
        }

        for i, local_history in enumerate(data[0].local_histories):

            history = get_history_arrays(local_history, data[0].direction)[key]

            if max_evaluations is not None and len(history) > max_evaluations:
                history = history[:max_evaluations]

            trace = go.Scatter(
                x=np.arange(len(history)),
                y=history,
                mode="lines",
                name=str(i),
                line=line_kws,
                **scatter_kws,
            )
            fig.add_trace(trace)

    # ==================================================================================
    # Plot different optimization objects

    for name, _data in data.items():

        if stack_multistart and _data.stacked_local_histories is not None:
            _history = _data.stacked_local_histories
        else:
            _history = _data.history
        history = get_history_arrays(_history, _data.direction)[key]

        if max_evaluations is not None and len(_data.history) > max_evaluations:
            history = history[:max_evaluations]

        scatter_kws = {
            "connectgaps": True,
            "showlegend": not plot_multistart,
        }

        _color = next(highlight_palette)
        if not isinstance(_color, str):
            msg = "highlight_palette needs to be a string or list of strings, but its "
            f"entry is of type {type(_color)}."
            raise ValueError(msg)

        line_kws = {
            "color": _color,
        }

        trace = go.Scatter(
            x=np.arange(len(history)),
            y=history,
            mode="lines",
            name="best result" if plot_multistart else name,
            line=line_kws,
            **scatter_kws,
        )
        fig.add_trace(trace)

    fig.update_layout(
        template=template,
        xaxis_title_text="No. of criterion evaluations",
        yaxis_title_text="Criterion value",
        legend={"yanchor": "top", "xanchor": "right", "y": 0.95, "x": 0.95},
    )
    return fig


def params_plot(
    res,
    selector=None,
    max_evaluations=None,
    template=PLOTLY_TEMPLATE,
):
    """Plot the params history of an optimization.

    Args:
        res (OptimizeResult): An optimization result with collected history.
        selector (callable): A callable that takes params and returns a subset
            of params. If provided, only the selected subset of params is plotted.
        max_evaluations (int): Clip the criterion history after that many entries.
        template (str): A plotly template.

    """
    if isinstance(res, OptimizeResult):
        if res.history is None:
            raise ValueError(
                "params_plot requires a optimize_result with history. "
                "Enable history collection by setting collect_history=True "
                "when calling maximize or minimize."
            )
        history = res.history["params"]
        start_params = res.start_params
    elif isinstance(res, (str, Path)):
        reader = OptimizeLogReader(res)
        start_params = reader.read_start_params()
        history = reader.read_history()["params"]
    else:
        raise ValueError("res must be an OptimizeResult or a path to a log file.")

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

    for name, data in zip(names, hist_arr):
        if max_evaluations is not None and len(data) > max_evaluations:
            data = data[:max_evaluations]

        trace = go.Scatter(
            x=np.arange(len(data)),
            y=data,
            mode="lines",
            name=name,
        )
        fig.add_trace(trace)

    fig.update_layout(
        template=template,
        xaxis_title_text="No. of criterion evaluations",
        yaxis_title_text="Parameter value",
    )

    return fig


def _extract_plotting_data_from_results_object(res):

    if res.history is None:
        raise ValueError(
            "Criterion_plot requires a optimize_result with history. "
            "Enable history collection by setting collect_history=True "
            "when calling maximize or minimize."
        )

    is_multistart = res.multistart_info is not None

    data = PlottingData(
        history=res.history,
        direction=res.direction,
        is_multistart=is_multistart,
    )

    if res.multistart_info is not None:
        data.local_histories = [
            opt.history for opt in res.multistart_info["local_optima"]
        ]
        data.stacked_local_histories = _get_stacked_local_histories(
            data.local_histories
        )

    return data


def _extract_plotting_data_from_data_base(res):

    reader = OptimizeLogReader(res)
    _problem_table = read_optimization_problem_table(res)

    direction = _problem_table["direction"].tolist()[-1]

    history, local_histories = reader.read_multistart_history(direction=direction)

    data = PlottingData(
        history=history,
        direction=direction,
        is_multistart=local_histories is not None,
        local_histories=local_histories,
    )

    if local_histories is not None:
        data.stacked_local_histories = _get_stacked_local_histories(local_histories)

    return data


def _get_stacked_local_histories(local_histories):
    stacked = {key: [h[key] for h in local_histories] for key in local_histories[0]}
    stacked = {key: list(itertools.chain(*value)) for key, value in stacked.items()}
    return stacked
