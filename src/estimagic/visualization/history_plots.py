import numpy as np
import plotly.graph_objects as go
from estimagic.config import PLOTLY_TEMPLATE
from estimagic.optimization.history_tools import get_history_arrays
from estimagic.parameters.tree_registry import get_registry
from pybaum import leaf_names
from pybaum import tree_flatten
from pybaum import tree_just_flatten
from pybaum import tree_unflatten


def criterion_plot(
    res,
    max_evaluations=None,
    template=PLOTLY_TEMPLATE,
    highlight_color="#497ea7",
    base_color="#bab0ac",
    monotone=False,
):
    """Plot the criterion history of an optimization.

    Args:
        res (OptimizeResult): An optimization result with collected history.
        max_evaluations (int): Clip the criterion history after that many entries.
        template (str): A plotly template.
        highlight_color (str): Hex code of the line color.
        base_color (str): Hex code of the line color for local optimizations in a
            multistart optimization.
        monotone (bool): If True, the criterion plot becomes monotone in the sense
            that only that at each iteration the current best criterion value is
            displayed.

    """
    if res.history is None:
        raise ValueError(
            "Criterion_plot requires a optimize_result with history. "
            "Enable history collection by setting collect_history=True "
            "when calling maximize or minimize."
        )

    key = "monotone_criterion" if monotone else "criterion"

    is_multistart = res.multistart_info is not None

    fig = go.Figure()

    if is_multistart:

        scatter_kws = {
            "connectgaps": True,
            "showlegend": False,
        }

        line_kws = {
            "color": base_color,
        }
        for i, opt in enumerate(res.multistart_info["local_optima"]):

            history = get_history_arrays(opt.history, opt.direction)[key]

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

    history = get_history_arrays(res.history, res.direction)[key]

    if max_evaluations is not None and len(history) > max_evaluations:
        history = history[:max_evaluations]

    scatter_kws = {
        "connectgaps": True,
        "showlegend": is_multistart,
    }

    line_kws = {
        "color": highlight_color,
    }

    trace = go.Scatter(
        x=np.arange(len(history)),
        y=history,
        mode="lines",
        name="best result",
        line=line_kws,
        **scatter_kws,
    )
    fig.add_trace(trace)

    fig.update_layout(
        template=template,
        xaxis_title_text="No. of criterion evaluations",
        yaxis_title_text="Criterion value",
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
    if res.history is None:
        raise ValueError(
            "params_plot requires a optimize_result with history. "
            "Enable history collection by setting collect_history=True "
            "when calling maximize or minimize."
        )
    fig = go.Figure()

    history = res.history["params"]

    registry = get_registry(extended=True)

    hist_arr = np.array([tree_just_flatten(p, registry=registry) for p in history]).T
    names = leaf_names(res.params, registry=registry)

    if selector is not None:
        flat, treedef = tree_flatten(res.params, registry=registry)
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
