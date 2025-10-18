import itertools
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast, runtime_checkable

import numpy as np
import plotly.graph_objects as go

from optimagic.config import IS_MATPLOTLIB_INSTALLED
from optimagic.exceptions import InvalidPlottingBackendError, NotInstalledError
from optimagic.visualization.plotting_utilities import LineData

if TYPE_CHECKING:
    import matplotlib.pyplot as plt


@runtime_checkable
class LinePlotFunction(Protocol):
    def __call__(
        self,
        lines: list[LineData],
        *,
        title: str | None,
        xlabel: str | None,
        ylabel: str | None,
        template: str | None,
        height: int | None,
        width: int | None,
        legend_properties: dict[str, Any] | None,
        margin_properties: dict[str, Any] | None,
        horizontal_line: float | None,
        subplot: Any | None = None,
    ) -> Any: ...


@runtime_checkable
class GridLinePlotFunction(Protocol):
    def __call__(
        self,
        lines_list: list[list[LineData]],
        *,
        n_rows: int,
        n_cols: int,
        titles: list[str] | None,
        xlabel: str | None,
        ylabel: str | None,
        template: str | None,
        height: int | None,
        width: int | None,
        legend_properties: dict[str, Any] | None,
        margin_properties: dict[str, Any] | None,
    ) -> Any: ...


def _line_plot_plotly(
    lines: list[LineData],
    *,
    title: str | None,
    xlabel: str | None,
    ylabel: str | None,
    template: str | None,
    height: int | None,
    width: int | None,
    legend_properties: dict[str, Any] | None,
    margin_properties: dict[str, Any] | None,
    horizontal_line: float | None,
    subplot: tuple[go.Figure, int, int] | None = None,
) -> go.Figure:
    if template is None:
        template = "simple_white"

    if subplot is None:
        fig = go.Figure()
        row, col = None, None

    else:
        fig, row, col = subplot

    fig.update_layout(
        title=title,
        template=template,
        height=height,
        width=width,
        legend=legend_properties,
        margin=margin_properties,
    )
    fig.update_xaxes(
        title=xlabel.format(linebreak="<br>") if xlabel else None, row=row, col=col
    )
    fig.update_yaxes(
        title=ylabel.format(linebreak="<br>") if ylabel else None, row=row, col=col
    )

    if horizontal_line is not None:
        fig.add_hline(
            y=horizontal_line,
            line_width=fig.layout.yaxis.linewidth or 1,
            opacity=1.0,
            row=row,
            col=col,
        )

    for line in lines:
        trace = go.Scatter(
            x=line.x,
            y=line.y,
            name=line.name,
            line_color=line.color,
            mode="lines",
            showlegend=line.show_in_legend,
            legendgroup=line.name,
        )
        fig.add_trace(trace, row=row, col=col)

    return fig


def _grid_line_plot_plotly(
    lines_list: list[list[LineData]],
    *,
    n_rows: int,
    n_cols: int,
    titles: list[str] | None,
    xlabel: str | None,
    ylabel: str | None,
    template: str | None,
    height: int | None,
    width: int | None,
    legend_properties: dict[str, Any] | None,
    margin_properties: dict[str, Any] | None,
) -> go.Figure:
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=titles,
    )

    for lines, (row, col) in zip(
        lines_list,
        itertools.product(range(1, n_rows + 1), range(1, n_cols + 1)),
        strict=False,
    ):
        _line_plot_plotly(
            lines,
            title=None,
            xlabel=xlabel,
            ylabel=ylabel,
            template=template,
            height=height,
            width=width,
            legend_properties=legend_properties,
            margin_properties=margin_properties,
            horizontal_line=None,
            subplot=(fig, row, col),
        )

    return fig


def _line_plot_matplotlib(
    lines: list[LineData],
    *,
    title: str | None,
    xlabel: str | None,
    ylabel: str | None,
    template: str | None,
    height: int | None,
    width: int | None,
    legend_properties: dict[str, Any] | None,
    margin_properties: dict[str, Any] | None,
    horizontal_line: float | None,
    subplot: "plt.Axes | None" = None,
) -> "plt.Axes":
    import matplotlib.pyplot as plt

    # In interactive environments (like Jupyter), explicitly enable matplotlib's
    # interactive mode. If it is not enabled, matplotlib's context manager will
    # revert to non-interactive mode after creating the first figure, causing
    # subsequent figures to not display inline.
    # See: https://github.com/matplotlib/matplotlib/issues/26716
    if plt.get_backend() == "module://matplotlib_inline.backend_inline":
        plt.ion()

    if template is None:
        template = "default"

    with plt.style.context(template):
        if subplot is None:
            px = 1 / plt.rcParams["figure.dpi"]  # pixel in inches
            fig, ax = plt.subplots(
                figsize=(width * px, height * px) if width and height else None,
                layout="constrained",
            )
        else:
            ax = subplot

        if horizontal_line is not None:
            ax.axhline(
                y=horizontal_line,
                color=ax.spines["left"].get_edgecolor() or "gray",
                linewidth=ax.spines["left"].get_linewidth() or 1.0,
            )

        for line in lines:
            ax.plot(
                line.x,
                line.y,
                label=line.name if line.show_in_legend else None,
                color=line.color,
            )

        ax.set(
            title=title,
            xlabel=xlabel.format(linebreak="\n") if xlabel else None,
            ylabel=ylabel.format(linebreak="\n") if ylabel else None,
        )

        if legend_properties is not None:
            fig.legend(**legend_properties)

    return ax


def _grid_line_plot_matplotlib(
    lines_list: list[list[LineData]],
    *,
    n_rows: int,
    n_cols: int,
    titles: list[str] | None,
    xlabel: str | None,
    ylabel: str | None,
    template: str | None,
    height: int | None,
    width: int | None,
    legend_properties: dict[str, Any] | None,
    margin_properties: dict[str, Any] | None,
) -> np.ndarray:
    import matplotlib.pyplot as plt

    px = 1 / plt.rcParams["figure.dpi"]  # pixel in inches
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        squeeze=False,
        figsize=(width * px, height * px) if width and height else None,
        layout="constrained",
    )

    for i, (row, col) in enumerate(itertools.product(range(n_rows), range(n_cols))):
        if i >= len(lines_list):
            axes[row, col].set_visible(False)
            continue

        _line_plot_matplotlib(
            lines_list[i],
            title=titles[i] if titles else None,
            xlabel=xlabel,
            ylabel=ylabel,
            template=template,
            height=None,
            width=None,
            legend_properties=None,
            margin_properties=None,
            horizontal_line=None,
            subplot=axes[row, col],
        )

    fig.legend(**legend_properties or {})

    return axes


def line_plot(
    lines: list[LineData],
    backend: Literal["plotly", "matplotlib"] = "plotly",
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    template: str | None = None,
    height: int | None = None,
    width: int | None = None,
    legend_properties: dict[str, Any] | None = None,
    margin_properties: dict[str, Any] | None = None,
    horizontal_line: float | None = None,
) -> Any:
    """Create a line plot corresponding to the specified backend.

    Args:
        lines: List of objects each containing data for a line in the plot.
            The order of lines in the list determines the order in which they are
            plotted, with later lines being rendered on top of earlier ones.
        backend: The backend to use for plotting.
        title: Title of the plot.
        xlabel: Label for the x-axis.
        ylabel: Label for the y-axis.
        template: Backend-specific template for styling the plot.
        height: Height of the plot (in pixels).
        width: Width of the plot (in pixels).
        legend_properties: Backend-specific properties for the legend.
        margin_properties: Backend-specific properties for the plot margins.
        horizontal_line: If provided, a horizontal line is drawn at the specified
            y-value.

    Returns:
        A figure object corresponding to the specified backend.

    """
    _line_plot_backend_function = cast(
        LinePlotFunction, _get_plot_function(backend, grid_plot=False)
    )

    fig = _line_plot_backend_function(
        lines,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        template=template,
        height=height,
        width=width,
        legend_properties=legend_properties,
        margin_properties=margin_properties,
        horizontal_line=horizontal_line,
    )

    return fig


def grid_line_plot(
    lines_list: list[list[LineData]],
    backend: Literal["plotly", "matplotlib"] = "plotly",
    *,
    n_rows: int,
    n_cols: int,
    titles: list[str] | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    template: str | None = None,
    height: int | None = None,
    width: int | None = None,
    legend_properties: dict[str, Any] | None = None,
    margin_properties: dict[str, Any] | None = None,
) -> Any:
    """Create a grid of line plots corresponding to the specified backend.

    Args:
        lines_list: A list where each element is a list of objects containing data
            for the lines in a subplot. The order of sublists determines the order
            of subplots in the grid (row-wise), and the order of lines within each
            sublist determines the order of lines in that subplot.
        backend: The backend to use for plotting.
        n_rows: Number of rows in the grid.
        n_cols: Number of columns in the grid.
        titles: Titles for each subplot in the grid.
        xlabel: Label for the x-axis of each subplot.
        ylabel: Label for the y-axis of each subplot.
        template: Backend-specific template for styling the plots.
        height: Height of the entire grid plot (in pixels).
        width: Width of the entire grid plot (in pixels).
        legend_properties: Backend-specific properties for the legend.
        margin_properties: Backend-specific properties for the plot margins.

    Returns:
        A figure object corresponding to the specified backend.

    """
    _grid_line_plot_backend_function = cast(
        GridLinePlotFunction, _get_plot_function(backend, grid_plot=True)
    )

    fig = _grid_line_plot_backend_function(
        lines_list,
        n_rows=n_rows,
        n_cols=n_cols,
        titles=titles,
        xlabel=xlabel,
        ylabel=ylabel,
        template=template,
        height=height,
        width=width,
        legend_properties=legend_properties,
        margin_properties=margin_properties,
    )

    return fig


BACKEND_AVAILABILITY_AND_LINE_PLOT_FUNCTION: dict[
    str, tuple[bool, LinePlotFunction, GridLinePlotFunction]
] = {
    "plotly": (True, _line_plot_plotly, _grid_line_plot_plotly),
    "matplotlib": (
        IS_MATPLOTLIB_INSTALLED,
        _line_plot_matplotlib,
        _grid_line_plot_matplotlib,
    ),
}


def _get_plot_function(
    backend: str, grid_plot: bool
) -> LinePlotFunction | GridLinePlotFunction:
    if backend not in BACKEND_AVAILABILITY_AND_LINE_PLOT_FUNCTION:
        msg = (
            f"Invalid plotting backend '{backend}'. "
            f"Available backends: "
            f"{', '.join(BACKEND_AVAILABILITY_AND_LINE_PLOT_FUNCTION.keys())}"
        )
        raise InvalidPlottingBackendError(msg)

    (
        _is_backend_available,
        _line_plot_backend_function,
        _grid_line_plot_backend_function,
    ) = BACKEND_AVAILABILITY_AND_LINE_PLOT_FUNCTION[backend]

    if not _is_backend_available:
        msg = (
            f"The {backend} backend is not installed. "
            f"Install the package using either 'pip install {backend}' or "
            f"'conda install -c conda-forge {backend}'"
        )
        raise NotInstalledError(msg)

    if grid_plot:
        return _grid_line_plot_backend_function
    else:
        return _line_plot_backend_function
