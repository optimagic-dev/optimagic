from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

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
) -> go.Figure:
    if template is None:
        template = "simple_white"

    fig = go.Figure()

    fig.update_layout(
        title=title,
        xaxis_title=xlabel.format(linebreak="<br>") if xlabel else None,
        yaxis_title=ylabel,
        template=template,
        height=height,
        width=width,
        legend=legend_properties,
        margin=margin_properties,
    )

    if horizontal_line is not None:
        fig.add_hline(
            y=horizontal_line,
            line_width=fig.layout.yaxis.linewidth or 1,
            opacity=1.0,
        )

    for line in lines:
        trace = go.Scatter(
            x=line.x,
            y=line.y,
            name=line.name,
            line_color=line.color,
            mode="lines",
            showlegend=line.show_in_legend,
        )
        fig.add_trace(trace)

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
        px = 1 / plt.rcParams["figure.dpi"]  # pixel in inches
        fig, ax = plt.subplots(
            figsize=(width * px, height * px) if width and height else None
        )

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
            ylabel=ylabel,
        )

        if legend_properties is None:
            legend_properties = {}
        ax.legend(**legend_properties)

        fig.tight_layout()

    return ax


BACKEND_AVAILABILITY_AND_LINE_PLOT_FUNCTION: dict[
    str, tuple[bool, LinePlotFunction]
] = {
    "plotly": (True, _line_plot_plotly),
    "matplotlib": (IS_MATPLOTLIB_INSTALLED, _line_plot_matplotlib),
}


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
    if backend not in BACKEND_AVAILABILITY_AND_LINE_PLOT_FUNCTION:
        msg = (
            f"Invalid plotting backend '{backend}'. "
            f"Available backends: "
            f"{', '.join(BACKEND_AVAILABILITY_AND_LINE_PLOT_FUNCTION.keys())}"
        )
        raise InvalidPlottingBackendError(msg)

    _is_backend_available, _line_plot_backend_function = (
        BACKEND_AVAILABILITY_AND_LINE_PLOT_FUNCTION[backend]
    )

    if not _is_backend_available:
        msg = (
            f"The {backend} backend is not installed. "
            f"Install the package using either 'pip install {backend}' or "
            f"'conda install -c conda-forge {backend}'"
        )
        raise NotInstalledError(msg)

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
