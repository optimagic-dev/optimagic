import itertools
from typing import TYPE_CHECKING, Any, Literal, Protocol, overload, runtime_checkable

import numpy as np
import plotly.graph_objects as go

from optimagic.config import (
    IS_ALTAIR_INSTALLED,
    IS_BOKEH_INSTALLED,
    IS_MATPLOTLIB_INSTALLED,
)
from optimagic.exceptions import InvalidPlottingBackendError, NotInstalledError
from optimagic.visualization.plotting_utilities import LineData, MarkerData

if TYPE_CHECKING:
    import altair as alt
    import bokeh
    import matplotlib.pyplot as plt


@runtime_checkable
class LinePlotFunction(Protocol):
    def __call__(
        self,
        lines: list[LineData],
        *,
        title: str | None,
        xlabel: str | None,
        xrange: tuple[float, float] | None,
        ylabel: str | None,
        yrange: tuple[float, float] | None,
        template: str | None,
        height: int | None,
        width: int | None,
        legend_properties: dict[str, Any] | None,
        margin_properties: dict[str, Any] | None,
        horizontal_line: float | None,
        marker: MarkerData | None,
        subplot: Any | None = None,
    ) -> Any:
        """Protocol of the line_plot function used for type checking.

        Args:
            ...: All other argument descriptions can be found in the docstring of the
                `line_plot` function.
            subplot: The subplot to which the lines should be plotted. The type of this
                argument depends on the backend used. If not provided, a new figure is
                created.

        """
        ...


@runtime_checkable
class GridLinePlotFunction(Protocol):
    def __call__(
        self,
        lines_list: list[list[LineData]],
        *,
        n_rows: int,
        n_cols: int,
        titles: list[str] | None,
        xlabels: list[str] | None,
        xrange: tuple[float, float] | None,
        share_x: bool,
        ylabels: list[str] | None,
        yrange: tuple[float, float] | None,
        share_y: bool,
        template: str | None,
        height: int | None,
        width: int | None,
        legend_properties: dict[str, Any] | None,
        margin_properties: dict[str, Any] | None,
        plot_title: str | None,
        marker_list: list[MarkerData] | None,
        make_subplot_kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Protocol of the grid_line_plot function used for type checking.

        Args:
            ...: All other argument descriptions can be found in the docstring of the
                `grid_line_plot` function.

        """
        ...


def _line_plot_plotly(
    lines: list[LineData],
    *,
    title: str | None,
    xlabel: str | None,
    xrange: tuple[float, float] | None,
    ylabel: str | None,
    yrange: tuple[float, float] | None,
    template: str | None,
    height: int | None,
    width: int | None,
    legend_properties: dict[str, Any] | None,
    margin_properties: dict[str, Any] | None,
    horizontal_line: float | None,
    marker: MarkerData | None,
    subplot: tuple[go.Figure, int, int] | None = None,
) -> go.Figure:
    """Create a line plot using Plotly.

    Args:
        ...: All other argument descriptions can be found in the docstring of the
            `line_plot` function.
        subplot: A tuple specifying the subplot to which the lines should be plotted.
            The tuple contains the Plotly `Figure` object, the row index, and the column
            index of the subplot. If not provided, a new `Figure` object is created.

    Returns:
        A Plotly Figure object.

    """
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
        title=xlabel.format(linebreak="<br>") if xlabel else None,
        range=xrange,
        row=row,
        col=col,
    )
    fig.update_yaxes(
        title=ylabel.format(linebreak="<br>") if ylabel else None,
        range=yrange,
        row=row,
        col=col,
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

    if marker is not None:
        trace = go.Scatter(
            x=[marker.x],
            y=[marker.y],
            name=marker.name,
            marker_color=marker.color,
            showlegend=False,
        )
        fig.add_trace(trace, row=row, col=col)

    return fig


def _grid_line_plot_plotly(
    lines_list: list[list[LineData]],
    *,
    n_rows: int,
    n_cols: int,
    titles: list[str] | None,
    xlabels: list[str] | None,
    xrange: tuple[float, float] | None,
    share_x: bool,
    ylabels: list[str] | None,
    yrange: tuple[float, float] | None,
    share_y: bool,
    template: str | None,
    height: int | None,
    width: int | None,
    legend_properties: dict[str, Any] | None,
    margin_properties: dict[str, Any] | None,
    plot_title: str | None,
    marker_list: list[MarkerData] | None,
    make_subplot_kwargs: dict[str, Any] | None = None,
) -> go.Figure:
    """Create a grid of line plots using Plotly.

    Args:
        ...: All other argument descriptions can be found in the docstring of the
            `grid_line_plot` function.

    Returns:
        A Plotly Figure object.

    """
    from plotly.subplots import make_subplots

    subplot_kwargs = dict(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=titles,
        shared_yaxes=share_y,
        shared_xaxes=share_x,
        horizontal_spacing=0.3 / n_cols,
    )
    subplot_kwargs.update(make_subplot_kwargs or {})
    fig = make_subplots(**subplot_kwargs)

    for i, (row, col) in enumerate(
        itertools.product(range(1, n_rows + 1), range(1, n_cols + 1))
    ):
        if i >= len(lines_list):
            break

        _line_plot_plotly(
            lines_list[i],
            title=None,
            xlabel=xlabels[i] if xlabels else None,
            xrange=xrange,
            ylabel=ylabels[i] if ylabels else None,
            yrange=yrange,
            template=template,
            height=height,
            width=width,
            legend_properties=legend_properties,
            margin_properties=margin_properties,
            horizontal_line=None,
            marker=marker_list[i] if marker_list else None,
            subplot=(fig, row, col),
        )

    if plot_title is not None:
        fig.update_layout(title=plot_title)

    return fig


def _line_plot_matplotlib(
    lines: list[LineData],
    *,
    title: str | None,
    xlabel: str | None,
    xrange: tuple[float, float] | None,
    ylabel: str | None,
    yrange: tuple[float, float] | None,
    template: str | None,
    height: int | None,
    width: int | None,
    legend_properties: dict[str, Any] | None,
    margin_properties: dict[str, Any] | None,
    horizontal_line: float | None,
    marker: MarkerData | None,
    subplot: "plt.Axes | None" = None,
) -> "plt.Axes":
    """Create a line plot using Matplotlib.

    Args:
        ...: All other argument descriptions can be found in the docstring of the
            `line_plot` function.
        subplot: A Matplotlib `Axes` object to which the lines should be plotted.
            If provided, the plot is drawn on the given `Axes`. If not provided,
            a new `Figure` and `Axes` are created.

    Returns:
        A Matplotlib Axes object.

    """
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

        for line in lines:
            ax.plot(
                line.x,
                line.y,
                label=line.name if line.show_in_legend else None,
                color=line.color,
            )

        if horizontal_line is not None:
            ax.axhline(
                y=horizontal_line,
                color=ax.spines["left"].get_edgecolor() or "gray",
                linewidth=ax.spines["left"].get_linewidth() or 1.0,
            )

        if marker is not None:
            ax.scatter(
                [marker.x],
                [marker.y],
                color=marker.color,
                label=None,
            )

        ax.set(
            title=title,
            xlabel=xlabel.format(linebreak="\n") if xlabel else None,
            xlim=xrange,
            ylabel=ylabel.format(linebreak="\n") if ylabel else None,
            ylim=yrange,
        )

        if subplot is None and legend_properties is not None:
            fig.legend(**legend_properties)

    return ax


def _grid_line_plot_matplotlib(
    lines_list: list[list[LineData]],
    *,
    n_rows: int,
    n_cols: int,
    titles: list[str] | None,
    xlabels: list[str] | None,
    xrange: tuple[float, float] | None,
    share_x: bool,
    ylabels: list[str] | None,
    yrange: tuple[float, float] | None,
    share_y: bool,
    template: str | None,
    height: int | None,
    width: int | None,
    legend_properties: dict[str, Any] | None,
    margin_properties: dict[str, Any] | None,
    plot_title: str | None,
    marker_list: list[MarkerData] | None,
    make_subplot_kwargs: dict[str, Any] | None = None,
) -> np.ndarray:
    """Create a grid of line plots using Matplotlib.

    Args:
        ...: All other argument descriptions can be found in the docstring of the
            `grid_line_plot` function.

    Returns:
        A 2D numpy array of Matplotlib Axes objects.

    """
    import matplotlib.pyplot as plt

    px = 1 / plt.rcParams["figure.dpi"]  # pixel in inches
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        squeeze=False,  # always return a 2D array of axes
        figsize=(width * px, height * px) if width and height else None,
        layout="constrained",
    )

    for i, (row, col) in enumerate(itertools.product(range(n_rows), range(n_cols))):
        if i >= len(lines_list):
            axes[row, col].set_visible(False)
            continue

        if share_x and row < n_rows - 1:
            # Share x-axis with bottom subplot in the same column
            axes[row, col].sharex(axes[-1, col])
            axes[row, col].xaxis.set_tick_params(labelbottom=False)
        if share_y and col > 0:
            # Share y-axis with left subplot in the same row
            axes[row, col].sharey(axes[row, 0])
            axes[row, col].yaxis.set_tick_params(labelleft=False)

        _line_plot_matplotlib(
            lines_list[i],
            title=titles[i] if titles else None,
            xlabel=xlabels[i] if xlabels else None,
            xrange=xrange,
            ylabel=ylabels[i] if ylabels else None,
            yrange=yrange,
            template=template,
            height=None,
            width=None,
            legend_properties=None,
            margin_properties=None,
            horizontal_line=None,
            marker=marker_list[i] if marker_list else None,
            subplot=axes[row, col],
        )

    if legend_properties is not None:
        fig.legend(**legend_properties)
    if plot_title is not None:
        fig.suptitle(plot_title)

    return axes


def _line_plot_bokeh(
    lines: list[LineData],
    *,
    title: str | None,
    xlabel: str | None,
    xrange: tuple[float, float] | None,
    ylabel: str | None,
    yrange: tuple[float, float] | None,
    template: str | None,
    height: int | None,
    width: int | None,
    legend_properties: dict[str, Any] | None,
    margin_properties: dict[str, Any] | None,
    horizontal_line: float | None,
    marker: MarkerData | None,
    subplot: "bokeh.plotting.figure | None" = None,
) -> "bokeh.plotting.figure":
    """Create a line plot using Bokeh.

    Args:
        ...: All other argument descriptions can be found in the docstring of the
            `line_plot` function.
        subplot: A Bokeh `Figure` object to which the lines should be plotted.
            If provided, the plot is drawn on the given `Figure`. If not provided,
            a new `Figure` is created.

    Returns:
        A Bokeh Figure object.

    """
    from bokeh import themes
    from bokeh.io import curdoc
    from bokeh.models import Range1d
    from bokeh.models.annotations import Legend, LegendItem, Span, Title
    from bokeh.plotting import figure

    if template is None:
        template = "light_minimal"
    curdoc().theme = themes.built_in_themes[template]

    if subplot is not None:
        p = subplot
    else:
        p = figure()

    if title is not None:
        p.title = Title(text=title)
    if xlabel is not None:
        p.xaxis.axis_label = xlabel.format(linebreak="\n")
    if xrange is not None:
        p.x_range = Range1d(*xrange)
    if ylabel is not None:
        p.yaxis.axis_label = ylabel.format(linebreak="\n")
    if yrange is not None:
        p.y_range = Range1d(*yrange)
    if height is not None:
        p.height = height
    if width is not None:
        p.width = width

    _legend_items = []
    for line in lines:
        glyph = p.line(
            line.x,
            line.y,
            line_color=line.color,
            line_width=2,
        )

        if line.show_in_legend:
            _legend_items.append(LegendItem(label=line.name, renderers=[glyph]))  # type: ignore[list-item]

    if horizontal_line is not None:
        span = Span(
            location=horizontal_line,
            dimension="width",
            line_color=p.yaxis.axis_line_color or "gray",
            line_width=p.yaxis.axis_line_width or 2,
        )
        p.add_layout(span)

    if marker is not None:
        p.scatter(
            x=[marker.x],
            y=[marker.y],
            marker="circle",
            fill_color=marker.color,
            line_color=marker.color,
            size=10,
        )

    if _legend_items:
        legend_kwargs = legend_properties.copy() if legend_properties else {}
        place = legend_kwargs.pop("place", "center")
        text = legend_kwargs.pop("title", None)

        legend = Legend(items=_legend_items, **(legend_kwargs))
        p.add_layout(legend, place=place)
        p.legend.title = text

    return p


def _grid_line_plot_bokeh(
    lines_list: list[list[LineData]],
    *,
    n_rows: int,
    n_cols: int,
    titles: list[str] | None,
    xlabels: list[str] | None,
    xrange: tuple[float, float] | None,
    share_x: bool,
    ylabels: list[str] | None,
    yrange: tuple[float, float] | None,
    share_y: bool,
    template: str | None,
    height: int | None,
    width: int | None,
    legend_properties: dict[str, Any] | None,
    margin_properties: dict[str, Any] | None,
    plot_title: str | None,
    marker_list: list[MarkerData] | None,
    make_subplot_kwargs: dict[str, Any] | None = None,
) -> "bokeh.models.GridPlot":
    """Create a grid of line plots using Bokeh.

    Args:
        ...: All other argument descriptions can be found in the docstring of the
            `grid_line_plot` function.

    Returns:
        A Bokeh gridplot object.

    """

    from bokeh.layouts import gridplot
    from bokeh.plotting import figure

    plots: list[list[figure]] = []

    for row in range(n_rows):
        subplot_row: list[Any] = []
        for col in range(n_cols):
            idx = row * n_cols + col
            if idx >= len(lines_list):
                break

            p = figure()

            _line_plot_bokeh(
                lines_list[idx],
                title=titles[idx] if titles else None,
                xlabel=xlabels[idx] if xlabels else None,
                xrange=xrange,
                ylabel=ylabels[idx] if ylabels else None,
                yrange=yrange,
                template=template,
                height=None,
                width=None,
                legend_properties=legend_properties,
                margin_properties=None,
                horizontal_line=None,
                marker=marker_list[idx] if marker_list else None,
                subplot=p,
            )

            if share_x:
                if row > 0:
                    # Share x-range with the top-most subplot in the same column
                    p.x_range = plots[0][col].x_range
                if row < n_rows - 1:
                    # Hide tick labels except for subplots in the last row
                    p.xaxis.major_label_text_font_size = "0pt"
            if share_y:
                if col > 0:
                    # Share y-range with the left-most subplot in the same row
                    p.y_range = subplot_row[0].y_range

                    # Hide tick labels except for subplots in the first column
                    p.yaxis.major_label_text_font_size = "0pt"

            subplot_row.append(p)
        plots.append(subplot_row)

    grid = gridplot(  # type: ignore[call-overload]
        plots,
        height=height // n_rows if height else None,
        width=width // n_cols if width else None,
        toolbar_location="right",
    )

    return grid


def _line_plot_altair(
    lines: list[LineData],
    *,
    title: str | None,
    xlabel: str | None,
    xrange: tuple[float, float] | None,
    ylabel: str | None,
    yrange: tuple[float, float] | None,
    template: str | None,
    height: int | None,
    width: int | None,
    legend_properties: dict[str, Any] | None,
    margin_properties: dict[str, Any] | None,
    horizontal_line: float | None,
    marker: MarkerData | None,
    subplot: None = None,
) -> "alt.Chart":
    """Create a line plot using Altair.

    Args:
        ...: All other argument descriptions can be found in the docstring of the
            `line_plot` function.
        subplot: Unused by Altair.

    Returns:
        An Altair Chart object.

    """
    import altair as alt
    import pandas as pd

    alt.data_transformers.disable_max_rows()

    if template is None:
        template = "default"
    alt.theme.enable(template)

    dfs = []
    for line in lines:
        df = pd.DataFrame(
            {"x": line.x, "y": line.y, "name": line.name, "color": line.color}
        )
        dfs.append(df)
    source = pd.concat(dfs)

    figure_properties: dict[str, str | int] = {}
    if title is not None:
        figure_properties["title"] = title
    if width is not None:
        figure_properties["width"] = width
    if height is not None:
        figure_properties["height"] = height

    chart = (
        alt.Chart(source)
        .mark_line()
        .encode(
            x=alt.X(
                "x",
                title=xlabel.split("{linebreak}") if xlabel else None,
                scale=alt.Scale(domain=list(xrange)) if xrange else alt.Undefined,
            ),
            y=alt.Y(
                "y",
                title=ylabel.split("{linebreak}") if ylabel else None,
                scale=alt.Scale(domain=list(yrange)) if yrange else alt.Undefined,
            ),
            color=alt.Color("color:N", scale=None),
            detail="name:N",
        )
        .properties(**figure_properties)
    )

    if any(line.show_in_legend for line in lines):
        legend = (
            alt.Chart(source)
            .mark_line()
            .encode(
                color=alt.Color(
                    "name:N",
                    title=None,
                    legend=alt.Legend(**(legend_properties or {})),
                    scale=alt.Scale(
                        domain=[line.name for line in lines if line.show_in_legend],
                        range=[
                            line.color or "" for line in lines if line.show_in_legend
                        ],
                    ),
                )
            )
        )
        chart = chart + legend

    if horizontal_line is not None:
        hline = (
            alt.Chart(pd.DataFrame({"y": [horizontal_line]})).mark_rule().encode(y="y")
        )
        chart = chart + hline

    if marker is not None:
        marker_chart = (
            alt.Chart(pd.DataFrame({"x": [marker.x], "y": [marker.y]}))
            .mark_point(size=100, shape="circle", color=marker.color, filled=True)
            .encode(x="x", y="y")
        )
        chart = chart + marker_chart

    return chart.interactive()


def _grid_line_plot_altair(
    lines_list: list[list[LineData]],
    *,
    n_rows: int,
    n_cols: int,
    titles: list[str] | None,
    xlabels: list[str] | None,
    xrange: tuple[float, float] | None,
    share_x: bool,
    ylabels: list[str] | None,
    yrange: tuple[float, float] | None,
    share_y: bool,
    template: str | None,
    height: int | None,
    width: int | None,
    legend_properties: dict[str, Any] | None,
    margin_properties: dict[str, Any] | None,
    plot_title: str | None,
    marker_list: list[MarkerData] | None,
    make_subplot_kwargs: dict[str, Any] | None = None,
) -> "alt.Chart | alt.HConcatChart | alt.VConcatChart":
    """Create a grid of line plots using Altair.

    Args:
        ...: All other argument descriptions can be found in the docstring of the
            `grid_line_plot` function.

    Returns:
        An Altair Chart if the grid contains only one subplot, an Altair HConcatChart
            if 'n_rows' is 1, otherwise an Altair VConcatChart.

    """
    import altair as alt

    subplot_height = height // n_rows if height else None
    subplot_width = width // n_cols if width else None

    charts = []
    for row_idx in range(n_rows):
        chart_row = []
        for col_idx in range(n_cols):
            i = row_idx * n_cols + col_idx
            if i >= len(lines_list):
                break

            chart = _line_plot_altair(
                lines_list[i],
                title=titles[i] if titles else None,
                xlabel=xlabels[i] if xlabels else None,
                xrange=xrange,
                ylabel=ylabels[i] if ylabels else None,
                yrange=yrange,
                template=template,
                height=subplot_height,
                width=subplot_width,
                legend_properties=legend_properties,
                margin_properties=None,
                horizontal_line=None,
                marker=marker_list[i] if marker_list else None,
                subplot=None,
            )

            chart_row.append(chart)
        charts.append(chart_row)

    row_selections = [
        alt.selection_interval(
            bind="scales", encodings=["y"], name=f"share_y_row{row_idx}"
        )
        for row_idx in range(n_rows)
    ]
    col_selections = [
        alt.selection_interval(
            bind="scales", encodings=["x"], name=f"share_x_col{col_idx}"
        )
        for col_idx in range(n_cols)
    ]

    for row_idx, row in enumerate(charts):
        for col_idx in range(len(row)):
            chart = row[col_idx]

            params = []
            if share_y:
                # Share y-axis for all subplots in the same row
                params.append(row_selections[row_idx])
            else:
                # Use independent y-axes for each subplot
                params.append(
                    alt.selection_interval(
                        bind="scales",
                        encodings=["y"],
                        name=f"ind_y_row{row_idx}_col{col_idx}",
                    )
                )
            if share_x:
                # Share x-axis for all subplots in the same column
                params.append(col_selections[col_idx])
            else:
                # Use independent x-axes for each subplot
                params.append(
                    alt.selection_interval(
                        bind="scales",
                        encodings=["x"],
                        name=f"ind_x_row{row_idx}_col{col_idx}",
                    )
                )
            chart = chart.add_params(*params)

            if share_y and col_idx > 0:
                # Hide y-axis ticklabels for all subplots except the leftmost column
                chart = chart.encode(y=alt.Y(axis=alt.Axis(labels=False)))
            if share_x and row_idx < n_rows - 1:
                # Hide x-axis ticklabels for all subplots except the bottom row
                chart = chart.encode(x=alt.X(axis=alt.Axis(labels=False)))

            charts[row_idx][col_idx] = chart

    row_charts = []
    for row in charts:
        row_chart: alt.Chart | alt.HConcatChart
        if len(row) == 1:
            row_chart = row[0]
        else:
            row_chart = alt.hconcat(*row)
        row_charts.append(row_chart)

    grid_chart: alt.Chart | alt.HConcatChart | alt.VConcatChart
    if len(row_charts) == 1:
        grid_chart = row_charts[0]
    else:
        grid_chart = alt.vconcat(*row_charts)

    if plot_title is not None:
        grid_chart = grid_chart.properties(title=plot_title)

    return grid_chart


def line_plot(
    lines: list[LineData],
    backend: Literal["plotly", "matplotlib", "bokeh", "altair"] = "plotly",
    *,
    title: str | None = None,
    xlabel: str | None = None,
    xrange: tuple[float, float] | None = None,
    ylabel: str | None = None,
    yrange: tuple[float, float] | None = None,
    template: str | None = None,
    height: int | None = None,
    width: int | None = None,
    legend_properties: dict[str, Any] | None = None,
    margin_properties: dict[str, Any] | None = None,
    horizontal_line: float | None = None,
    marker: MarkerData | None = None,
) -> Any:
    """Create a line plot corresponding to the specified backend.

    Args:
        lines: List of objects each containing data for a line in the plot.
            The order of lines in the list determines the order in which they are
            plotted, with later lines being rendered on top of earlier ones.
        backend: The backend to use for plotting.
        title: Title of the plot.
        xlabel: Label for the x-axis.
        xrange: View limits for the x-axis.
        ylabel: Label for the y-axis.
        yrange: View limits for the y-axis.
        template: Backend-specific template for styling the plot.
        height: Height of the plot (in pixels).
        width: Width of the plot (in pixels).
        legend_properties: Backend-specific properties for the legend.
        margin_properties: Backend-specific properties for the plot margins.
        horizontal_line: If provided, a horizontal line is drawn at the specified
            y-value.
        marker: An object containing data for a marker in the plot.

    Returns:
        A figure object corresponding to the specified backend.

    """
    _line_plot_backend_function = _get_plot_function(backend, grid_plot=False)

    fig = _line_plot_backend_function(
        lines,
        title=title,
        xlabel=xlabel,
        xrange=xrange,
        ylabel=ylabel,
        yrange=yrange,
        template=template,
        height=height,
        width=width,
        legend_properties=legend_properties,
        margin_properties=margin_properties,
        horizontal_line=horizontal_line,
        marker=marker,
    )

    return fig


def grid_line_plot(
    lines_list: list[list[LineData]],
    backend: Literal["plotly", "matplotlib", "bokeh", "altair"] = "plotly",
    *,
    n_rows: int,
    n_cols: int,
    titles: list[str] | None = None,
    xlabels: list[str] | None = None,
    xrange: tuple[float, float] | None = None,
    share_x: bool = False,
    ylabels: list[str] | None = None,
    yrange: tuple[float, float] | None = None,
    share_y: bool = False,
    template: str | None = None,
    height: int | None = None,
    width: int | None = None,
    legend_properties: dict[str, Any] | None = None,
    margin_properties: dict[str, Any] | None = None,
    plot_title: str | None = None,
    marker_list: list[MarkerData] | None = None,
    make_subplot_kwargs: dict[str, Any] | None = None,
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
        xlabels: Labels for the x-axis of each subplot.
        xrange: View limits for the x-axis of each subplot.
        share_x: If True, all subplots share the same x-axis limits and each subplot in
            a column actually share the x-axis.
        ylabels: Labels for the y-axis of each subplot.
        yrange: View limits for the y-axis of each subplot.
        share_y: If True, all subplots share the same y-axis limits and each subplot in
            a row actually share the y-axis.
        template: Backend-specific template for styling the plots.
        height: Height of the entire grid plot (in pixels).
        width: Width of the entire grid plot (in pixels).
        legend_properties: Backend-specific properties for the legend.
        margin_properties: Backend-specific properties for the plot margins.
        plot_title: Title for the entire grid plot.
        marker_list: A list where where each element is an object containing data
            for a marker in a subplot. The order of objects in the list determines
            the subplot on which the marker is plotted.

    Returns:
        A figure object corresponding to the specified backend.

    """
    _grid_line_plot_backend_function = _get_plot_function(backend, grid_plot=True)

    fig = _grid_line_plot_backend_function(
        lines_list,
        n_rows=n_rows,
        n_cols=n_cols,
        titles=titles,
        xlabels=xlabels,
        xrange=xrange,
        share_x=share_x,
        ylabels=ylabels,
        yrange=yrange,
        share_y=share_y,
        template=template,
        height=height,
        width=width,
        legend_properties=legend_properties,
        margin_properties=margin_properties,
        plot_title=plot_title,
        marker_list=marker_list,
        make_subplot_kwargs=make_subplot_kwargs,
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
    "bokeh": (
        IS_BOKEH_INSTALLED,
        _line_plot_bokeh,
        _grid_line_plot_bokeh,
    ),
    "altair": (
        IS_ALTAIR_INSTALLED,
        _line_plot_altair,
        _grid_line_plot_altair,
    ),
}


@overload
def _get_plot_function(
    backend: Literal["plotly", "matplotlib", "bokeh", "altair"],
    grid_plot: Literal[False],
) -> LinePlotFunction: ...


@overload
def _get_plot_function(
    backend: Literal["plotly", "matplotlib", "bokeh", "altair"],
    grid_plot: Literal[True],
) -> GridLinePlotFunction: ...


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
