import abc
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from optimagic.visualization.plotting_utilities import LineData


class PlotBackend(abc.ABC):
    default_template: str
    default_palette: list

    @abc.abstractmethod
    def __init__(self, template: str | None):
        if template is None:
            template = self.default_template

        self.template = template
        self.figure: Any = None

    @abc.abstractmethod
    def add_lines(self, lines: list[LineData]) -> None:
        pass

    @abc.abstractmethod
    def set_labels(self, xlabel: str | None = None, ylabel: str | None = None) -> None:
        pass

    @abc.abstractmethod
    def set_legend_props(self, legend_props: dict[str, Any]) -> None:
        pass


class PlotlyBackend(PlotBackend):
    default_template: str = "simple_white"
    default_palette: list = px.colors.qualitative.Set2

    def __init__(self, template: str | None):
        super().__init__(template)
        self._fig = go.Figure()
        self._fig.update_layout(template=self.template)
        self.figure = self._fig

    def add_lines(self, lines: list[LineData]) -> None:
        for line in lines:
            trace = go.Scatter(
                x=line.x,
                y=line.y,
                name=line.name,
                mode="lines",
                line_color=line.color,
                showlegend=line.show_in_legend,
                connectgaps=True,
            )
            self._fig.add_trace(trace)

    def set_labels(self, xlabel: str | None = None, ylabel: str | None = None) -> None:
        self._fig.update_layout(xaxis_title_text=xlabel, yaxis_title_text=ylabel)

    def set_legend_props(self, legend_props: dict[str, Any]) -> None:
        self._fig.update_layout(legend=legend_props)


class MatplotlibBackend(PlotBackend):
    default_template: str = "default"
    default_palette: list = list(mpl.colormaps["Set2"].colors)

    def __init__(self, template: str | None):
        super().__init__(template)
        plt.style.use(self.template)
        self._fig, self._ax = plt.subplots()
        self.figure = self._fig

    def add_lines(self, lines: list[LineData]) -> None:
        for line in lines:
            self._ax.plot(
                line.x,
                line.y,
                color=line.color,
                label=line.name if line.show_in_legend else None,
            )

    def set_labels(self, xlabel: str | None = None, ylabel: str | None = None) -> None:
        self._ax.set(xlabel=xlabel, ylabel=ylabel)

    def set_legend_props(self, legend_props: dict[str, Any]) -> None:
        self._ax.legend(**legend_props)


PLOT_BACKEND_CLASSES = {
    "plotly": PlotlyBackend,
    "matplotlib": MatplotlibBackend,
}


def get_plot_backend_class(backend_name: str) -> type[PlotBackend]:
    if backend_name not in PLOT_BACKEND_CLASSES:
        msg = (
            f"Invalid backend name '{backend_name}'. "
            f"Supported backends are: {', '.join(PLOT_BACKEND_CLASSES.keys())}."
        )
        raise ValueError(msg)

    return PLOT_BACKEND_CLASSES[backend_name]
