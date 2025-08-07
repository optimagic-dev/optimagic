import abc
from typing import Any

import plotly.express as px
import plotly.graph_objects as go

from optimagic.config import IS_MATPLOTLIB_INSTALLED
from optimagic.exceptions import InvalidPlottingBackendError, NotInstalledError
from optimagic.visualization.plotting_utilities import LineData

if IS_MATPLOTLIB_INSTALLED:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    # Handle the case where matplotlib is used in notebooks (inline backend)
    # to ensure that interactive mode is disabled to avoid double plotting.
    # (See: https://github.com/matplotlib/matplotlib/issues/26221)
    if mpl.get_backend() == "module://matplotlib_inline.backend_inline":
        plt.install_repl_displayhook()
        plt.ioff()


class PlotBackend(abc.ABC):
    is_available: bool
    default_template: str

    @classmethod
    @abc.abstractmethod
    def get_default_palette(cls) -> list:
        pass

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
    def set_legend_properties(self, legend_properties: dict[str, Any]) -> None:
        pass


class PlotlyBackend(PlotBackend):
    is_available: bool = True
    default_template: str = "simple_white"

    @classmethod
    def get_default_palette(cls) -> list:
        return px.colors.qualitative.Set2

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

    def set_legend_properties(self, legend_properties: dict[str, Any]) -> None:
        self._fig.update_layout(legend=legend_properties)


class MatplotlibBackend(PlotBackend):
    is_available: bool = IS_MATPLOTLIB_INSTALLED
    default_template: str = "default"

    @classmethod
    def get_default_palette(cls) -> list:
        return [mpl.colormaps["Set2"](i) for i in range(mpl.colormaps["Set2"].N)]

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

    def set_legend_properties(self, legend_properties: dict[str, Any]) -> None:
        self._ax.legend(**legend_properties)


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
        raise InvalidPlottingBackendError(msg)

    return _get_backend_if_installed(backend_name)


def _get_backend_if_installed(backend_name: str) -> type[PlotBackend]:
    plot_cls = PLOT_BACKEND_CLASSES[backend_name]

    if not plot_cls.is_available:
        msg = (
            f"The '{backend_name}' backend is not installed. "
            f"Install the package using either 'pip install {backend_name}' or "
            f"'conda install -c conda-forge {backend_name}'"
        )
        raise NotInstalledError(msg)

    return plot_cls
