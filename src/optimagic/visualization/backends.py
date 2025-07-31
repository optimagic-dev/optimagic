import abc
from typing import Callable

import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from optimagic.visualization.plotting_utilities import LineData, PlotConfig


class BackendWrapper(abc.ABC):
    default_template: str
    default_palette: list

    def __init__(self, plot_config: PlotConfig):
        self.plot_config = plot_config

    @abc.abstractmethod
    def line_plot(self, lines: list[LineData]) -> None:
        pass

    @abc.abstractmethod
    def label(self, **kwargs):
        pass

    @abc.abstractmethod
    def return_obj(self):
        pass


class BackendRegistry:
    _registry: dict[str, type[BackendWrapper]] = {}

    @classmethod
    def register(cls, backend_name: str) -> Callable:
        def decorator(backend_wrapper):
            cls._registry[backend_name] = backend_wrapper
            return backend_wrapper

        return decorator

    @classmethod
    def get_backend_wrapper(cls, backend_name: str) -> type[BackendWrapper]:
        if backend_name not in cls._registry:
            raise ValueError(
                f"Backend '{backend_name}' is not supported. "
                f"Supported backends are: {', '.join(cls._registry.keys())}."
            )
        return cls._registry[backend_name]


@BackendRegistry.register("plotly")
class PlotlyBackend(BackendWrapper):
    default_template: str = "simple_white"
    default_palette: list = px.colors.qualitative.Set2

    def __init__(self, plot_config):
        super().__init__(plot_config)
        self.fig = go.Figure()

    def line_plot(self, lines: list[LineData]) -> None:
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
            self.fig.add_trace(trace)

    def label(self, *, xlabel=None, ylabel=None, **kwargs):
        self.fig.update_layout(
            template=self.plot_config.template,
            xaxis_title_text=xlabel,
            yaxis_title_text=ylabel,
            legend=self.plot_config.legend,
        )

    def return_obj(self):
        return self.fig


@BackendRegistry.register("matplotlib")
class MatplotlibBackend(BackendWrapper):
    default_template: str = "default"
    default_palette: list = list(mpl.colormaps["Set2"].colors)

    def __init__(self, plot_config: PlotConfig):
        super().__init__(plot_config)
        plt.style.use(self.plot_config.template)
        self.fig, self.ax = plt.subplots()

    def line_plot(self, lines: list[LineData]) -> None:
        for line in lines:
            self.ax.plot(
                line.x,
                line.y,
                color=line.color,
                label=line.name if line.show_in_legend else None,
            )

    def label(self, *, xlabel=None, ylabel=None, **kwargs):
        self.ax.set(xlabel=xlabel, ylabel=ylabel)
        self.ax.legend(**self.plot_config.legend)

    def return_obj(self):
        return self.fig
