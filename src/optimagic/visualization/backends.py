import abc
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import plotly.graph_objects as go


@dataclass(frozen=True)
class PlotConfig:
    template: str
    plotly_legend: dict[str, Any]
    matplotlib_legend: dict[str, Any]


class BackendWrapper(abc.ABC):
    def __init__(self, plot_config):
        self.plot_config = plot_config

    @abc.abstractmethod
    def create_figure(self):
        pass

    @abc.abstractmethod
    def lineplot(self, **kwargs):
        pass

    @abc.abstractmethod
    def post_plot(self, **kwargs):
        pass

    @abc.abstractmethod
    def return_obj(self):
        pass


class BackendRegistry:
    _registry: dict[str, BackendWrapper] = {}

    @classmethod
    def register(cls, backend_name):
        def decorator(backend_wrapper):
            cls._registry[backend_name] = backend_wrapper
            return backend_wrapper

        return decorator

    @classmethod
    def get_backend_wrapper(cls, backend_name):
        if backend_name not in cls._registry:
            raise ValueError(
                f"Backend '{backend_name}' is not supported. "
                f"Supported backends are: {', '.join(cls._registry.keys())}."
            )
        return cls._registry.get(backend_name)


@BackendRegistry.register("plotly")
class PlotlyBackend(BackendWrapper):
    def __init__(self, plot_config):
        super().__init__(plot_config)
        self.fig = self.create_figure()

    def create_figure(self):
        fig = go.Figure()
        return fig

    def lineplot(self, *, x, y, color, name=None, plotly_scatter_kws=None, **kwargs):
        if plotly_scatter_kws is None:
            plotly_scatter_kws = {}

        trace = go.Scatter(
            x=x, y=y, mode="lines", line_color=color, name=name, **plotly_scatter_kws
        )
        self.fig.add_trace(trace)

    def post_plot(self, *, xlabel=None, ylabel=None, **kwargs):
        self.fig.update_layout(
            template=self.plot_config.template,
            xaxis_title_text=xlabel,
            yaxis_title_text=ylabel,
            legend=self.plot_config.plotly_legend,
        )

    def return_obj(self):
        return self.fig


@BackendRegistry.register("matplotlib")
class MatplotlibBackend(BackendWrapper):
    def __init__(self, plot_config):
        super().__init__(plot_config)
        self.fig, self.ax = self.create_figure()

    def create_figure(self):
        plt.style.use(self.plot_config.template)
        fig, ax = plt.subplots()
        return fig, ax

    def lineplot(self, *, x, y, color, name=None, **kwargs):
        self.ax.plot(x, y, color=color, label=name)

    def post_plot(self, *, xlabel=None, ylabel=None, **kwargs):
        self.ax.set(xlabel=xlabel, ylabel=ylabel)
        self.ax.legend(**self.plot_config.matplotlib_legend)

    def return_obj(self):
        return self.fig
