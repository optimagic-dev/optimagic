"""Functions to setup the dashboard and run a bokeh server with a supplied function."""
import random
import time
from datetime import datetime
from functools import partial
from threading import Thread

from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler
from bokeh.command.util import report_server_init_errors
from bokeh.layouts import row
from bokeh.models import ColumnDataSource
from bokeh.models import Panel
from bokeh.models.annotations import Legend
from bokeh.models.widgets import Tabs
from bokeh.palettes import Viridis256
from bokeh.plotting import figure
from bokeh.server.server import Server


def run_with_dashboard(func, notebook):
    """
    Run the *func* in a Bokeh server and return the optimization result.

    Args:
        func (function):
            Python function that takes doc and res as arguments.
            Note: doc must be the first argument of the function
        notebook (bool):
            whether the code is run in a notebook or not
            .. to-do::
                Automatically identify whether code is being run in a notebook or not.

    """
    res = []
    apps = {"/": Application(FunctionHandler(partial(func, res=res)))}
    server = _setup_server(apps=apps, notebook=notebook)

    if notebook is False:
        t_server = Thread(target=_start_server, args=(server, notebook))
        t_server.start()

        while len(res) == 0:
            time.sleep(0.1)
        return res

    else:
        _start_server(server, notebook)


def configure_dashboard(doc, param_df):
    conv_p, param_data = _setup_convergence_plot(param_df)

    tab1 = Panel(child=row([conv_p]), title="Convergence Plot")
    tabs = Tabs(tabs=[tab1])
    doc.add_root(tabs)

    return doc, [param_data]


def _setup_convergence_plot(param_df):
    # ToDo: split up convergence plot depending on MultiIndex and/or nr of parameters

    # this must only be modified from a Bokeh session callback
    param_data = ColumnDataSource(data=_data_dict_from_param_values(param_df["value"]))
    conv_p = figure(plot_height=350, tools="tap,reset,save")
    conv_p.title.text = "Convergence Plot"

    colors = random.sample(Viridis256, len(param_df))
    legend_items = []

    for i, name in enumerate(param_df.index):
        legend_items.append(
            (
                str(name),
                [
                    conv_p.line(
                        source=param_data,
                        x="time",
                        y=str(name),
                        line_width=2,
                        name=str(name),
                        color=colors[i],
                        nonselection_alpha=0,
                    )
                ],
            )
        )

    # Add Legend manually as our update somehow messes up the legend
    legend = Legend(items=legend_items)
    conv_p.add_layout(legend)
    return conv_p, param_data


def _data_dict_from_param_values(param_sr):
    entry = {"time": [datetime.now()]}
    entry.update({str(k): [v] for k, v in param_sr.to_dict().items()})
    return entry


def _setup_server(apps, notebook, port=5477):
    # this is adapted from bokeh.subcommands.serve
    with report_server_init_errors(port=port):
        server = Server(apps, port=port)
        if notebook is True:
            return server
        else:

            def show_callback():
                for route in apps.keys():
                    server.show(route)

            server.io_loop.add_callback(show_callback)

            address_string = "localhost"
            if server.address is not None and server.address != "":
                address_string = server.address

            for route in sorted(apps.keys()):
                url = "http://{}:{}{}{}".format(
                    address_string, server.port, server.prefix, route
                )
                print("Bokeh app running at: " + url)
        return server


def _start_server(server, notebook):
    if notebook is False:
        server._loop.start()
        server.start()
    else:
        server.start()
        server.show("/")
