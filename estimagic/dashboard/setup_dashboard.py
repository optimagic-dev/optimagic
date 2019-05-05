from threading import Thread

from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler
from bokeh.command.util import report_server_init_errors
from bokeh.layouts import row
from bokeh.models import ColumnDataSource
from bokeh.models import Panel
from bokeh.models.widgets import Tabs
from bokeh.plotting import figure
from bokeh.server.server import Server


def run_with_dashboard(func, notebook=False):
    apps = {"/": Application(FunctionHandler(func))}
    server = _setup_server(apps)
    t_server = Thread(target=_start_server, args=(server, notebook))
    t_server.start()


def configure_dashboard(doc):
    # this must only be modified from a Bokeh session callback
    data = ColumnDataSource(data={"x": [0], "y": [0]})
    data2 = ColumnDataSource(data={"x": [0], "y": [0]})

    p = figure(x_range=[0, 1], y_range=[0, 1])
    p.circle(x="x", y="y", source=data)
    layout = row([p])
    tab1 = Panel(child=layout, title="Plot1")

    p2 = figure(x_range=[0, 1], y_range=[0, 1])
    p2.circle(x="x", y="y", source=data2, color="red")
    layout = row([p2])
    # Make a tab with the layout
    tab2 = Panel(child=layout, title="Plot2")

    tabs = Tabs(tabs=[tab1, tab2])
    doc.add_root(tabs)

    return doc, [data, data2]


def _setup_server(apps, notebook=False, port=5473):
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
