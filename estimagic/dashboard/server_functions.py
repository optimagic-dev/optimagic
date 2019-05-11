"""Functions for setting up and running the BokehServer displaying the dashboard."""
import asyncio
from functools import partial

from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler
from bokeh.command.util import report_server_init_errors
from bokeh.server.server import Server

from estimagic.dashboard.dashboard import run_dashboard


def run_server(queue, port):
    """
    Setup and run a server creating und continuously updating a dashboard.

    The main building of the dashboard is done in run_dashboard.
    Here this function is only turned into an Application and run in a server.

    Args:
        queue (Queue):
            queue to which originally the parameters DataFrame is supplied and
            to which later the updated parameter Series will be supplied.

        port (int):
            port at which to display the dashboard.

    """
    asyncio.set_event_loop(asyncio.new_event_loop())

    apps = {"/": Application(FunctionHandler(partial(run_dashboard, queue=queue)))}

    server = _setup_server(apps, port)

    server._loop.start()
    server.start()


def _setup_server(apps, port):
    """
    Setup the server similarly to bokeh serve subcommand.

    In contrast to bokeh serve this supports being called from within a notebook.
    It also allows the server to be run in a separate thread while a main script
    is waiting for the output.

    Args:
        apps (dict):
            dictionary mapping suffixes of the address to Applications

        port (int):
            port where to host the BokehServer

    """
    # this is adapted from bokeh.subcommands.serve
    with report_server_init_errors(port=port):
        server = Server(apps, port=port)

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
