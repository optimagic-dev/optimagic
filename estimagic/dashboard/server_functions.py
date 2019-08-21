"""Functions for setting up and running the BokehServer displaying the dashboard."""
import asyncio
import socket
from contextlib import closing
from functools import partial
from multiprocessing import Process

from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler
from bokeh.command.util import report_server_init_errors
from bokeh.server.server import Server

from estimagic.dashboard.dashboard import run_dashboard


def run_server(queue, stop_signal, db_options, start_param_df, start_fitness):
    """
    Setup and run a server creating und continuously updating a dashboard.

    The main building of the dashboard is done in run_dashboard.
    Here this function is only turned into an Application and run in a server.

    Args:
        queue (Queue):
            queue to which the updated parameter Series will be supplied.

        stop_signal (Event):
            signal from parent thread to stop the dashboard

        db_options (dict):
            dictionary with options. see ``run_dashboard`` for details.

        start_param_df (pd.DataFrame):
            DataFrame with the start params and information on them.

        start_fitness (float):
            fitness evaluation at the start parameters.
    """
    db_options, port, open_browser = _process_db_options(db_options)
    asyncio.set_event_loop(asyncio.new_event_loop())

    apps = {
        "/": Application(
            FunctionHandler(
                partial(
                    run_dashboard,
                    queue=queue,
                    db_options=db_options,
                    start_param_df=start_param_df,
                    start_fitness=start_fitness,
                    stop_signal=stop_signal,
                )
            )
        )
    }

    inner_server_process = Process(
        target=_setup_server,
        kwargs={"apps": apps, "port": port, "open_browser": open_browser},
        daemon=False,
    )
    inner_server_process.start()


def _process_db_options(db_options):
    db_options = db_options.copy()

    port = db_options.pop("port", _find_free_port())
    open_browser = db_options.pop("open_browser", True)

    if db_options.get("rollover", 1) <= 0:
        db_options["rollover"] = None
    else:
        db_options["rollover"] = db_options.get("rollover", 500)

    full_db_options = {"evaluations_to_skip": 0, "time_btw_updates": 0.001}
    full_db_options.update(db_options)

    return full_db_options, port, open_browser


def _find_free_port():
    """
    Find a free port on the localhost.

    Adapted from https://stackoverflow.com/a/45690594
    """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("localhost", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def _setup_server(apps, port, open_browser):
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

        # On a remote server, we do not want to start the dashboard here.
        if open_browser:

            def show_callback():
                for route in apps.keys():
                    server.show(route)

            server.io_loop.add_callback(show_callback)

        address_string = server.address if server.address else "localhost"

        for route in sorted(apps):
            url = "http://{}:{}{}{}".format(
                address_string, server.port, server.prefix, route
            )
            print("Bokeh app running at:", url)

        # For Windows, it is important that the server is started here as otherwise a
        # pickling error happens within multiprocess. See
        # https://stackoverflow.com/a/38236630/7523785 for more information.
        server._loop.start()
        server.start()
