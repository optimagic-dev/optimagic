import asyncio
import pathlib
import socket
from contextlib import closing
from functools import partial

from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler
from bokeh.command.util import report_server_init_errors
from bokeh.server.server import Server
from estimagic.dashboard.dashboard_app import dashboard_app


def run_dashboard(
    database_path,
    no_browser,
    port,
    updating_options,
):
    """Start the dashboard pertaining to one database.

    Args:
        database_path (str or pathlib.Path): Path to an sqlite3 file which
            typically has the file extension ``.db``.
        no_browser (bool): If True the dashboard does not open in the browser.
        port (int): Port where to display the dashboard.
        updating_options (dict): Specification how to update the plotting data.
            It contains "rollover", "update_frequency", "update_chunk", "jump" and
            "stride".

    """
    port = _find_free_port() if port is None else port
    port = int(port)

    if not isinstance(database_path, (str, pathlib.Path)):
        raise TypeError(
            "database_path must be string or pathlib.Path. ",
            f"You supplied {type(database_path)}.",
        )
    else:
        database_path = pathlib.Path(database_path)
        if not database_path.exists():
            raise ValueError(
                f"The database path {database_path} you supplied does not exist."
            )

    session_data = {
        "last_retrieved": 0,
        "database_path": database_path,
        "callbacks": {},
    }

    app_func = partial(
        dashboard_app,
        session_data=session_data,
        updating_options=updating_options,
    )
    apps = {"/": Application(FunctionHandler(app_func))}

    _start_server(apps=apps, port=port, no_browser=no_browser)


def _find_free_port():
    """Find a free port on the localhost.

    Adapted from https://stackoverflow.com/a/45690594

    """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("localhost", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def _start_server(apps, port, no_browser):
    """Create and start a bokeh server with the supplied apps.

    Args:
        apps (dict): mapping from relative paths to bokeh Applications.
        port (int): port where to show the dashboard.
        no_browser (bool): whether to show the dashboard in the browser

    """
    # necessary for the dashboard to work when called from a notebook
    asyncio.set_event_loop(asyncio.new_event_loop())

    # this is adapted from bokeh.subcommands.serve
    with report_server_init_errors(port=port):
        server = Server(apps, port=port)

        # On a remote server, we do not want to start the dashboard here.
        if not no_browser:

            def show_callback():
                server.show("/")

            server.io_loop.add_callback(show_callback)

        address_string = server.address if server.address else "localhost"

        print(
            "Bokeh app running at:",
            f"http://{address_string}:{server.port}{server.prefix}/",
        )
        server._loop.start()
        server.start()
