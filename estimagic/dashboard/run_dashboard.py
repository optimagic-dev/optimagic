""""""
import pathlib
import socket
from contextlib import closing

from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler
from bokeh.command.util import report_server_init_errors
from bokeh.server.server import Server
from monitoring_app import monitoring_app


def run_dashboard(databases, no_browser=False, port=None):
    """Start the dashboard pertaining to one or several databases.

    Args:
        databases (str or pathlib.Path or list of them):
            Path(s) to an sqlite3 file which typically has the file extension ``.db``.
            See :ref:`logging` for details.
        no_browser (bool, optional):
            Whether or not to open the dashboard in the browser.
        port (int, optional): port where to display the dashboard.
    """
    databases, no_browser, port = _process_arguments(databases, no_browser, port)

    apps = {"/": Application(FunctionHandler(monitoring_app))}

    # this is adapted from bokeh.subcommands.serve
    with report_server_init_errors(port=port):
        server = Server(apps, port=port)

        # On a remote server, we do not want to start the dashboard here.
        if not no_browser:

            def show_callback():
                server.show("/")

            server.io_loop.add_callback(show_callback)

        address_string = server.address if server.address else "localhost"

        for route in sorted(apps):
            url = "http://{}:{}{}{}".format(
                address_string, server.port, server.prefix, route
            )
            print("Bokeh app running at:", url)
        server._loop.start()
        server.start()


def _process_arguments(databases, no_browser, port):
    if not isinstance(databases, (list, tuple)):
        databases = [databases]

    for db in databases:
        if not isinstance(db, (str, pathlib.Path)):
            raise TypeError(
                f"Databases must be string or pathlib.Path. You supplied {type(db)}."
            )

    if not isinstance(no_browser, bool):
        raise TypeError(f"no_browser must be a bool. You supplied {type(no_browser)}")

    if port is None:
        port = _find_free_port()

    return databases, no_browser, port


def _find_free_port():
    """
    Find a free port on the localhost.

    Adapted from https://stackoverflow.com/a/45690594
    """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("localhost", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
