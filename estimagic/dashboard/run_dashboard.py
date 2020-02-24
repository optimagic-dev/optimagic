import asyncio
import pathlib
from functools import partial
from multiprocessing import Process

from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler
from bokeh.command.util import report_server_init_errors
from bokeh.models import ColumnDataSource
from bokeh.server.server import Server

from estimagic.dashboard.master_app import master_app
from estimagic.dashboard.monitoring_app import monitoring_app
from estimagic.dashboard.utilities import find_free_port
from estimagic.dashboard.utilities import short_name_to_database_path


def run_dashboard_in_separate_process(database_paths, no_browser=False, port=None):
    p = Process(
        target=run_dashboard,
        kwargs={
            "database_paths": database_paths,
            "no_browser": no_browser,
            "port": port,
        },
        daemon=False,
    )
    p.start()
    return p


def run_dashboard(database_paths, no_browser=False, port=None):
    """Start the dashboard pertaining to one or several databases.

    Args:
        database_paths (str or pathlib.Path or list of them):
            Path(s) to an sqlite3 file which typically has the file extension ``.db``.
            See :ref:`logging` for details.
        no_browser (bool, optional):
            Whether or not to open the dashboard in the browser.
        port (int, optional): port where to display the dashboard.

    """
    database_name_to_path, no_browser, port = _process_arguments(
        database_paths=database_paths, no_browser=no_browser, port=port
    )

    master_partialed = partial(master_app, database_name_to_path=database_name_to_path)
    apps = {"/": Application(FunctionHandler(master_partialed))}

    for short_name, full_path in database_name_to_path.items():
        partialed = partial(monitoring_app, short_name=short_name, full_path=full_path)
        apps[f"/{short_name}"] = Application(FunctionHandler(partialed))

    _start_server(apps=apps, port=port, no_browser=no_browser)


def _process_arguments(database_paths, no_browser, port):
    """Check arguments and find free port if none was given.
    Args:
        database_paths (str or pathlib.Path or list of them):
            Path(s) to an sqlite3 file which typically has the file extension ``.db``.
            See :ref:`logging` for details.
        no_browser (bool):
            Whether or not to open the dashboard in the browser.
        port (int or None): port where to display the dashboard.

    Returns:
        database_paths (str or pathlib.Path or list of them):
            Path(s) to an sqlite3 file which typically has the file extension ``.db``.
            See :ref:`logging` for details.
        no_browser (bool):
            Whether or not to open the dashboard in the browser.
        port (int): port where to display the dashboard.
    """
    if not isinstance(database_paths, (list, tuple)):
        database_paths = [database_paths]

    for single_database_path in database_paths:
        if not isinstance(single_database_path, (str, pathlib.Path)):
            raise TypeError(
                f"database_paths must be string or pathlib.Path. ",
                "You supplied {type(single_database_path)}.",
            )
    database_name_to_path = short_name_to_database_path(path_list=database_paths)

    if not isinstance(no_browser, bool):
        raise TypeError(f"no_browser must be a bool. You supplied {type(no_browser)}.")

    if port is None:
        port = find_free_port()
    elif not isinstance(port, int):
        raise TypeError(f"port must be an integer. You supplied {type(port)}.")

    return database_name_to_path, no_browser, port


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
