import asyncio
import pathlib
import warnings
from functools import partial
from multiprocessing import Process

from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler
from bokeh.command.util import report_server_init_errors
from bokeh.server.server import Server

from estimagic.dashboard.master_app import master_app
from estimagic.dashboard.monitoring_app import monitoring_app
from estimagic.dashboard.utilities import create_short_database_names
from estimagic.dashboard.utilities import find_free_port
from estimagic.logging.create_database import load_database
from estimagic.logging.read_database import read_scalar_field


def run_dashboard_in_separate_process(database_paths):
    """Run the dashboard in a separate process.

    Args:
        database_paths (str or pathlib.Path or list of them):
            Path(s) to an sqlite3 file which typically has the file extension ``.db``.
            See :ref:`logging` for details.

    Returns:
        p (multiprocessing.Process): Process in which the dashboard is running.

    """
    p = Process(
        target=run_dashboard, kwargs={"database_paths": database_paths}, daemon=False
    )
    p.start()
    return p


def run_dashboard(database_paths, no_browser=None, port=None):
    """Start the dashboard pertaining to one or several databases.

    Args:
        database_paths (str or pathlib.Path or list of them):
            Path(s) to an sqlite3 file which typically has the file extension ``.db``.
            See :ref:`logging` for details.
        no_browser (bool, optional):
            Whether or not to open the dashboard in the browser.
        port (int, optional): port where to display the dashboard.

    """
    database_name_to_path, no_browser, port = _process_dashboard_args(
        database_paths=database_paths, no_browser=no_browser, port=port
    )

    session_data = _create_session_data(database_name_to_path)

    master_app_func = partial(
        master_app,
        database_name_to_path=database_name_to_path,
        session_data=session_data,
    )
    apps = {"/": Application(FunctionHandler(master_app_func))}

    for database_name in database_name_to_path.keys():
        partialed = partial(
            monitoring_app,
            database_name=database_name,
            session_data=session_data[database_name],
        )
        apps[f"/{database_name}"] = Application(FunctionHandler(partialed))

    if len(database_name_to_path) == 1:
        path_to_open = f"/{list(database_name_to_path.keys())[0]}"
    else:
        path_to_open = "/"

    _start_server(
        apps=apps, port=port, no_browser=no_browser, path_to_open=path_to_open
    )


def _process_dashboard_args(database_paths, no_browser, port):
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
    database_name_to_path = create_short_database_names(path_list=database_paths)

    all_dash_options = []
    for single_database_path in database_paths:
        database = load_database(single_database_path)
        dash_options = read_scalar_field(database, "dash_options")
        all_dash_options.append(dash_options)

    if port is None:
        ports = {
            d.pop("port", None)
            for d in all_dash_options
            if d.pop("port", None) is not None
        }
        if len(ports) == 0:
            port = find_free_port()
        else:
            port = ports.pop()
            if len(ports) > 1:
                warnings.warn(f"You supplied more than one port. {port} will be used.")

    if not isinstance(port, int):
        raise TypeError(f"port must be an integer. You supplied {type(port)}.")

    if no_browser is None:
        no_browser_vals = {d.pop("no_browser", False) for d in all_dash_options}
        no_browser = no_browser_vals.pop()
        if len(no_browser_vals) > 1:
            no_browser = False
            warnings.warn(
                "You supplied both True and False for no_browser. It is set to False."
            )
    return database_name_to_path, no_browser, port


def _create_session_data(database_name_to_path):
    session_data = {"master_app": {}}
    for database_name, database_path in database_name_to_path.items():
        session_data[database_name] = {
            "last_retrieved": 0,
            "database_path": database_path,
            "callbacks": {},
        }
    return session_data


def _start_server(apps, port, no_browser, path_to_open):
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
                server.show(path_to_open)

            server.io_loop.add_callback(show_callback)

        address_string = server.address if server.address else "localhost"

        print(
            "Bokeh app running at:",
            f"http://{address_string}:{server.port}{server.prefix}/",
        )
        server._loop.start()
        server.start()
