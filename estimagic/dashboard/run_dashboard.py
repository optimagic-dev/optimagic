import asyncio
import pathlib
from functools import partial

from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler
from bokeh.command.util import report_server_init_errors
from bokeh.server.server import Server

from estimagic.dashboard.master_app import master_app
from estimagic.dashboard.monitoring_app import monitoring_app
from estimagic.dashboard.utilities import create_short_database_names
from estimagic.dashboard.utilities import find_free_port


def run_dashboard(database_paths, no_browser, port, rollover, jump):
    """Start the dashboard pertaining to one or several databases.

    Args:
        database_paths (str or pathlib.Path or list): Path(s) to an sqlite3 file which
            typically has the file extension ``.db``.
        no_browser (bool): If True the dashboard does not open in the browser.
        port (int): Port where to display the dashboard.
        rollover (int): After how many iterations the convergence plots are truncated.
        jump (bool): If True the dashboard will jump directly to the last `rollover`
            observations and not display the full history.

    """
    database_name_to_path = _process_database_paths(database_paths)

    port = find_free_port() if port is None else port
    port = int(port)
    rollover = int(rollover)

    session_data = _create_session_data(database_name_to_path)

    master_app_func = partial(
        master_app,
        database_name_to_path=database_name_to_path,
        session_data=session_data,
    )
    apps = {"/": Application(FunctionHandler(master_app_func))}

    for database_name in database_name_to_path:
        partialed = partial(
            monitoring_app,
            database_name=database_name,
            session_data=session_data[database_name],
            rollover=rollover,
            jump=jump,
        )
        apps[f"/{database_name}"] = Application(FunctionHandler(partialed))

    if len(database_name_to_path) == 1:
        path_to_open = f"/{list(database_name_to_path)[0]}"
    else:
        path_to_open = "/"

    _start_server(
        apps=apps, port=port, no_browser=no_browser, path_to_open=path_to_open
    )


def _process_database_paths(database_paths):
    """Process the database paths.

    Args:
        database_paths (str or pathlib.Path or list of them): Path(s) to an sqlite3
            file which typically has the file extension ``.db``.

    Returns:
        database_paths (str or pathlib.Path or list of them):
            Path(s) to an sqlite3 file which typically has the file extension ``.db``.

    """
    if not isinstance(database_paths, (list, tuple)):
        database_paths = [database_paths]

    for single_database_path in database_paths:
        if not isinstance(single_database_path, (str, pathlib.Path)):
            raise TypeError(
                "database_paths must be string or pathlib.Path. ",
                f"You supplied {type(single_database_path)}.",
            )
    database_name_to_path = create_short_database_names(path_list=database_paths)

    return database_name_to_path


def _create_session_data(database_name_to_path):
    """Create a nested dictionary with info to be passed between and within bokeh apps.

    Args:
        short_name_to_path (dict): mapping from the new unique names to their full path.

    Returns:
        session_data (dict): Infos to be passed between and within apps.
            It contains one entry for the master app and one for each monitoring app.
            The keys of the monitoring app's entries are:
            - last_retrieved (int): last iteration currently in the ColumnDataSource.
            - database_path (str or pathlib.Path)
            - callbacks (dict): dictionary to be populated with callbacks.

    """
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
