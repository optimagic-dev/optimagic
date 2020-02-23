import asyncio
import pathlib
import socket
from contextlib import closing
from functools import partial
from multiprocessing import Process
from pathlib import Path

from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler
from bokeh.command.util import report_server_init_errors
from bokeh.models import ColumnDataSource
from bokeh.server.server import Server

from estimagic.dashboard.master_app import master_app
from estimagic.dashboard.monitoring_app import monitoring_app
from estimagic.logging.create_database import load_database
from estimagic.logging.read_database import read_last_iterations
from estimagic.logging.read_database import read_scalar_field


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
    database_paths, no_browser, port = _process_arguments(
        database_paths=database_paths, no_browser=no_browser, port=port
    )

    elements_dict = _common_elements_dict(database_paths=database_paths)

    master_partialed = partial(master_app, elements_dict=elements_dict)
    apps = {"/": Application(FunctionHandler(master_partialed))}
    for nice_database_name, inner_dict in elements_dict.items():
        partialed = partial(monitoring_app, inner_dict=inner_dict)
        apps[f"/{nice_database_name}"] = Application(FunctionHandler(partialed))

    _start_server(apps=apps, port=port, no_browser=no_browser)


def _process_arguments(database_paths, no_browser, port):
    if not isinstance(database_paths, (list, tuple)):
        database_paths = [database_paths]

    for single_database_path in database_paths:
        if not isinstance(single_database_path, (str, pathlib.Path)):
            raise TypeError(
                f"database_paths must be string or pathlib.Path. ",
                "You supplied {type(single_database_path)}.",
            )

    if not isinstance(no_browser, bool):
        raise TypeError(f"no_browser must be a bool. You supplied {type(no_browser)}")

    if port is None:
        port = _find_free_port()

    return database_paths, no_browser, port


def _find_free_port():
    """
    Find a free port on the localhost.

    Adapted from https://stackoverflow.com/a/45690594
    """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("localhost", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def _nice_names(path_list):
    """Generate short but unique names from each path.

    Args:
        path_list (list): List of strings or pathlib.path.

    Returns:
        list: List of strings with names.

    Example:

    >>> pl = ["bla/blubb/blabb.db", "a/b", "bla/blabb"]
    >>> _nice_names(pl)
    ['blubb/blabb', 'b', 'bla/blabb']

    """
    path_list = [Path(p).resolve().with_suffix("") for p in path_list]
    # The assert statement makes sure that the while loop terminates
    assert len(set(path_list)) == len(
        path_list
    ), "path_list must not contain duplicates."
    short_names = []
    for path in path_list:
        parts = tuple(reversed(path.parts))
        needed_parts = 1
        candidate = parts[:needed_parts]
        while _name_clash(candidate, path_list):
            needed_parts += 1
            candidate = parts[:needed_parts]

        short_names.append("/".join(reversed(candidate)))
    return short_names


def _name_clash(candidate, path_list, allowed_occurences=1):
    """Determine if candidate leads to a name clash.

    Args:
        candidate (tuple): tuple with parts of a path.
        path_list (list): List of pathlib.path
        allowed_occurences (int): How often a name can occur before
            we call it a clash.

    Returns:
        bool

    """
    duplicate_counter = -allowed_occurences
    for path in path_list:
        parts = tuple(reversed(path.parts))
        if len(parts) >= len(candidate) and parts[: len(candidate)] == candidate:
            duplicate_counter += 1
    return duplicate_counter > 0


def _common_elements_dict(database_paths):
    """For each database map their tables to ColumnDataSources.

    Args:
        database_paths (list): list of the paths to the databases.

    Returns:
        elements_dict (dict): nested dictionary.
            The outer keys are the shortened paths to the databases.
            The inner keys are "nice_database_name", "full_path", "db_options",
            "start_params" and the table names "criterion_history" and "params_history".
            The inner values are ColumnDataSources with the initially available data
            for the table names.
    """
    elements_dict = {}
    database_names = _nice_names(database_paths)
    for nice_database_name, full_path in zip(database_names, database_paths):
        inner_dict = {
            "nice_database_name": nice_database_name,
            "full_path": full_path,
        }
        inner_dict = _update_inner_dict(inner_dict, full_path)
        elements_dict[nice_database_name] = inner_dict
    return elements_dict


def _update_inner_dict(inner_dict, full_path):
    database = load_database(full_path)

    full_dict = inner_dict.copy()
    full_dict["start_params"] = read_scalar_field(database, "start_params")
    full_dict["db_options"] = read_scalar_field(database, "db_options")

    data_dict = read_last_iterations(
        database=database,
        tables=["criterion_history", "params_history"],
        n=-1,
        return_type="bokeh",
    )
    for table_name, data in data_dict.items():
        nice_database_name = full_dict["nice_database_name"]
        full_dict[table_name] = ColumnDataSource(
            data=data, name=f"{nice_database_name}_{table_name}_cds"
        )
    return full_dict


def _start_server(apps, port, no_browser):
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
