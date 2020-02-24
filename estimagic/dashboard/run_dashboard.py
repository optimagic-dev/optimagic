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
from estimagic.dashboard.utilities import short_and_unique_optimization_names
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
    for nice_database_name, single_optim_info in elements_dict.items():
        partialed = partial(monitoring_app, single_optim_info=single_optim_info)
        apps[f"/{nice_database_name}"] = Application(FunctionHandler(partialed))

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

    if not isinstance(no_browser, bool):
        raise TypeError(f"no_browser must be a bool. You supplied {type(no_browser)}.")

    if port is None:
        port = find_free_port()
    elif not isinstance(port, int):
        raise TypeError(f"port must be an integer. You supplied {type(port)}.")

    return database_paths, no_browser, port


def _common_elements_dict(database_paths):
    """For each database map their tables to ColumnDataSources.

    Args:
        database_paths (list): list of the paths to the databases.

    Returns:
        elements_dict (dict): nested dictionary.
            The outer keys are the nice names of the databases.
            The inner keys are "nice_database_name", "full_path", "db_options",
            "start_params", "criterion_history" and "params_history".
            The inner values are ColumnDataSources with the initially available data
            for "criterion_history" and "params_history".

    """
    elements_dict = {}
    database_names = short_and_unique_optimization_names(path_list=database_paths)
    for nice_database_name, full_path in zip(database_names, database_paths):
        single_optim_info = {
            "nice_database_name": nice_database_name,
            "full_path": full_path,
        }
        single_optim_info = _add_from_database(
            single_optim_info=single_optim_info, full_path=full_path
        )
        elements_dict[nice_database_name] = single_optim_info
    return elements_dict


def _add_from_database(single_optim_info, full_path):
    """Add entries to the info dictionary on one optimization from its database.

    Args:
        single_optim_info (dict): dictionary with information on one optimization
        full_path (str or pathlib.Path): path to the database

    Returns:
        full_dict (dict):
            copy of the single_optim_info with added entries
            "start_params", "db_options", "criterion_history", "params_history".

    """
    database = load_database(full_path)

    full_dict = single_optim_info.copy()
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
