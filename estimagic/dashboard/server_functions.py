"""Functions for setting up and running the BokehServer."""
from bokeh.command.util import report_server_init_errors
from bokeh.server.server import Server


def setup_server(apps, notebook, port=5478):
    """
    Setup the server similarly to bokeh serve subcommand.

    In contrast to bokeh serve this supports being called from within a notebook.
    It also allows the server to be run in a separate thread while a main script
    is waiting for the output.

    Args:
        apps (dict):
            dictionary mapping suffixes to the address to Applications

        notebook (bool):
            whether the function is called in a jupyter notebook

        port (int):
            port where to host the BokehServer

    """
    # this is adapted from bokeh.subcommands.serve
    with report_server_init_errors(port=port):
        server = Server(apps, port=port)
        if notebook is False:
            _prepare_server_without_notebook(apps, server)
            _announce_server_addresses(apps, server)
        return server


def start_server(server, notebook):
    """Start the server and an IOLoop if necessary."""
    if notebook is True:
        server.start()
        server.show("/")
    else:
        server._loop.start()
        server.start()


def _prepare_server_without_notebook(apps, server):
    def show_callback():
        for route in apps.keys():
            server.show(route)

    server.io_loop.add_callback(show_callback)


def _announce_server_addresses(apps, server):
    address_string = "localhost"
    if server.address is not None and server.address != "":
        address_string = server.address

    for route in sorted(apps.keys()):
        url = "http://{}:{}{}{}".format(
            address_string, server.port, server.prefix, route
        )
        print("Bokeh app running at: " + url)
