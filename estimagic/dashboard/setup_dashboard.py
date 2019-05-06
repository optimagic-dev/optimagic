"""Functions to setup the dashboard and run a bokeh server with a supplied function."""
import time
from functools import partial
from threading import Thread

from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler
from bokeh.layouts import row
from bokeh.models import Panel
from bokeh.models.widgets import Tabs

from estimagic.dashboard.convergence_plot import setup_convergence_plot
from estimagic.dashboard.server_functions import setup_server
from estimagic.dashboard.server_functions import start_server


def run_with_dashboard(func, notebook):
    """
    Run the *func* in a Bokeh server and return the optimization result.

    Args:
        func (function):
            Python function that takes doc and res as arguments.
            Note: doc must be the first argument of the function
        notebook (bool):
            whether the code is run in a notebook or not
            .. to-do::
                Automatically identify whether code is being run in a notebook or not.

    """
    res = []
    my_func = partial(func, res=res)
    apps = {"/": Application(FunctionHandler(partial(my_func)))}
    server = setup_server(apps=apps, notebook=notebook)

    if notebook is False:
        server_thread = Thread(target=start_server, args=(server, notebook))
        server_thread.start()
        server

        while len(res) == 0:
            time.sleep(0.1)
        return res

    else:
        start_server(server, notebook)


def configure_dashboard(doc, param_df, start_time):
    """
    Setup the basic dashboard.

    Args:
        doc (bokeh Document):
            bokeh document to be configured

        param_df (pandas DataFrame):
            DataFrame with the initial parameter values, constraints etc.

        start_time (datetime):
            time at which the optimization started

    """
    # To-Do: use label based object to store tha data!
    conv_p, param_data = setup_convergence_plot(param_df, start_time)

    tab1 = Panel(child=row([conv_p]), title="Convergence Plot")
    tabs = Tabs(tabs=[tab1])
    doc.add_root(tabs)

    return doc, [param_data]
