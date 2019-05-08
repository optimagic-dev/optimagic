"""
Functions that setup and update the dashboard.

These are the main functions of the dashboard.

The functions defined here are not executed directly.
Instead a BokehServer is setup and run to execute the main function run_dashboard.
The functions to setup and run the BokehServer can be found in server_functions.

The functions to build and update each tab can be found in the respective module.

"""
from functools import partial
from threading import Thread

from bokeh.layouts import row
from bokeh.models import Panel
from bokeh.models.widgets import Tabs

from estimagic.dashboard.convergence_plot import setup_convergence_plot
from estimagic.dashboard.convergence_plot import update_convergence_plot


def run_dashboard(doc, queue):
    """Configure the dashboard and update constantly as new parameters arrive.

    This is the main function that is supplied to the bokeh FunctionHandler.
    Note that the first argument must be doc for the FunctionHandler
    to create the bokeh Applications correctly.

    Args:
        doc (bokeh Document):
            document instance where the Dashboard will be stored.
            Note this must stay the first argument for the bokeh FunctionHandler
            to work properly.

        queue (Queue):
            queue to which originally the parameters DataFrame is supplied and to which
            the updated parameter Series will be supplied later.

    """
    param_df = queue.get()

    doc, data = _setup_dashboard(doc=doc, param_df=param_df)

    # this thread is necessary to not lock the server
    callbacks = partial(_update_dashboard, doc=doc, dashboard_data=data, queue=queue)
    update_data_thread = Thread(target=callbacks)
    update_data_thread.start()


def _setup_dashboard(doc, param_df):
    """
    Setup the basic dashboard.

    Args:
        doc (bokeh Document):
            bokeh document to be configured

        param_df (pandas DataFrame):
            See :ref:`params_df`.
    """
    conv_p, param_data = setup_convergence_plot(param_df)

    tab1 = Panel(child=row([conv_p]), title="Convergence Plot")
    tabs = Tabs(tabs=[tab1])
    doc.add_root(tabs)

    # To-Do: use label based object to store tha data!
    return doc, [param_data]


def _update_dashboard(doc, dashboard_data, queue):
    """
    Update the dashboard after each call of the criterion function.

    Args:
        doc (bokeh Document):
            Document of the dashboard.

        dashboard_data (list):
            List of datasets used for the dashboard.

        param_sr (pandas Series):
            new parameter values

    """
    param_data, = dashboard_data

    while True:
        update_convergence_plot(doc=doc, queue=queue, param_data=param_data)
