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

from bokeh.layouts import column
from bokeh.models import Panel
from bokeh.models.widgets import Tabs

from estimagic.dashboard.convergence_plot import setup_convergence_tab
from estimagic.dashboard.convergence_plot import update_convergence_tab


def run_dashboard(doc, queue, db_options):
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

        db_options (dict):
            dictionary with options. Supported so far:
                rollover (int):
                    How many data points to store, default None.

    """
    rollover = db_options["rollover"]
    param_df, initial_fitness = queue.get()

    doc, data = _configure_dashboard(
        doc=doc, param_df=param_df, initial_fitness=initial_fitness
    )

    # this thread is necessary to not lock the server
    callbacks = partial(
        _update_dashboard, doc=doc, dashboard_data=data, queue=queue, rollover=rollover
    )
    update_data_thread = Thread(target=callbacks)
    update_data_thread.start()


def _configure_dashboard(doc, param_df, initial_fitness):
    """
    Setup the basic dashboard.

    Args:
        doc (bokeh Document):
            bokeh document to be configured

        param_df (pandas DataFrame):
            See :ref:`params_df`.

        initial_fitness (float):
            criterion function evaluated at the initial parameters
    """
    plots, data = setup_convergence_tab(param_df, initial_fitness)

    tab1 = Panel(child=column(plots), title="Convergence Plots")
    tabs = Tabs(tabs=[tab1])
    doc.add_root(tabs)

    # To-Do: use label based object to store tha data!
    return doc, data


def _update_dashboard(doc, dashboard_data, queue, rollover):
    """
    Update the dashboard after each call of the criterion function.

    Args:
        doc (bokeh Document):
            Document of the dashboard.

        dashboard_data (list):
            List of datasets used for the dashboard.

        queue (Queue):
            queue to which the updated parameter Series are supplied.

    """
    while True:
        update_convergence_tab(
            doc=doc, queue=queue, datasets=dashboard_data, rollover=rollover
        )
