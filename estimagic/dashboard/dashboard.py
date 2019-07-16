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
from time import sleep

from bokeh.models.widgets import Tabs

from estimagic.dashboard.convergence_tab import setup_convergence_tab
from estimagic.dashboard.convergence_tab import update_convergence_data


def run_dashboard(doc, queue, start_signal, db_options, start_param_df, start_fitness):
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
            queue to which the updated parameter Series will be supplied later.

        start_signal (Event):
            signal to parent thread to start the optimization.

        db_options (dict):
            dictionary with options. Supported so far:
                rollover (int):
                    How many data points to store, default None.
                port (int):
                    Port at which to display the dashboard.
                evaluations_to_skip (int):
                    Plot at most every (k+1)th evaluation of the criterion function.
                    The default is 0, i.e. plot every evaluation.
                    Example:
                    If this is 9, at most every 10th evaluation's results are plotted.
                time_btw_queue_checks (float):
                    Seconds to wait until checking for new results in the queue.

        start_param_df (pd.DataFrame):
            DataFrame with the start params and information on them.

        start_fitness (float):
            fitness evaluation at the start parameters.

    """
    start_param_df, start_fitness = _trim_start_values(
        start_param_df=start_param_df, start_fitness=start_fitness
    )

    doc, data = _configure_dashboard(
        doc=doc, param_df=start_param_df, start_fitness=start_fitness
    )

    # this thread is necessary to not lock the server
    callbacks = partial(
        _update_dashboard, doc=doc, dashboard_data=data, queue=queue, **db_options
    )
    update_data_thread = Thread(target=callbacks)
    update_data_thread.start()
    start_signal.set()


def _trim_start_values(start_param_df, start_fitness):
    """
    Trim extremely large start values.

    Extremely large (>1e17) starting value cause the dashboard to crash.
    This does not seem to be an issue later on."""
    start_fitness = max(min(start_fitness, 1e15), -1e15)
    start_param_df["value"] = start_param_df["value"].where(
        start_param_df["value"] < 1e15, 1e15
    )
    start_param_df["value"] = start_param_df["value"].where(
        start_param_df["value"] > -1e15, -1e15
    )

    return start_param_df, start_fitness


def _configure_dashboard(doc, param_df, start_fitness):
    """
    Setup the basic dashboard.

    Args:
        doc (bokeh Document):
            bokeh document to be configured

        param_df (pandas DataFrame):
            See :ref:`params`.

        start_fitness (float):
            criterion function evaluated at the initial parameters
    """
    conv_data, tab1 = setup_convergence_tab(param_df, start_fitness)

    tabs = Tabs(tabs=[tab1])
    doc.add_root(tabs)

    return doc, [conv_data]


def _update_dashboard(
    doc, dashboard_data, queue, rollover, evaluations_to_skip, time_btw_queue_checks
):
    """
    Update the dashboard after each call of the criterion function.

    Args:
        doc (bokeh Document):
            Document of the dashboard.

        dashboard_data (list):
            List of datasets used for the dashboard.

        queue (Queue):
            queue to which the updated parameter Series are supplied.

        rollover (int or None):
            How many data points to store, default None.

        evaluations_to_skip (int):
            only plot (at most) every (k+1)th evaluation of the criterion function.
            The default is 0, i.e. plot every evaluation.
            If 9 was supplied, at most every 10th evaluation's results are plotted.
            This does not guarantee that every 9th evaluation is plotted.
            If the queue grows too fast, whenever the queue is checked the latest
            evaluation is plotted and the rest is discarded.

        time_btw_queue_checks (float):
            Seconds to wait until checking for new results in the queue.

    """
    conv_data, = dashboard_data
    while True:
        if queue.qsize() >= evaluations_to_skip + 1:
            new_params, new_fitness = queue.queue.pop()
            queue.queue.clear()
            doc.add_next_tick_callback(
                partial(
                    update_convergence_data,
                    data=conv_data,
                    new_params=new_params,
                    new_fitness=new_fitness,
                    rollover=rollover,
                )
            )

        sleep(time_btw_queue_checks)
