"""Functions to update the dashboard plots."""
from functools import partial

from tornado import gen

from estimagic.dashboard.convergence_plot import data_dict_from_param_values


def add_callbacks(doc, dashboard_data, params_sr, start_time):
    """
    Update the dashboard after each call of the criterion function.

    Args:
        doc (bokeh Document):
            Document of the dashboard.

        dashboard_data (list):
            List of datasets used for the dashboard

        param_sr (pandas Series):
            new parameter values

        start_time (datetime):
            time at which the optimization started

    """
    doc.add_next_tick_callback(
        partial(
            _update_convergence_plot,
            new_params=params_sr,
            data=dashboard_data[0],
            start_time=start_time,
        )
    )


@gen.coroutine
def _update_convergence_plot(new_params, data, start_time):
    to_add = data_dict_from_param_values(new_params, start_time)
    data.stream(to_add)
