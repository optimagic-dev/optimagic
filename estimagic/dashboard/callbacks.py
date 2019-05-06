"""Functions to update the dashboard plots."""
from functools import partial

from tornado import gen

from estimagic.dashboard.setup_dashboard import _data_dict_from_param_values


def add_callbacks(doc, dashboard_data, params_sr, start_time):
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
    data.stream(_data_dict_from_param_values(new_params, start_time))
