"""Functions to update the dashboard plots."""
from functools import partial
from random import random

from tornado import gen


def add_callbacks(doc, dashboard_data, params_sr):
    x, y = random(), random()
    doc.add_next_tick_callback(partial(_update, x=x, y=y, data=dashboard_data[0]))
    doc.add_next_tick_callback(partial(_update, x=x, y=y, data=dashboard_data[1]))


@gen.coroutine
def _update(x, y, data):
    data.stream({"x": [x], "y": [y]})
