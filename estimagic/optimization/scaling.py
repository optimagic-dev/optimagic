"""Functions to calculate and apply scaling factors and offsets.

All scaling operations are applied to internal parameter vectors, i.e.
after the reparametrizations for constraints have been applied.

The order of applying the operations is the following:

to internal: (x - scaling_offset) / scaling_factor
from internal: x * scaling_factor + scaling_offset

"""
import numpy as np  # noqa
import pandas as pd  # noqa


def calculate_scaling_factor_and_offset(criterion, params, constraints):

    pass
