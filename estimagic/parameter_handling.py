"""High level functions to convert between parameter vectors.

High level means:

- All functions are save to use directly with user input
- Defaults are filled automatically
- Robust checks and error handling

"""
import numpy as np
import pandas as pd


def add_default_bounds_to_params(params):
    """Fill missing bounds with -np.inf and np.inf."""
    defaults = pd.DataFrame(
        {"lower_bound": -np.inf, "upper_bound": np.inf},
        index=params.index,
    )
    params = params.combine_first(defaults)

    return params


def check_params_index_is_valid(params):
    """Check params has a unique index.

    Args:
        params (pd.DataFrame or list of pd.DataFrames): See :ref:`params`.

    Raises:
        AssertionError: The index contains duplicates.

    """
    if params.index.duplicated().any():
        raise ValueError("No duplicates allowed in the index of params.")

    invalid_bounds = params.query("lower_bound > value | upper_bound < value")

    if len(invalid_bounds) > 0:
        raise ValueError(f"value out of bounds for:\n{invalid_bounds.index}")
