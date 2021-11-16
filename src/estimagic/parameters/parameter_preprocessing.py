"""High level functions to convert between parameter vectors.

High level means:

- Functions are save to use directly with user input (if applicable)
- Defaults are filled automatically (if applicable)
- Functions include robustness checks and error handling

"""
import numpy as np


def add_default_bounds_to_params(params):
    """Fill missing bounds with -np.inf and np.inf."""
    params = params.copy()

    if "lower_bound" not in params:
        params["lower_bound"] = -np.inf
    else:
        params["lower_bound"].fillna(-np.inf, inplace=True)

    if "upper_bound" not in params:
        params["upper_bound"] = np.inf
    else:
        params["upper_bound"].fillna(np.inf, inplace=True)

    return params


def check_params_are_valid(params):
    """Check params has a unique index and valid bounds.

    Args:
        params (pd.DataFrame or list of pd.DataFrames): See :ref:`params`.

    Raises:
        AssertionError: The index contains duplicates.

    """
    if params.index.duplicated().any():
        raise ValueError("No duplicates allowed in the index of params.")

    invalid_bounds = params[params["lower_bound"].values > params["upper_bound"].values]

    if len(invalid_bounds) > 0:
        raise ValueError(f"value out of bounds for:\n{invalid_bounds.index}")
