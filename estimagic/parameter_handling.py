"""High level functions to convert between parameter vectors.

High level means:

- All functions are save to use directly with user input
- Defaults are filled automatically
- Robust checks and error handling

"""
import numpy as np
import pandas as pd


def get_reparametrize_functions(params, constraints, scaling_factor, scaling_offset):
    """Construct functions to map between internal and external parameters.

    All required information is partialed into the functions.

    Args:
        params (pandas.DataFrame): See :ref:`params`.
        constraints (list): List of constraint dictionaries.
        scaling_factor (np.ndarray or None): If None, no scaling factor is used.
        scaling_offset (np.ndarray or None): If None, no scaling offset is used

    Returns:
        func: Function that maps an external parameter vector to an internal one
        func: Function that maps an internal parameter vector to an external one

    """
    if (scaling_factor, scaling_offset) != (None, None):
        raise NotImplementedError()

    params = add_default_bounds_to_params(params)
    check_params_are_valid(params)


def get_derivative_conversion_functions(
    params, constraints, scaling_factor, scaling_offset
):
    """Construct functions to map between internal and external derivatives.

    All required information is partialed into the functions.

    Args:
        params (pandas.DataFrame): See :ref:`params`.
        constraints (list): List of constraint dictionaries.
        scaling_factor (np.ndarray or None): If None, no scaling factor is used.
        scaling_offset (np.ndarray or None): If None, no scaling offset is used


    Returns:
        func: Function that maps an external derivative to an internal one
        func: Function that maps an internal derivative to an external one

    """
    if (scaling_factor, scaling_offset) != (None, None):
        raise NotImplementedError()

    pass


def add_default_bounds_to_params(params):
    """Fill missing bounds with -np.inf and np.inf."""
    defaults = pd.DataFrame(
        {"lower_bound": -np.inf, "upper_bound": np.inf},
        index=params.index,
    )
    params = params.combine_first(defaults)

    return params


def check_params_are_valid(params):
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
