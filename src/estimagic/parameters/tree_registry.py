"""Wrapper around pybaum get_registry to tailor it to estimagic."""
from functools import partial
from itertools import product

import numpy as np
import pandas as pd
from pybaum import get_registry as get_pybaum_registry


def get_registry(extended=False, value_col="value"):
    """Return pytree registry.

    Args:
        extended (bool): If True appends types 'numpy.ndarray', 'pandas.Series' and
            'pandas.DataFrame' to the registry.
        value_col (str): This column is used as the data source for flattening and
            unflattening a pytree. Defaults to 'value'.

    Returns:
        dict: The pytree registry.

    """
    types = ["numpy.ndarray", "pandas.Series"] if extended else None
    registry = get_pybaum_registry(types=types)
    if extended:
        registry[pd.DataFrame] = {
            "flatten": partial(_flatten_df, value_col=value_col),
            "unflatten": partial(_unflatten_df, value_col=value_col),
            "names": _get_df_names,
        }
    return registry


def _flatten_df(df, value_col):
    is_value_col_df = value_col in df
    if is_value_col_df:
        flat = df[value_col].tolist()
    else:
        flat = [np.nan] * len(df)

    aux_data = {
        f"is_{value_col}_df": is_value_col_df,
        "df": df,
    }
    return flat, aux_data


def _unflatten_df(aux_data, leaves, value_col):
    if aux_data[f"is_{value_col}_df"]:
        out = aux_data["df"].assign(**{value_col: leaves})
    else:
        out = pd.DataFrame(
            data=np.array(leaves).reshape(aux_data["df"].shape),
            columns=aux_data["df"].columns,
            index=aux_data["df"].index,
        )
    return out


def _get_df_names(df):
    index_strings = list(df.index.map(_index_element_to_string))
    if "value" in df:
        out = index_strings
    else:
        out = ["_".join([loc, col]) for loc, col in product(index_strings, df.columns)]

    return out


def _index_element_to_string(element):
    if isinstance(element, (tuple, list)):
        as_strings = [str(entry) for entry in element]
        res_string = "_".join(as_strings)
    else:
        res_string = str(element)

    return res_string
