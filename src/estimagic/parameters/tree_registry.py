"""Wrapper around pybaum get_registry to tailor it to estimagic."""
from functools import partial
from itertools import product

import numpy as np
import pandas as pd
from pybaum import get_registry as get_pybaum_registry


def get_registry(extended=False, data_col=None):
    """Return pytree registry.

    Special Rules
    -------------
    If extended is True the registry contains pd.DataFrame. In estimagic sometimes a
    data frame represents a 2d object, but sometimes only a 1d object with extra
    information. In the latter case one has to specify which column contains the data
    source using the data_col argument. If the default (None) is used, the function
    looks for a 'value' column. If that is not found the whole data frame is used as a
    source.

    Args:
        extended (bool): If True appends types 'numpy.ndarray', 'pandas.Series' and
            'pandas.DataFrame' to the registry.
        data_col (str): This column is used as the data source in a data frame when
            flattening and unflattening a pytree. Defaults to None; see special rules
            above for default behavior.

    Returns:
        dict: The pytree registry.

    """
    types = ["numpy.ndarray", "pandas.Series"] if extended else None
    registry = get_pybaum_registry(types=types)
    if extended:
        registry[pd.DataFrame] = {
            "flatten": partial(_flatten_df, data_col=data_col),
            "unflatten": partial(_unflatten_df, data_col=data_col),
            "names": _get_df_names,
        }
    return registry


def _flatten_df(df, data_col):
    data_col = "value" if data_col is None and "value" in df else data_col

    is_data_col_df = data_col in df
    if is_data_col_df and data_col is not None:
        flat = df[data_col].tolist()
    elif data_col is not None:
        flat = [np.nan] * len(df)
    else:
        flat = df.to_numpy().flatten().tolist()

    aux_data = {
        f"is_{data_col}_df": is_data_col_df,
        "df": df,
    }
    return flat, aux_data


def _unflatten_df(aux_data, leaves, data_col):
    data_col = "value" if data_col is None and "value" in aux_data["df"] else data_col

    if aux_data[f"is_{data_col}_df"]:
        out = aux_data["df"].assign(**{data_col: leaves})
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
