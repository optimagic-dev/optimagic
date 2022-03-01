"""Wrapper around pybaum get_registry to tailor it to estimagic."""
from itertools import product

import numpy as np
import pandas as pd
from pybaum import get_registry as get_pybaum_registry


def get_registry(extended=False):
    types = ["numpy.ndarray", "pandas.Series"] if extended else None
    registry = get_pybaum_registry(types=types)
    if extended:
        registry[pd.DataFrame] = {
            "flatten": _flatten_df,
            "unflatten": _unflatten_df,
            "names": _get_df_names,
        }
    return registry


def _flatten_df(df):
    if "value" in df:
        flat = df["value"].tolist()
        aux_data = {
            "is_value_df": True,
            "df": df,
        }
    else:
        flat = df.to_numpy().flatten().tolist()
        aux_data = {
            "is_value_df": False,
            "df": df,
        }

    return flat, aux_data


def _unflatten_df(aux_data, leaves):
    if aux_data["is_value_df"]:
        out = aux_data["df"].assign(value=leaves)
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
