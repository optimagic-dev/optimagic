import typing
import warnings
from pathlib import Path

import pandas as pd
from estimagic.shared.check_option_dicts import check_numdiff_options


def check_optimize_kwargs(**kwargs):
    valid_types = {
        "direction": str,
        "criterion": typing.Callable,
        "params": pd.DataFrame,
        "algorithm": (str, typing.Callable),
        "criterion_kwargs": dict,
        "constraints": list,
        "algo_options": dict,
        "derivative": (type(None), typing.Callable),
        "derivative_kwargs": dict,
        "criterion_and_derivative": (type(None), typing.Callable),
        "criterion_and_derivative_kwargs": dict,
        "numdiff_options": dict,
        "logging": (bool, Path),
        "log_options": dict,
        "error_handling": str,
        "error_penalty": dict,
        "cache_size": (int, float),
        "scaling": bool,
        "scaling_options": dict,
        "multistart": bool,
        "multistart_options": dict,
    }

    for arg in kwargs:
        if not isinstance(kwargs[arg], valid_types[arg]):
            raise TypeError(
                f"Argument '{arg}' is {kwargs[arg]} which is not {valid_types[arg]}."
            )

    if kwargs["direction"] not in ["minimize", "maximize"]:
        raise ValueError("diretion must be 'minimize' or 'maximize'")

    parcols = kwargs["params"].columns
    if "value" not in parcols:
        raise ValueError("The params DataFrame must contain a 'value' column.")

    if "lower" in parcols and "lower_bounds" not in parcols:
        msg = "There is a column 'lower' in params. Did you mean 'lower_bounds'?"
        warnings.warn(msg)

    if "upper" in parcols and "upper_bounds" not in parcols:
        msg = "There is a column 'upper' in in params. Did you mean 'upper_bounds'?"
        warnings.warn(msg)

    if kwargs["error_handling"] not in ["raise", "continue"]:
        raise ValueError("error_handling must be 'raise' or 'continue'")

    check_numdiff_options(kwargs["numdiff_options"], "optimization")
