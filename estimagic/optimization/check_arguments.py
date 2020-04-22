import typing
from pathlib import Path

import pandas as pd


def check_arguments(arguments):
    valid_types = {
        "general_options": dict,
        "dashboard": bool,
        "dash_options": dict,
        "criterion": typing.Callable,
        "params": pd.DataFrame,
        "algorithm": str,
        "algo_options": dict,
        "gradient": (typing.Callable, type(None)),
        "gradient_kwargs": dict,
        "gradient_options": (dict, type(None)),
        "log_options": dict,
        "criterion_kwargs": dict,
        "constraints": list,
        "logging": (bool, Path),
    }

    for args in arguments:
        for arg in args:
            if not isinstance(args[arg], valid_types[arg]):
                raise TypeError(
                    f"Argument '{arg}' is {args[arg]} which is not {valid_types[arg]}."
                )

    # Sanity check if there some False and some paths that the paths are not the same.
    only_paths = [
        args["logging"] for args in arguments if isinstance(args["logging"], Path)
    ]
    if len(set(only_paths)) != len(only_paths):
        raise ValueError("Paths to databases cannot be identical.")
