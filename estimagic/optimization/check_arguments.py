import types
from pathlib import Path

import pandas as pd


def check_arguments(arguments):
    valid_types = {
        "general_options": dict,
        "dashboard": bool,
        "db_options": dict,
        "criterion": types.FunctionType,
        "params": pd.DataFrame,
        "algorithm": str,
        "algo_options": dict,
        "gradient": (types.FunctionType, type(None)),
        "gradient_options": dict,
        "log_options": dict,
        "criterion_kwargs": dict,
        "constraints": list,
    }

    for args in arguments:
        assert all(isinstance(args[arg], valid_types[arg]) for arg in args)

    # Sanity check if there some False and some paths that the paths are not the same.
    only_paths = [path for path in arguments["logging"] if isinstance(path, Path)]
    if len(set(only_paths)) != len(only_paths):
        raise ValueError("Paths to databases cannot be identical.")
