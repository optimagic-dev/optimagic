import typing
from pathlib import Path

from estimagic.shared.check_option_dicts import check_numdiff_options


def check_optimize_kwargs(**kwargs):
    valid_types = {
        "direction": str,
        "criterion": typing.Callable,
        "algorithm": (str, typing.Callable),
        "criterion_kwargs": dict,
        "constraints": (list, dict),
        "algo_options": dict,
        "derivative": (type(None), typing.Callable, dict),
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
        if arg in valid_types:
            if not isinstance(kwargs[arg], valid_types[arg]):
                raise TypeError(
                    f"Argument '{arg}' must be {valid_types[arg]} not {kwargs[arg]}."
                )

    if kwargs["direction"] not in ["minimize", "maximize"]:
        raise ValueError("diretion must be 'minimize' or 'maximize'")

    if kwargs["error_handling"] not in ["raise", "continue"]:
        raise ValueError("error_handling must be 'raise' or 'continue'")

    check_numdiff_options(kwargs["numdiff_options"], "optimization")
