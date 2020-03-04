from pathlib import Path

import sqlalchemy


def broadcast_arguments(**arguments):
    """Broadcast arguments.

    All passed keyword arguments are broadcasted to the argument with the most elements.

    If `n` is the maximum number of elements per keyword arguments, single elements are
    duplicated `n`-times. Arguments with `n` elements are matched by positions.

    """
    dict_args = [
        "criterion_kwargs",
        "algo_options",
        "log_options",
        "dash_options",
        "general_options",
    ]
    for arg in dict_args:
        if arg in arguments and arguments[arg] is None:
            arguments[arg] = {}

    if "constraints" in arguments and arguments["constraints"] is None:
        arguments["constraints"] = []

    # Remove logging as it needs special treatment.
    logging = arguments.pop("logging", False)

    # Turn all scalar arguments to lists with a single element.
    for key, value in arguments.items():
        if key == "constraints":
            if not arguments["constraints"]:
                arguments["constraints"] = [[]]
            elif isinstance(value[0], dict):
                arguments["constraints"] = [value]
            elif isinstance(value[0], list):
                pass
        else:
            if not isinstance(value, list):
                arguments[key] = [value]

    n_optimizations = max(len(value) for value in arguments.values())

    # Broadcast to the correct length.
    for key, value in arguments.items():
        if 1 == len(value) < n_optimizations:
            arguments[key] = value * n_optimizations
        elif len(value) == n_optimizations:
            pass
        else:
            raise ValueError(
                f"Argument '{key}' cannot be broadcasted to {n_optimizations} "
                "optimizations."
            )

    logging = _process_path_or_metadata_for_logging(logging, n_optimizations)
    arguments.update({"logging": logging})

    # Convert arguments from dictionary of lists to lists of dictionaries.
    arguments = [
        {key: value[i] for key, value in arguments.items()}
        for i in range(n_optimizations)
    ]

    return arguments


def _process_path_or_metadata_for_logging(logging, n_optimizations):
    """Process paths or sqlalchemy.MetaData object to databases.

    `logging` can be a single value in which case it becomes an iterable with a single
    value to simplify the processing.

    `logging` can also be a `list` or a `tuple` in which case

    - a single path receives a number as a suffix for more than one optimizations.
    - everything else is parsed according to the following transformation rules.

    Every candidate value for `logging` is transformed according to the following rules.
    If `logging` is a `str`, it is converted to :class:`pathlib.Path` and a
    :class:`pathlib.Path` is left unchanged. Everything which evaluates to `False` is
    set to `False` which turns logging off.

    Raises
    ------
    ValueError
        If `n_logging` is not one and not `n_optimizations`.
    ValueError
        If there are identical paths in `logging`.

    Example
    -------
    >>> _process_path_or_metadata_for_logging(False, 1)
    [False]
    >>> _process_path_or_metadata_for_logging(False, 2)
    [False, False]
    >>> _process_path_or_metadata_for_logging([False], 2)
    [False, False]

    """
    if not isinstance(logging, (tuple, list)):
        # Handle the special case, where we have one path and multiple optimizations.
        # Then, add numbers as suffixes to the path.
        if n_optimizations >= 2 and isinstance(logging, (str, Path)):
            path = Path(logging).absolute()
            logging = [
                path.parent / (path.stem + f"_{i}.db") for i in range(n_optimizations)
            ]
        else:
            logging = [logging] * n_optimizations
    else:
        if len(logging) == 1:
            logging = logging * n_optimizations
        elif len(logging) == n_optimizations:
            pass
        else:
            raise ValueError(
                f"logging has {len(logging)} entries and there are {n_optimizations} "
                "optimizations. Cannot harmonize entries."
            )

    logging = [_process_path_or_metadata(path) for path in logging]

    return logging


def _process_path_or_metadata(path):
    """Processes an individual path."""
    if not path:
        path = False
    elif isinstance(path, (str, Path)):
        path = Path(path).absolute()
    elif isinstance(path, sqlalchemy.MetaData):
        pass
    else:
        raise ValueError(
            "logging has to be a str/pathlib.Path/sqlalchemy.MetaData, a list of the "
            "same elements or False."
        )

    return path
