from pathlib import Path


def broadcast_arguments(**arguments):
    """Broadcast arguments.

    All passed keyword arguments are broadcasted to the argument with the most elements.

    If `n` is the maximum number of elements per keyword arguments, single elements are
    duplicated `n`-times. Arguments with `n` elements are matched by positions.

    """
    dict_args = [
        "criterion_kwargs",
        "algo_options",
        "derivative_kwargs",
        "criterion_and_derivative_kwargs",
        "numdiff_options",
        "log_options",
        "error_penalty",
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

    arguments.update({"logging": _process_logging(logging, n_optimizations)})

    # Convert arguments from dictionary of lists to lists of dictionaries.
    arguments = [
        {key: value[i] for key, value in arguments.items()}
        for i in range(n_optimizations)
    ]

    return arguments


def _process_logging(logging, n_optimizations):
    """Process paths to databases.

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
    >>> _process_logging(False, 1)
    [False]
    >>> _process_logging(False, 2)
    [False, False]
    >>> _process_logging([False], 2)
    [False, False]

    """
    if not isinstance(logging, (tuple, list)):
        if not logging:
            logging = [False] * n_optimizations
        elif not isinstance(logging, (str, Path)):
            raise ValueError("logging has to be a pathlib.Path, string or False")

        elif n_optimizations >= 2:
            path = Path(logging).absolute()
            logging = [
                path.parent / (path.stem + f"_{i}.db") for i in range(n_optimizations)
            ]
        else:
            logging = [Path(logging).absolute()]

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

    # replace everything that evaluates to False by an actual False
    logging = [path if path else False for path in logging]

    return logging
