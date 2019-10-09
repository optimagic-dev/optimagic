from collections import Callable
import pandas as pd


def _check_type_nonlist_argument(candidate, scalar_type, name):
    """ Calculate number of inputs and check input types for argument
    that is expected as scalar.

    Args:
        candidate:
            Argument

        scalar_type:
            Expected type of a scalar argument

        name (str):
            Name of argument used in error message
    """
    msg = f"{name} has to be a {scalar_type}"
    if not isinstance(candidate, scalar_type):
        raise ValueError(msg)


def _get_n_opt_and_check_type_list_argument(
    candidate, scalar_type, argument_required, name
):
    """ Calculate number of inputs and check input types for argument
    that is expected as scalar or list/tuple.

    Args:
        candidate:
            Argument

        scalar_type:
            Expected type of a scalar argument

        argument_required (bool):
            Whether an empty argument is allowed

        name (str):
            Name of argument used in error message
    """
    msg = f"{name} has to be a {scalar_type} or list of {scalar_type}s."

    # list/tuple
    if isinstance(candidate, (list, tuple)):

        # Empty argument
        if candidate in ([], ()):
            num_opt = 1
            if argument_required:
                raise ValueError(f"{name} needs to be specified")

        # Non-empty list/tuple
        else:
            num_opt = len(candidate)

            # Assert that all values are of the correct type
            for c in candidate:
                if not isinstance(c, scalar_type):
                    raise ValueError(msg)

    # Scalar
    else:
        num_opt = 1

        # Assert that scalar is of the correct type
        if not isinstance(candidate, scalar_type):
            raise ValueError(msg)

    return num_opt


def _get_n_opt_and_check_type_nested_list_argument(candidate, argument_required, name):
    """ Calculate number of inputs and check input types for argument
    that is expected as list/tuple or nested list/tuple.

    ToDo: Check scalar types, as well?

    Args:
        candidate
            Argument

        argument_required (bool):
            Whether an empty argument is allowed

        name (str):
            Name of argument used in error message

    """
    msg = f"{name} has to be a list/tuple or a nested list/tuple."

    # No list or tuple
    if not isinstance(candidate, (list, tuple)):
        raise ValueError(msg)

    # Empty list/tuple
    elif candidate in ([], ()):
        num_opt = 1
        if argument_required:
            raise ValueError(f"{name} needs to be specified")

    # Nested list/tuple
    elif isinstance(candidate[0], (list, tuple)):
        num_opt = len(candidate)
        for c in candidate:
            if not isinstance(c, (list, tuple)):
                raise ValueError(msg)

    # Non-empty 1-dimensional list/tuple
    else:
        num_opt = 1
    return num_opt


def broadcast_argument(argument, len_arg, n_opts_total, name):
    """Broadcast argument if of length 1. Otherwise, make sure that it is of
    the same length as all other arguments.

    Args:
        argument:
            Argument

        len_arg (int):
            Length of argument

        n_opts_total (int):
            Number of optimizatiions that should be run

        name (str):
            Name of argument used in error message (not used so far)
    """
    msg = (
        f"All arguments entered as list/tuple must be of the same length."
        + f"The length of {name} is below the length of another argument."
    )
    # ToDo: deep copy!

    # Only one optimization
    if n_opts_total == 1:
        if isinstance(argument, (list, tuple)) and len(argument) == 1:
            res = argument[0]
        else:
            res = argument

    # Argument is broadcasted
    elif len_arg == 1:
        if isinstance(argument, (list, tuple)) and len(argument) == 1:
            res = argument * n_opts_total
        else:
            res = [argument] * n_opts_total

    # Argument was entered as list/tuple
    elif len_arg == n_opts_total:
        res = argument

    # Argument was entered as too short list/tuple
    else:
        raise ValueError(msg)
    # print(res)
    return res


def process_optimization_arguments(
    criterion,
    params,
    algorithm,
    criterion_args=None,
    criterion_kwargs=None,
    constraints=None,
    general_options=None,
    algo_options=None,
    dashboard=False,
    db_options=None,
):
    """Process and validate arguments for minimize or maximize.

    Args:
        criterion (function or list of functions):
            Python function that takes a pandas Series with parameters as the first
            argument and returns a scalar floating point value.

        params (pd.DataFrame or list of pd.DataFrames):
            See :ref:`params`.

        algorithm (str or list of strings):
            specifies the optimization algorithm. See :ref:`list_of_algorithms`.

        criterion_args (list)::
            additional positional arguments for criterion

        criterion_kwargs (dict or list of dicts):
            additional keyword arguments for criterion

        constraints (list or list of lists):
            list with constraint dictionaries. See for details.

        general_options (dict):
            additional configurations for the optimization

        algo_options (dict or list of dicts):
            algorithm specific configurations for the optimization

        dashboard (bool):
            whether to create and show a dashboard

        db_options (dict):
            dictionary with kwargs to be supplied to the run_server function.
    """

    # set default arguments
    criterion_args = [] if criterion_args is None else criterion_args
    criterion_kwargs = {} if criterion_kwargs is None else criterion_kwargs
    constraints = [] if constraints is None else constraints
    algo_options = {} if algo_options is None else algo_options
    db_options = {} if db_options is None else db_options
    general_options = {} if general_options is None else general_options

    # Determine number of optimizations and check types

    # Specify name and expected type for all arguments.
    # Three groups of arguments are relevant: (i) expected as scalar,
    # (ii) expected as scalar or list/tuple, (iii) expected as list/tuple or nested list/tuple

    args_non_list = [
        {"candidate": general_options, "scalar_type": dict, "name": "general_options"},
        {"candidate": dashboard, "scalar_type": bool, "name": "dashboard"},
        {"candidate": db_options, "scalar_type": dict, "name": "db_options"},
        {"candidate": criterion_args, "scalar_type": list, "name": "criterion_args"},
    ]
    args_list = [
        {
            "candidate": criterion,
            "scalar_type": Callable,
            "argument_required": True,
            "name": "criterion",
        },
        {
            "candidate": params,
            "scalar_type": pd.DataFrame,
            "argument_required": True,
            "name": "params",
        },
        {
            "candidate": algorithm,
            "scalar_type": str,
            "argument_required": True,
            "name": "algorithm",
        },
        {
            "candidate": criterion_kwargs,
            "scalar_type": dict,
            "argument_required": False,
            "name": "criterion_kwargs",
        },
        {
            "candidate": algo_options,
            "scalar_type": dict,
            "argument_required": False,
            "name": "algo_options",
        },
    ]
    args_nested_list = [
        {"candidate": constraints, "argument_required": False, "name": "constraints"}
    ]
    # ToDo: criterion_args?

    # Check type of all inputs
    for arg in args_non_list:
        _check_type_nonlist_argument(**arg)

    n_opts_args_list = [
        _get_n_opt_and_check_type_list_argument(**arg) for arg in args_list
    ]
    n_opts_args_nested_list = [
        _get_n_opt_and_check_type_nested_list_argument(**arg)
        for arg in args_nested_list
    ]

    # Calc number of optimizations that should be run
    n_opts = max(n_opts_args_list + n_opts_args_nested_list)

    # Broadcast inputs
    criterion, params, algorithm, algo_options, criterion_kwargs, constraints = [
        broadcast_argument(
            argument=arg["candidate"],
            len_arg=len_arg,
            n_opts_total=n_opts,
            name=arg["name"],
        )
        for arg, len_arg in zip(
            args_list + args_nested_list, n_opts_args_list + n_opts_args_nested_list
        )
    ]

    general_options, dashboard, db_options, criterion_args = [
        broadcast_argument(
            argument=arg["candidate"], len_arg=1, n_opts_total=n_opts, name=arg["name"]
        )
        for arg in args_non_list
    ]

    # Put arguments together
    processed_arguments = {
        "criterion": criterion,
        "params": params,
        "algorithm": algorithm,
        "criterion_args": criterion_args,
        "criterion_kwargs": criterion_kwargs,
        "constraints": constraints,
        "general_options": general_options,
        "algo_options": algo_options,
        "dashboard": dashboard,
        "db_options": db_options,
    }
    # # Use pandas to convert dict of lists to list of dicts
    # processed_arguments = pd.DataFrame(processed_arguments).to_dict("list")

    # if n_opts == 1:
    #     processed_arguments = processed_arguments
    return processed_arguments
