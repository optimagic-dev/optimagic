import copy
from collections import Callable

import pandas as pd


def process_optimization_arguments(
    criterion,
    params,
    algorithm,
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
            Python function that takes a pandas DataFrame with parameters as the first
            argument and returns a scalar floating point value.

        params (pd.DataFrame or list of pd.DataFrames):
            See :ref:`params`.

        algorithm (str or list of strings):
            specifies the optimization algorithm. See :ref:`list_of_algorithms`.

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


    Returns:
        List: List of dictionaries. One dict for each optimization.

    """

    # set default arguments
    criterion_kwargs = {} if criterion_kwargs is None else criterion_kwargs
    constraints = [] if constraints is None else constraints
    algo_options = {} if algo_options is None else algo_options
    db_options = {} if db_options is None else db_options
    general_options = {} if general_options is None else general_options
    # Determine number of optimizations and check types

    # Specify name, expected type and expected dimensionality for all arguments.
    # Three groups of expected dimensionality are relevant:
    # (i) expected as scalar,
    # (ii) expected as scalar or list/tuple,
    # (iii) expected as list/tuple or nested list/tuple
    arguments = {
        "general_options": {
            "candidate": general_options,
            "scalar_type": dict,
            "expected_dim": "scalar",
        },
        "dashboard": {
            "candidate": dashboard,
            "scalar_type": bool,
            "expected_dim": "scalar",
        },
        "db_options": {
            "candidate": db_options,
            "scalar_type": dict,
            "expected_dim": "scalar",
        },
        "criterion": {
            "candidate": criterion,
            "scalar_type": Callable,
            "argument_required": True,
            "expected_dim": "scalar_or_list",
        },
        "params": {
            "candidate": params,
            "scalar_type": pd.DataFrame,
            "argument_required": True,
            "expected_dim": "scalar_or_list",
        },
        "algorithm": {
            "candidate": algorithm,
            "scalar_type": str,
            "argument_required": True,
            "expected_dim": "scalar_or_list",
        },
        "algo_options": {
            "candidate": algo_options,
            "scalar_type": dict,
            "argument_required": False,
            "expected_dim": "scalar_or_list",
        },
        "criterion_kwargs": {
            "candidate": criterion_kwargs,
            "scalar_type": dict,
            "argument_required": False,
            "expected_dim": "scalar_or_list",
        },
        "constraints": {
            "candidate": constraints,
            "argument_required": False,
            "expected_dim": "list_or_nested_list",
        },
    }

    # Check type and calc n_opts for each argument
    for arg_name, arg_spec in arguments.items():
        if arg_spec["expected_dim"] == "scalar":
            _check_type_nonlist_argument(
                candidate=arg_spec["candidate"],
                scalar_type=arg_spec["scalar_type"],
                name=arg_name,
            )
            arg_spec["n_opts_entered"] = 1
        elif arg_spec["expected_dim"] == "scalar_or_list":
            arg_spec["n_opts_entered"] = _get_n_opt_and_check_type_list_argument(
                candidate=arg_spec["candidate"],
                scalar_type=arg_spec["scalar_type"],
                argument_required=arg_spec["argument_required"],
                name=arg_name,
            )
        elif arg_spec["expected_dim"] == "list_or_nested_list":
            arg_spec["n_opts_entered"] = _get_n_opt_and_check_type_nested_list_argument(
                candidate=arg_spec["candidate"],
                argument_required=arg_spec["argument_required"],
                name=arg_name,
            )

    # Calc number of optimizations that should be run
    n_opts = max(a["n_opts_entered"] for a in arguments.values())

    # Put arguments together
    processed_arguments = []
    for run in range(n_opts):
        args_one_run = {}
        for arg_name, arg_spec in arguments.items():
            # Entered as scalar
            if arg_spec["n_opts_entered"] == 1:
                is_list = isinstance(arg_spec["candidate"], (list, tuple))
                if is_list and len(arg_spec["candidate"]) == 1:
                    if arg_name == "constraints":
                        args_one_run[arg_name] = copy.deepcopy(arg_spec["candidate"])
                    else:
                        args_one_run[arg_name] = copy.deepcopy(arg_spec["candidate"][0])
                else:
                    args_one_run[arg_name] = copy.deepcopy(arg_spec["candidate"])

            # Entered as list of correct length
            elif arg_spec["n_opts_entered"] == n_opts:
                args_one_run[arg_name] = arg_spec["candidate"][run]

            # Entered as too short list
            else:
                raise ValueError(
                    "All arguments entered as list/tuple must be of the same length. "
                    f"The length of {arg_name} is below the length of another argument."
                )
        processed_arguments.append(args_one_run)

    return processed_arguments


def _check_type_nonlist_argument(candidate, scalar_type, name):
    """Check input types for argument that is expected as scalar.

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
    """Calculate number of inputs and check input types for argument
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
    """Calculate number of inputs and check input types for argument
    that is expected as list/tuple or nested list/tuple.

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
