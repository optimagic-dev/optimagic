import functools
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize._numdiff import approx_derivative

from estimagic.decorators import expand_criterion_output
from estimagic.decorators import handle_exceptions
from estimagic.decorators import log_evaluation
from estimagic.decorators import log_gradient
from estimagic.decorators import log_gradient_status
from estimagic.decorators import negative_criterion
from estimagic.decorators import numpy_interface
from estimagic.logging.create_database import prepare_database
from estimagic.optimization.process_constraints import process_constraints
from estimagic.optimization.reparametrize import reparametrize_to_internal
from estimagic.optimization.utilities import index_element_to_string
from estimagic.optimization.utilities import propose_algorithms


def process_arguments(broadcasted_arguments):
    optim_arguments = []
    results_arguments = []
    db_paths = []
    for arguments in broadcasted_arguments:
        with warnings.catch_warnings():
            warnings.simplefilter(
                action="ignore", category=pd.errors.PerformanceWarning
            )
            record_path = arguments.pop("dashboard")
            optim, res = _process_args_of_one_optimization(**arguments)
            if record_path:
                db_paths.append(arguments["logging"])
            optim_arguments.append(optim)
            results_arguments.append(res)

    return optim_arguments, db_paths, results_arguments


def _process_args_of_one_optimization(
    criterion,
    params,
    algorithm,
    criterion_kwargs,
    constraints,
    general_options,
    algo_options,
    gradient_options,
    logging,
    log_options,
    db_options,
    gradient,
):
    general_options = general_options.copy()
    params = _pre_process_params(params)

    # Apply decorator two handle criterion functions with one or two returns.
    criterion = expand_criterion_output(criterion)

    is_maximization = general_options.pop("_maximization", False)
    criterion = negative_criterion(criterion) if is_maximization else criterion

    # first criterion evaluation
    fitness_factor = -1 if is_maximization else 1
    criterion_out, comparison_plot_data = criterion(params, **criterion_kwargs)
    if np.isscalar(criterion_out):
        fitness_eval = fitness_factor * criterion_out
    else:
        fitness_eval = fitness_factor * np.mean(np.square(criterion_out))
    if np.any(np.isnan(fitness_eval)):
        raise ValueError(
            "The criterion function evaluated at the start parameters returns NaNs."
        )
    general_options["start_criterion_value"] = fitness_eval

    constraints, params = process_constraints(constraints, params)
    internal_params = reparametrize_to_internal(params, constraints)

    if logging:
        database = prepare_database(
            path=logging,
            params=params,
            comparison_plot_data=comparison_plot_data,
            db_options=db_options,
            constraints=constraints,
        )
    else:
        database = False

    logging_decorator = functools.partial(
        log_evaluation,
        database=database,
        tables=["params_history", "criterion_history", "comparison_plot"],
    )

    internal_criterion = create_internal_criterion(
        criterion=criterion,
        params=params,
        constraints=constraints,
        criterion_kwargs=criterion_kwargs,
        logging_decorator=logging_decorator,
        general_options=general_options,
        database=database,
        fitness_factor=fitness_factor,
    )

    internal_gradient = create_internal_gradient(
        gradient=gradient,
        gradient_options=gradient_options,
        criterion=criterion,
        params=params,
        internal_params=internal_params,
        constraints=constraints,
        criterion_kwargs=criterion_kwargs,
        general_options=general_options,
        database=database,
        fitness_factor=fitness_factor,
        algorithm=algorithm,
    )

    origin, algo_name = _process_algorithm(algorithm)
    bounds = _internal_bounds_from_params(params)

    optim_kwargs = {
        "internal_criterion": internal_criterion,
        "internal_params": internal_params,
        "bounds": bounds,
        "origin": origin,
        "algo_name": algo_name,
        "algo_options": algo_options,
        "internal_gradient": internal_gradient,
        "database": database,
        "general_options": general_options,
    }

    result_kwargs = {
        "params": params,
        "constraints": constraints,
    }

    return optim_kwargs, result_kwargs


def _pre_process_params(params):
    """Set defaults and run checks on the user-supplied params."""
    params = params.copy()
    if "lower" not in params.columns:
        params["lower"] = -np.inf
    else:
        params["lower"].fillna(-np.inf)

    if "upper" not in params.columns:
        params["upper"] = np.inf
    else:
        params["upper"].fillna(np.inf)

    if "group" not in params.columns:
        params["group"] = "All Parameters"

    if "name" not in params.columns:
        names = [index_element_to_string(tup) for tup in params.index]
        params["name"] = names

    assert (
        not params.index.duplicated().any()
    ), "No duplicates allowed in the index of params."

    invalid_names = [
        "_fixed",
        "_fixed_value",
        "_is_fixed_to_value",
        "_is_fixed_to_other",
    ]
    invalid_present_columns = []
    for col in params.columns:
        if col in invalid_names or col.startswith("_internal"):
            invalid_present_columns.append(col)

    if len(invalid_present_columns) > 0:
        msg = (
            "Column names starting with '_internal' and as well as any other of the "
            f"following columns are not allowed in params:\n{invalid_names}."
            f"This is violated for:\n{invalid_present_columns}."
        )
        raise ValueError(msg)
    return params


def create_internal_criterion(
    criterion,
    params,
    constraints,
    criterion_kwargs,
    logging_decorator,
    general_options,
    database,
    fitness_factor,
):
    """Create the internal criterion function.

    Args:
        criterion (function):
            Python function that takes a pandas DataFrame with parameters as the first
            argument and returns a scalar floating point value.

        params (pd.DataFrame):
            See :ref:`params`.

        constraints (list):
            list with constraint dictionaries. See for details.

        criterion_kwargs (dict):
            additional keyword arguments for criterion

        logging_decorator (callable):
            Decorator used for logging information. Either log parameters and fitness
            values during the optimization or log the gradient status.

        general_options (dict):
            additional configurations for the optimization

        database (sqlalchemy.MetaData). The engine that connects to the
            database can be accessed via ``database.bind``.

        fitness_factor (float):
            multiplicative factor for the fitness displayed in the dashboard.
            Set to -1 for maximizations to plot the fitness that is being maximized.

    Returns:
        internal_criterion (function):
            function that takes an internal_params DataFrame as only argument.
            It calls the original criterion function after the necessary
            reparametrizations.

    """

    @handle_exceptions(database, params, constraints, params, general_options)
    @numpy_interface(params, constraints)
    @logging_decorator
    def internal_criterion(p):
        criterion_out, comparison_plot_data = criterion(p, **criterion_kwargs)
        return criterion_out, comparison_plot_data

    return internal_criterion


def create_internal_gradient(
    gradient,
    gradient_options,
    criterion,
    params,
    internal_params,
    constraints,
    criterion_kwargs,
    general_options,
    database,
    fitness_factor,
    algorithm,
):
    n_internal_params = params["_internal_free"].sum()
    gradient_options = {} if gradient_options is None else gradient_options

    if gradient is None:
        gradient = approx_derivative
        default_options = {
            "method": "2-point",
            "rel_step": None,
            "f0": None,
            "sparsity": None,
            "as_linear_operator": False,
        }
        gradient_options = {**default_options, **gradient_options}

        if gradient_options["method"] == "2-point":
            n_gradient_evaluations = 2 * n_internal_params
        elif gradient_options["method"] == "3-point":
            n_gradient_evaluations = 3 * n_internal_params
        else:
            raise ValueError(
                f"Gradient method '{gradient_options['method']} not supported."
            )

    else:
        n_gradient_evaluations = gradient_options.pop("n_gradient_evaluations", None)

    logging_decorator = functools.partial(
        log_gradient_status,
        database=database,
        n_gradient_evaluations=n_gradient_evaluations,
    )

    internal_criterion = create_internal_criterion(
        criterion=criterion,
        params=params,
        constraints=constraints,
        criterion_kwargs=criterion_kwargs,
        logging_decorator=logging_decorator,
        general_options=general_options,
        database=database,
        fitness_factor=fitness_factor,
    )
    bounds = _internal_bounds_from_params(params)
    names = params.query("_internal_free")["name"].tolist()

    @log_gradient(database, names)
    def internal_gradient(x):
        return gradient(internal_criterion, x, bounds=bounds, **gradient_options)

    return internal_gradient


def _internal_bounds_from_params(params):
    bounds = tuple(
        params.query("_internal_free")[["_internal_lower", "_internal_upper"]]
        .to_numpy()
        .T
    )
    return bounds


def _process_algorithm(algorithm):
    current_dir_path = Path(__file__).resolve().parent
    with open(current_dir_path / "algo_dict.json") as j:
        algos = json.load(j)
    origin, algo_name = algorithm.split("_", 1)

    try:
        assert algo_name in algos[origin], "Invalid algorithm requested: {}".format(
            algorithm
        )
    except (AssertionError, KeyError):
        proposals = propose_algorithms(algorithm, algos)
        raise NotImplementedError(
            f"{algorithm} is not a valid choice. Did you mean one of {proposals}?"
        )

    return origin, algo_name
