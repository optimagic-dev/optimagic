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
from estimagic.optimization.utilities import propose_algorithms


def transform_problem(
    criterion,
    params,
    algorithm,
    criterion_kwargs,
    constraints,
    general_options,
    algo_options,
    gradient,
    gradient_options,
    logging,
    log_options,
    dashboard,
    dash_options,
):
    """Transform the user supplied problem.

    The transformed optimization problem is converted from the original problem
    which consists of the user supplied criterion, params DataFrame, criterion_kwargs,
    constraints and gradient (if supplied).
    In addition, the transformed optimization problem provides sophisticated logging
    tools if activated by the user.

    The transformed problem can be solved by almost any optimizer package:
        1. The only constraints are bounds on the parameters.
        2. The internal_criterion function takes an one dimensional np.array as input.
        3. The internal criterion function returns a scalar value
            (except for the case of the tao_pounders algorithm).

    Note that because of the reparametrizations done by estimagic to implement
    constraints on behalf of the user the internal params cannot be interpreted without
    reparametrizing it to the full params DataFrame.

    Args:
        criterion (callable or list of callables): Python function that takes a pandas
            DataFrame with parameters as the first argument. Supported outputs are:
                - scalar floating point
                - np.ndarray: contributions for the tao Pounders algorithm.
                - tuple of a scalar floating point and a pd.DataFrame:
                    In this case the first output is the criterion value.
                    The second output are the comparison_plot_data.
                    See :ref:`comparison_plot`.
                    .. warning::
                        This feature is not implemented in the dashboard yet.
        params (pd.DataFrame or list of pd.DataFrames): See :ref:`params`.
        algorithm (str or list of strings): Name of the optimization algorithm.
            See :ref:`list_of_algorithms`.
        criterion_kwargs (dict or list of dict): Additional criterion keyword arguments.
        constraints (list or list of lists): List with constraint dictionaries.
            See :ref:`constraints` for details.
        general_options (dict): Additional configurations for the optimization.
            Keys can include:
                - keep_dashboard_alive (bool): if True and dashboard is True the process
                    in which the dashboard is run is not terminated when maximize or
                    minimize finish.
        algo_options (dict or list of dicts): Algorithm specific configurations.
        gradient_options (dict): Options for the gradient function.
        logging (str or pathlib.Path or list thereof): Path to an sqlite3 file which
            typically has the file extension ``.db``. If the file does not exist,
            it will be created. See :ref:`logging` for details.
        log_options (dict or list of dict): Keyword arguments to influence the logging.
            See :ref:`logging` for details.
        dashboard (bool): Whether to create and show a dashboard, default is False.
            See :ref:`dashboard` for details.
        dash_options (dict or list of dict, optional): Options passed to the dashboard.
            Supported keys are:
                - port (int): port where to display the dashboard
                - no_browser (bool): whether to display the dashboard in a browser
                - rollover (int): how many iterations to keep in the monitoring plots

    Returns:
        optim_kwargs (dict): Dictionary collecting all arguments that are going to be
            passed to _internal_minimize.
        database_path (str or pathlib.Path or None): Path to the database.
        result_kwargs (dict): Arguments needed to reparametrize back from the internal
            paramater array to the params DataFrame of the user supplied problem.
            In addition it contains whether the dashboard process should be kept alive
            after the optimization(s) terminate(s).

    """
    optim_kwargs, params, dash_options, database_path = _pre_process_arguments(
        params=params,
        algorithm=algorithm,
        algo_options=algo_options,
        logging=logging,
        dashboard=dashboard,
        dash_options=dash_options,
    )

    # harmonize criterion interface
    is_maximization = general_options.pop("_maximization", False)
    criterion = expand_criterion_output(criterion)
    criterion = negative_criterion(criterion) if is_maximization else criterion

    # first criterion evaluation for the database and the pounders algorithm
    fitness_eval, comparison_plot_data = _evaluate_criterion(
        criterion=criterion, params=params, criterion_kwargs=criterion_kwargs
    )
    general_options = general_options.copy()
    general_options["start_criterion_value"] = fitness_eval

    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

        # transform the user supplied inputs into the internal inputs.
        constraints, params = process_constraints(constraints, params)
        internal_params = reparametrize_to_internal(params, constraints)
        bounds = _get_internal_bounds(params)

    # setup the database to pass it to the internal functions for logging
    if logging:
        database = prepare_database(
            path=logging,
            params=params,
            comparison_plot_data=comparison_plot_data,
            dash_options=dash_options,
            constraints=constraints,
            **log_options,
        )
    else:
        database = False

    # transform the user supplied criterion and gradient function into their
    # internal counterparts that use internal inputs.

    # this must be passed to _create_internal_criterion because the internal
    # gradient creates its own internal criterion function whose calls are
    # logged differently by the database.
    logging_decorator = functools.partial(
        log_evaluation,
        database=database,
        tables=["params_history", "criterion_history", "comparison_plot", "timestamps"],
    )

    internal_criterion = _create_internal_criterion(
        criterion=criterion,
        params=params,
        constraints=constraints,
        criterion_kwargs=criterion_kwargs,
        logging_decorator=logging_decorator,
        general_options=general_options,
        database=database,
    )

    internal_gradient = _create_internal_gradient(
        gradient=gradient,
        gradient_options=gradient_options,
        criterion=criterion,
        params=params,
        constraints=constraints,
        criterion_kwargs=criterion_kwargs,
        general_options=general_options,
        database=database,
    )

    internal_kwargs = {
        "internal_criterion": internal_criterion,
        "internal_params": internal_params,
        "bounds": bounds,
        "internal_gradient": internal_gradient,
        "database": database,
        "general_options": general_options,
    }
    optim_kwargs.update(internal_kwargs)

    result_kwargs = {
        "params": params,
        "constraints": constraints,
        "keep_dashboard_alive": general_options.pop("keep_dashboard_alive", False),
    }
    return optim_kwargs, database_path, result_kwargs


def _pre_process_arguments(
    params, algorithm, algo_options, logging, dashboard, dash_options
):
    """Process user supplied arguments without affecting the optimization problem.

    Args:
        params (pd.DataFrame or list of pd.DataFrames): See :ref:`params`.
        algorithm (str or list of strings): Identifier of the optimization algorithm.
            See :ref:`list_of_algorithms` for supported values.
        algo_options (dict or list of dicts):
            algorithm specific configurations for the optimization
        dashboard (bool): Whether to create and show a dashboard, default is False.
            See :ref:`dashboard` for details.
        dash_options (dict or list of dict, optional): Options passed to the dashboard.
            Supported keys are:
                - port (int): port where to display the dashboard
                - no_browser (bool): whether to display the dashboard in a browser
                - rollover (int): how many iterations to keep in the monitoring plots

    Returns:
        optim_kwargs (dict): dictionary collecting the arguments that are going to be
            passed to _internal_minimize
        params (pd.DataFrame): The expanded params DataFrame with all needed columns.
            See :ref:`params`.
        database_path (str or pathlib.Path or None): path to the database.

    """
    standard_dash_options = {"no_browser": False, "port": None, "rollover": 500}
    # important for dash_options to be last for standards to be overwritten
    dash_options = {**standard_dash_options, **dash_options}

    origin, algo_name = _process_algorithm(algorithm)
    optim_kwargs = {
        "origin": origin,
        "algo_name": algo_name,
        "algo_options": algo_options,
    }

    params = _set_params_defaults_if_missing(params)
    _check_params(params)

    database_path = logging if dashboard else None

    return optim_kwargs, params, dash_options, database_path


def _process_algorithm(algorithm):
    """Identify the algorithm from the user-supplied string.

    Args:
        algorithm (str): Package and name of the algorithm. It should be of the format
            {pkg}_{name}.

    Returns:
        origin (str): Name of the package.
        algo_name (str): Name of the algorithm.

    """
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


def _set_params_defaults_if_missing(params):
    """Set defaults and run checks on the user-supplied params.

    Args:
        params (pd.DataFrame): See :ref:`params`.

    Returns:
        params (pd.DataFrame): With defaults expanded params DataFrame.

    """
    params = params.copy()
    if "lower" not in params.columns:
        params["lower"] = -np.inf
    else:
        params["lower"].fillna(-np.inf, inplace=True)

    if "upper" not in params.columns:
        params["upper"] = np.inf
    else:
        params["upper"].fillna(np.inf, inplace=True)

    if "group" not in params.columns:
        params["group"] = "All Parameters"

    if "name" not in params.columns:
        names = [_index_element_to_string(tup) for tup in params.index]
        params["name"] = names
    return params


def _index_element_to_string(element, separator="_"):
    if isinstance(element, (tuple, list)):
        as_strings = [str(entry).replace("-", "_") for entry in element]
        res_string = separator.join(as_strings)
    else:
        res_string = str(element)
    return res_string


def _check_params(params):
    """Check params has a unique index and contains no columns to be created internally.

    Args:
        params (pd.DataFrame or list of pd.DataFrames): See :ref:`params`.

    Raises:
        AssertionError: The index contains duplicates.
        ValueError: The DataFrame contains internal columns.

    """
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


def _evaluate_criterion(criterion, params, criterion_kwargs):
    """Evaluate the criterion function for the first time.

    The comparison_plot_data output is needed to initialize the database.
    The criterion value is stored in the general options for the tao pounders algorithm.

    Args:
        criterion (callable): Python function that takes a pandas DataFrame with
            parameters as the first argument and returns a value or array to be
            minimized and data for the comparison plot.
        params (pd.DataFrame): See :ref:`params`.
        criterion_kwargs (dict): Additional keyword arguments for criterion.

    Returns:
        fitness_eval (float): The scalar criterion value.
        comparison_plot_data (np.array or pd.DataFrame): Data for the comparison_plot.

    """
    criterion_out, comparison_plot_data = criterion(params, **criterion_kwargs)
    if np.any(np.isnan(criterion_out)):
        raise ValueError(
            "The criterion function evaluated at the start parameters returns NaNs."
        )
    elif np.isscalar(criterion_out):
        fitness_eval = criterion_out
    else:
        fitness_eval = np.mean(np.square(criterion_out))
    return fitness_eval, comparison_plot_data


def _create_internal_criterion(
    criterion,
    params,
    constraints,
    criterion_kwargs,
    logging_decorator,
    general_options,
    database,
):
    """Create the internal criterion function.

    The internal criterion function takes a numpy array of free parameters
    (called internal params) and returns the criterion value of the user supplied
    problem. The reparametrization of the free parameters to the full paramteres
    insures that the user supplied constraints are fulfilled for all values the
    optimizer algorithm could try respecting the bounds created elsewhere.

    If logging is activated, the returned function also logs its calls.

    Args:
        criterion (callable):
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

    Returns:
        internal_criterion (function):
            function that takes an internal_params np.array as only argument.
            It calls the original criterion function after the necessary
            reparametrizations.

    """

    @handle_exceptions(database, params, constraints, params, general_options)
    @numpy_interface(params, constraints)
    @logging_decorator
    def internal_criterion(p):
        """Criterion of the transformed problem."""
        criterion_out, comparison_plot_data = criterion(p, **criterion_kwargs)
        return criterion_out, comparison_plot_data

    return internal_criterion


def _create_internal_gradient(
    gradient,
    gradient_options,
    criterion,
    params,
    constraints,
    criterion_kwargs,
    general_options,
    database,
):
    """Create the internal gradient function.

    Args:
        gradient (None): Gradients are currently not allowed to be passed to minimize.
        gradient_options (dict): Options for the gradient function.
        criterion (callable or list of callables): Python function that takes a pandas
            DataFrame with parameters as the first argument. Supported outputs are:
                - scalar floating point
                - np.ndarray: contributions for the tao Pounders algorithm.
                - tuple of a scalar floating point and a pd.DataFrame:
                    In this case the first output is the criterion value.
                    The second output are the comparison_plot_data.
                    See :ref:`comparison_plot`.
                    .. warning::
                        This feature is not implemented in the dashboard yet.
        params (pd.DataFrame or list of pd.DataFrames): See :ref:`params`.
        constraints (list or list of lists): List with constraint dictionaries.
            See :ref:`constraints` for details.
        criterion_kwargs (dict or list of dict): Additional criterion keyword arguments.
        general_options (dict): Additional configurations for the optimization.
            Keys can include:
                - keep_dashboard_alive (bool): if True and dashboard is True the process
                    in which the dashboard is run is not terminated when maximize or
                    minimize finish.
        database (sqlalchemy.MetaData)

    Returns:
        internal_gradient (function)

    """
    n_internal_params = params["_internal_free"].sum()
    gradient_options = {} if gradient_options is None else gradient_options
    names = params.query("_internal_free")["name"].tolist()

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
        logging_decorator = functools.partial(
            log_gradient_status,
            database=database,
            n_gradient_evaluations=n_gradient_evaluations,
        )

        internal_criterion = _create_internal_criterion(
            criterion=criterion,
            params=params,
            constraints=constraints,
            criterion_kwargs=criterion_kwargs,
            logging_decorator=logging_decorator,
            general_options=general_options,
            database=database,
        )
        bounds = _get_internal_bounds(params)

        @log_gradient(database, names)
        def internal_gradient(x):
            return gradient(internal_criterion, x, bounds=bounds, **gradient_options)

    else:
        if constraints not in [[], None]:
            raise NotImplementedError(
                "A user provided gradient is not compatible with constraints."
            )

        @log_gradient(database, names)
        @numpy_interface(params, constraints)
        def internal_gradient(p):
            return gradient(p)

    return internal_gradient


def _get_internal_bounds(params):
    """Extract the internal bounds from params.

    Args:
        params (pd.DataFrame): See :ref:`params`.

    Returns:
        bounds (tuple): Bounds of the free parameters. The tuple has two entries.
            The first are the lower bounds as numpy array.
            The second are the upper bounds as numpy array.

    """
    bounds = tuple(
        params.query("_internal_free")[["_internal_lower", "_internal_upper"]]
        .to_numpy()
        .T
    )
    return bounds
