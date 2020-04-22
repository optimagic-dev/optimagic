"""Functional wrapper around the pygmo, nlopt and scipy libraries."""
import numpy as np
from joblib import delayed
from joblib import Parallel

from estimagic.config import DEFAULT_DATABASE_NAME
from estimagic.dashboard.run_dashboard import run_dashboard_in_separate_process
from estimagic.decorators import negative_gradient
from estimagic.logging.update_database import update_scalar_field
from estimagic.optimization.broadcast_arguments import broadcast_arguments
from estimagic.optimization.check_arguments import check_arguments
from estimagic.optimization.pounders import minimize_pounders_np
from estimagic.optimization.pygmo import minimize_pygmo_np
from estimagic.optimization.reparametrize import reparametrize_from_internal
from estimagic.optimization.scipy import minimize_scipy_np
from estimagic.optimization.transform_problem import transform_problem


def maximize(
    criterion,
    params,
    algorithm,
    criterion_kwargs=None,
    constraints=None,
    general_options=None,
    algo_options=None,
    gradient=None,
    gradient_kwargs=None,
    gradient_options=None,
    logging=DEFAULT_DATABASE_NAME,
    log_options=None,
    dashboard=False,
    dash_options=None,
):
    """Maximize criterion using algorithm subject to constraints and bounds.
    Each argument except for general_options can also be replaced by a list of
    arguments in which case several optimizations are run in parallel. For this, either
    all arguments must be lists of the same length, or some arguments can be provided
    as single arguments in which case they are automatically broadcasted.
    Args:
        criterion (callable or list of callables):
            Python callable that takes a pandas DataFrame with parameters as the first
            argument. Supported outputs are:
                - scalar floating point
                - np.ndarray: contributions for the tao Pounders algorithm.
                - tuple of a scalar floating point and a pd.DataFrame:
                    In this case the first output is the criterion value.
                    The second output are the comparison_plot_data.
        params (pd.DataFrame or list of pd.DataFrames):
            See :ref:`params`.
        algorithm (str or list of strings): Specifies the optimization algorithm.
            See :ref:`list_of_algorithms`.
        criterion_kwargs (dict or list of dicts): Additional keyword arguments for
            criterion.
        constraints (list or list of lists): List with constraint dictionaries.
            See :ref:`constraints`.
        general_options (dict): Additional configurations for the optimization.
            Keys can include:
                - keep_dashboard_alive (bool): Do not terminate the dashboard process
                    after the optimization(s) finish(es).
        algo_options (dict or list of dicts): Algorithm specific configurations for the
            optimization.
        gradient (callable): Gradient of the criterion function. Takes params as first
            argument and returns the gradient as numpy array or pandas Series.
        gradient_kwargs (dict): Additional keyword arguments for the gradient.
        gradient_options (dict): Options for the gradient function.
        logging (str or pathlib.Path or list): Path(s) to (an) sqlite3 file(s) which
            typically has the file extension ``.db``. If the file does not exist,
            it will be created. See :ref:`logging` for details.
        log_options (dict or list of dict): Keyword arguments to influence the logging.
            See :ref:`logging` for details.
        dashboard (bool): Whether to create and show a dashboard, default is False.
            See :ref:`dashboard` for details.
        dash_options (dict or list of dict, optional): Options passed to the dashboard.
            Supported keys are:
                - port (int): port where to display the dashboard.
                - no_browser (bool): whether to display the dashboard in a browser.
                - rollover (int): how many iterations to keep in the convergence plots.
    Returns:
        results (tuple or list of tuples): Each tuple consists of the harmonized result
        info dictionary and the params DataFrame with the minimizing parameter values
        of the untransformed problem as specified of the user.
    """
    # Set a flag for a maximization problem.
    general_options = {} if general_options is None else general_options
    general_options["_maximization"] = True

    if isinstance(gradient, list):
        gradient = [negative_gradient(grad) for grad in gradient]
    else:
        gradient = negative_gradient(gradient)

    results = minimize(
        criterion=criterion,
        params=params,
        algorithm=algorithm,
        criterion_kwargs=criterion_kwargs,
        constraints=constraints,
        general_options=general_options,
        algo_options=algo_options,
        gradient=gradient,
        gradient_kwargs=gradient_kwargs,
        gradient_options=gradient_options,
        logging=logging,
        log_options=log_options,
        dashboard=dashboard,
        dash_options=dash_options,
    )

    # Change the fitness value. ``results`` is either a tuple of results and params or a
    # list of tuples.
    if not isinstance(results, list):
        results = [results]

    results = [_undo_sign_switch(res) for res in results]

    results = results[0] if len(results) == 1 else results

    return results


def _undo_sign_switch(res):
    info, params = res
    info = info.copy()
    info["fitness"] = -info["fitness"]
    if "jacobian" in info and info["jacobian"] is not None:
        info["jacobian"] = (-np.array(info["jacobian"])).tolist()
    if "hessian" in info and info["hessian"] is not None:
        info["hessian"] = (-np.array(info["hessian"])).tolist()
    return info, params


def minimize(
    criterion,
    params,
    algorithm,
    criterion_kwargs=None,
    constraints=None,
    general_options=None,
    algo_options=None,
    gradient=None,
    gradient_kwargs=None,
    gradient_options=None,
    logging=DEFAULT_DATABASE_NAME,
    log_options=None,
    dashboard=False,
    dash_options=None,
):
    """Minimize *criterion* using *algorithm* subject to *constraints* and bounds.
    Each argument except for ``general_options`` can also be replaced by a list of
    arguments in which case several optimizations are run in parallel. For this, either
    all arguments must be lists of the same length, or some arguments can be provided
    as single arguments in which case they are automatically broadcasted.
    Args:
        criterion (callable or list of callables):
            Python callable that takes a pandas DataFrame with parameters as the first
            argument. Supported outputs are:
                - scalar floating point
                - np.ndarray: contributions for the tao Pounders algorithm.
                - tuple of a scalar floating point and a pd.DataFrame:
                    In this case the first output is the criterion value.
                    The second output are the comparison_plot_data.
        params (pd.DataFrame or list of pd.DataFrames):
            See :ref:`params`.
        algorithm (str or list of strings): Specifies the optimization algorithm.
            See :ref:`list_of_algorithms`.
        criterion_kwargs (dict or list of dicts): Additional keyword arguments for
            criterion.
        constraints (list or list of lists): List with constraint dictionaries.
            See :ref:`constraints`.
        general_options (dict): Additional configurations for the optimization.
            Keys can include:
                - keep_dashboard_alive (bool): Do not terminate the dashboard process
                    after the optimization(s) finish(es).
        algo_options (dict or list of dicts): Algorithm specific configurations for the
            optimization.
        gradient (callable): Gradient of the criterion function. Takes params as first
            argument and returns the gradient as numpy array or pandas Series.
        gradient_kwargs (dict): Additional keyword arguments for the gradient.
        gradient_options (dict): Options for the gradient function.
        logging (str or pathlib.Path or list): Path(s) to (an) sqlite3 file(s) which
            typically has the file extension ``.db``. If the file does not exist,
            it will be created. See :ref:`logging` for details.
        log_options (dict or list of dict): Keyword arguments to influence the logging.
            See :ref:`logging` for details.
        dashboard (bool): Whether to create and show a dashboard, default is False.
            See :ref:`dashboard` for details.
        dash_options (dict or list of dict, optional): Options passed to the dashboard.
            Supported keys are:
                - port (int): port where to display the dashboard.
                - no_browser (bool): whether to display the dashboard in a browser.
                - rollover (int): how many iterations to keep in the convergence plots.
    Returns:
        results (tuple or list of tuples): Each tuple consists of the harmonized result
        info dictionary and the params DataFrame with the minimizing parameter values
        of the untransformed problem as specified of the user.
    """
    arguments = broadcast_arguments(
        criterion=criterion,
        params=params,
        algorithm=algorithm,
        criterion_kwargs=criterion_kwargs,
        constraints=constraints,
        general_options=general_options,
        algo_options=algo_options,
        gradient=gradient,
        gradient_kwargs=gradient_kwargs,
        gradient_options=gradient_options,
        logging=logging,
        log_options=log_options,
        dashboard=dashboard,
        dash_options=dash_options,
    )

    check_arguments(arguments)

    optim_arguments = []
    results_arguments = []
    database_paths_for_dashboard = []
    for single_arg in arguments:
        optim_kwargs, database_path, result_kwargs = transform_problem(**single_arg)
        optim_arguments.append(optim_kwargs)
        results_arguments.append(result_kwargs)
        if database_path is not None:
            database_paths_for_dashboard.append(database_path)

    if dashboard:
        dashboard_process = run_dashboard_in_separate_process(
            database_paths=database_paths_for_dashboard
        )

    if len(arguments) == 1:
        # Run only one optimization
        results = [_internal_minimize(**optim_arguments[0])]
    else:
        # Run multiple optimizations
        if "n_cores" not in optim_arguments[0]["general_options"]:
            raise ValueError(
                "n_cores need to be specified in general_options"
                + " if multiple optimizations should be run."
            )
        n_cores = optim_arguments[0]["general_options"]["n_cores"]

        results = Parallel(n_jobs=n_cores)(
            delayed(_internal_minimize)(**optim_kwargs)
            for optim_kwargs in optim_arguments
        )

    if dashboard and dashboard_process is not None:
        if not results_arguments[0]["keep_dashboard_alive"]:
            dashboard_process.terminate()

    results = _process_optimization_results(results, results_arguments)

    return results


def _internal_minimize(
    internal_criterion,
    internal_params,
    bounds,
    origin,
    algo_name,
    algo_options,
    internal_gradient,
    database,
    general_options,
):
    """Run one optimization of the transformed optimization problem.
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
        internal_criterion (func): The transformed criterion function.
            It takes the internal_params numpy array as only argument, automatically
            enforcing constraints specified by the user. It calls the original
            criterion function after the necessary reparametrizations.
            If logging is activated it protocols every call automatically to the
            specified database.
        internal_params (np.array): One-dimenisonal array with the values of
            the free parameters.
        bounds (tuple): tuple of the length of internal_params. Every entry contains
            the lower and upper bound of the respective internal parameter.
        origin (str): Name of the package to which the algorithm belongs.
        algo_name (str): Name of the algorithm.
        algo_options (dict): Algorithm specific configurations.
        internal_gradient (func): The internal gradient
        database (sqlalchemy.MetaData or False). The engine that connects to the
            database can be accessed via ``database.bind``. This is only used to record
            the start and end of the optimization
        general_options (dict): Only used to pass the start_criterion_value in case
            the tao pounders algorithm is used.
    Returns:
        results (tuple): Tuple of the harmonized result info dictionary and the params
            DataFrame with the minimizing parameter values of the untransformed problem
            as specified of the user.
    """
    if database:
        update_scalar_field(database, "optimization_status", "running")

    if origin in ["nlopt", "pygmo"]:
        results = minimize_pygmo_np(
            internal_criterion,
            internal_params,
            bounds,
            origin,
            algo_name,
            algo_options,
            internal_gradient,
        )
    elif origin == "scipy":
        results = minimize_scipy_np(
            internal_criterion,
            internal_params,
            bounds=bounds,
            algo_name=algo_name,
            algo_options=algo_options,
            gradient=internal_gradient,
        )
    elif origin == "tao":
        crit_val = general_options["_start_criterion_value"]
        len_criterion_value = 1 if np.isscalar(crit_val) else len(crit_val)
        results = minimize_pounders_np(
            internal_criterion,
            internal_params,
            bounds,
            n_errors=len_criterion_value,
            **algo_options,
        )
    else:
        raise NotImplementedError("Invalid algorithm requested.")

    if database:
        update_scalar_field(database, "optimization_status", results["status"])

    return results


def _process_optimization_results(results, results_arguments):
    """Expand the solutions back to the original problems.
    Args:
        results (list):
            list of dictionaries with the harmonized results objects.
        results_arguments (list):
            each element is a dictionary supplying the start params DataFrame
            and the constraints to the original problem.
            The keys are "params", "constraints" and "keep_dashboard_alive".
    Returns:
        results (tuple): Tuple of the harmonized result info dictionary and the params
            DataFrame with the minimizing parameter values of the untransformed problem
            as specified of the user.
    """
    new_results = []
    for res, args in zip(results, results_arguments):
        res["x"] = list(res["x"])
        start_params = args["params"]
        params = reparametrize_from_internal(
            internal=np.array(res["x"]),
            fixed_values=start_params["_internal_fixed_value"].to_numpy(),
            pre_replacements=start_params["_pre_replacements"].to_numpy(dtype="int"),
            processed_constraints=args["constraints"],
            post_replacements=start_params["_post_replacements"].to_numpy(dtype="int"),
            processed_params=start_params,
        )
        new_results.append((res, params))

    if len(new_results) == 1:
        new_results = new_results[0]
    return new_results
