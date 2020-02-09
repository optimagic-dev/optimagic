import numpy as np

from estimagic.config import DEFAULT_DATABASE_NAME
from estimagic.decorators import aggregate_criterion_output
from estimagic.decorators import expand_criterion_output
from estimagic.optimization.optimize import maximize
from estimagic.optimization.process_arguments import process_optimization_arguments


def maximize_log_likelihood(
    log_like_obs,
    params,
    algorithm,
    criterion_kwargs=None,
    constraints=None,
    general_options=None,
    algo_options=None,
    gradient_options=None,
    logging=DEFAULT_DATABASE_NAME,
    log_options=None,
    dashboard=False,
    db_options=None,
):
    """Estimate parameters via maximum likelihood.

    This function provides a convenient interface for estimating models via maximum
    likelihood. In the future, it will also calculate standard errors for the solution.

    The criterion function ``log_like_obs`` has to return an array of log likelihoods at
    the first position, not the mean log likelihood. The array is internally aggregated
    to whatever output is needed. For example, the mean is used for maximization, the
    sum for standard error calculations.

    The second return can be a :class:`pandas.DataFrame` in the `tidy data format`_ to
    display the distribution of contributions for subgroups via the comparison plot in
    the future.

    The limitation to log likelihoods instead of likelihoods may seem unnecessarily
    restrictive, but it is preferred for two reasons.

    1. Optimization methods which rely on gradients generally work better optimizing the
       log transformation. See `1`_ for a simplified example.

    2. Using the log transformation to convert products of probabilities to sums of log
       probabilities is numerically more stable as it prevents over- and underflows. See
       `2`_ for an example.

    Args:
        log_like_obs (callable or list of callables):
            Python function that takes a pandas DataFrame with parameters as the first
            argument and returns an array of log likelihood contributions as the first
            return.

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

        gradient_options (dict):
            Options for the gradient function.

        logging (str or pathlib.Path): Path to an sqlite3 file which typically has the
            file extension ``.db``. If the file does not exist, it will be created. See
            :ref:`logging` for details.

        log_options (dict): Keyword arguments to influence the logging. See
            :ref:`logging` for details.

        dashboard (bool):
            whether to create and show a dashboard. See :ref:`dashboard` for details.

        db_options (dict):
            dictionary with kwargs to be supplied to the run_server function. See
                :ref:`dashboard` for details.

    Returns:
        results (tuple or list of tuples):
            The return is either a tuple containing a dictionary of the results and the
            parameters or a list of tuples containing multiples of the former.

    .. _tidy data format:
        http://dx.doi.org/10.18637/jss.v059.i10

    .. _1:
        https://stats.stackexchange.com/a/176563/218971

    .. _2:
        https://statmodeling.stat.columbia.edu/2016/06/11/log-sum-of-exponentials/

    """
    if isinstance(log_like_obs, list):
        extended_loglikelobs = [
            expand_criterion_output(crit_func) for crit_func in log_like_obs
        ]
        wrapped_loglikeobs = [
            aggregate_criterion_output(np.mean)(crit_func)
            for crit_func in extended_loglikelobs
        ]
    else:
        extended_loglikelobs = expand_criterion_output(log_like_obs)
        wrapped_loglikeobs = aggregate_criterion_output(np.mean)(extended_loglikelobs)

    results = maximize(
        wrapped_loglikeobs,
        params,
        algorithm,
        criterion_kwargs,
        constraints,
        general_options,
        algo_options,
        gradient_options,
        logging,
        log_options,
        dashboard,
        db_options,
    )

    # To convert the mean log likelihood in the results dictionary to the log
    # likelihood, get the length of contributions for each optimization.
    arguments = process_optimization_arguments(
        criterion=extended_loglikelobs,
        params=params,
        algorithm=algorithm,
        criterion_kwargs=criterion_kwargs,
        constraints=constraints,
        general_options=general_options,
        algo_options=algo_options,
        gradient=None,
        gradient_options=gradient_options,
        logging=logging,
        log_options=log_options,
        dashboard=dashboard,
        db_options=db_options,
    )

    contribs_and_cp_data = [
        args_one_run["criterion"](
            args_one_run["params"], **args_one_run["criterion_kwargs"]
        )
        for args_one_run in arguments
    ]
    n_contributions = [len(c_and_cp[0]) for c_and_cp in contribs_and_cp_data]

    if isinstance(results, list):
        for result, n_contribs in zip(results, n_contributions):
            result[0]["fitness"] = result[0]["fitness"] * n_contribs
    else:
        results[0]["fitness"] = results[0]["fitness"] * n_contributions[0]

    return results
