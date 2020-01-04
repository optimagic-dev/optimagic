import numpy as np

from estimagic.config import DEFAULT_DATABASE_NAME
from estimagic.decorators import aggregate_criterion_output
from estimagic.decorators import expand_criterion_output
from estimagic.optimization.optimize import maximize
from estimagic.optimization.process_arguments import process_optimization_arguments


def estimate_likelihood(
    criterion,
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
    """Estimate parameters via likelihood.

    Every criterion function needs to be wrapped with a decorator which converts the log
    likelihood contributions of each observation, the first return, to the likelihood
    value.

    """
    if isinstance(criterion, list):
        wrapped_crit = [expand_criterion_output(crit_func) for crit_func in criterion]
        wrapped_crit = [
            aggregate_criterion_output(np.mean)(crit_func) for crit_func in wrapped_crit
        ]
    else:
        wrapped_crit = expand_criterion_output(criterion)
        wrapped_crit = aggregate_criterion_output(np.mean)(wrapped_crit)

    results = maximize(
        wrapped_crit,
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
        criterion=criterion,
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

    n_contributions = [
        len(
            list(
                args_one_run["criterion"](
                    args_one_run["params"], **args_one_run["criterion_kwargs"]
                )
            )
        )
        for args_one_run in arguments
    ]

    if isinstance(results, list):
        for result, n_contribs in zip(results, n_contributions):
            result[0]["fitness"] = result[0]["fitness"] * n_contribs
    else:
        results[0]["fitness"] = results[0]["fitness"] * n_contributions[0]

    return results
