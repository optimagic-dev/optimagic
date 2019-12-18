import numpy as np

from estimagic.config import DEFAULT_DATABASE_NAME
from estimagic.decorators import aggregate_criterion_output
from estimagic.optimization.optimize import maximize


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
        wrapped_crit = [
            aggregate_criterion_output(np.mean)(crit_func) for crit_func in criterion
        ]
    else:
        wrapped_crit = aggregate_criterion_output(np.mean)(criterion)

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

    return results
