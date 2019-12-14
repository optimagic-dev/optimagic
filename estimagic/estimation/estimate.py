import numpy as np
import pandas as pd

from estimagic.config import DEFAULT_DATABASE_NAME
from estimagic.decorators import log_estimate_likelihood
from estimagic.logging.create_database import prepare_database_for_estimation
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

    out = criterion(params, **criterion_kwargs)
    log_contributions = pd.Series(out) if isinstance(out, np.ndarray) else out[1]

    loggings = []
    for args_one_run in arguments:
        logging = args_one_run["logging"]
        database = prepare_database_for_estimation(logging, log_contributions)

        wrapped_criterion = log_estimate_likelihood(database)(criterion)
        loggings.append(database)

    results = maximize(
        wrapped_criterion,
        params,
        algorithm,
        criterion_kwargs,
        constraints,
        general_options,
        algo_options,
        gradient_options,
        loggings,
        log_options,
        dashboard,
        db_options,
    )

    return results
