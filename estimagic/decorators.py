import functools
import itertools

import numpy as np
import pandas as pd

from estimagic.config import MAX_CRITERION_PENALTY
from estimagic.logging.update_database import append_rows
from estimagic.logging.update_database import update_scalar_field
from estimagic.optimization.reparametrize import reparametrize_from_internal


def numpy_interface(params, constraints=None):
    """Convert x to params."""

    def decorator_numpy_interface(func):
        @functools.wraps(func)
        def wrapper_numpy_interface(x, *args, **kwargs):
            # Handle usage in :func:`internal_function` for gradients.
            if constraints is None:
                p = params.copy()
                p["value"] = x

            # Handle usage in :func:`internal_criterion`.
            else:
                p = reparametrize_from_internal(
                    internal=x,
                    fixed_values=params["_internal_fixed_value"].to_numpy(),
                    pre_replacements=params["_pre_replacements"].to_numpy().astype(int),
                    processed_constraints=constraints,
                    post_replacements=(
                        params["_post_replacements"].to_numpy().astype(int)
                    ),
                    processed_params=params,
                )

            out = func(p, *args, **kwargs)

            if isinstance(out, (pd.DataFrame, pd.Series)):
                out = out.to_numpy()

            return out

        return wrapper_numpy_interface

    return decorator_numpy_interface


def log_evaluation(database, tables):
    """Log parameters and fitness values."""

    def decorator_evaluation(func):
        @functools.wraps(func)
        def wrapper_evaluation(params, *args, **kwargs):
            criterion_value = func(params, *args, **kwargs)

            if database:
                adj_params = params.copy().set_index("name")["value"]
                append_rows(database, tables, [adj_params, {"value": criterion_value}])

            return criterion_value

        return wrapper_evaluation

    return decorator_evaluation


def log_estimate_likelihood(database):
    """Log log likelihood contributions."""

    def decorator_estimate_likelihood(func):
        @functools.wraps(func)
        def wrapper_estimate_likelihood(params, *args, **kwargs):
            out = func(params, *args, **kwargs)

            if isinstance(out, np.ndarray):
                log_like_obs = out
                log_contributions = out
            else:
                log_like_obs, df_or_series = out
                if isinstance(df_or_series, pd.Series):
                    log_contributions = df_or_series.to_numpy()
                else:
                    log_contributions = df_or_series["value"].to_numpy()

            criterion_value = log_like_obs.mean()

            if database:
                append_rows(
                    database,
                    "log_contributions",
                    {"value": log_contributions.astype("float32")},
                )

            return criterion_value

        return wrapper_estimate_likelihood

    return decorator_estimate_likelihood


def log_gradient(database, names):
    """Log gradient."""

    def decorator_log_gradient(func):
        @functools.wraps(func)
        def wrapper_log_gradient(*args, **kwargs):
            gradient = func(*args, **kwargs)

            if database:
                append_rows(
                    database, ["gradient_history"], [dict(zip(names, gradient))]
                )

            return gradient

        return wrapper_log_gradient

    return decorator_log_gradient


def log_gradient_status(database, n_gradient_evaluations):
    """Log the gradient status.

    The gradient status is between 0 and 1 and shows the share of function evaluations
    to compute the gradients.

    """
    counter = itertools.count(1)

    def decorator_log_gradient_status(func):
        @functools.wraps(func)
        def wrapper_log_gradient_status(params, *args, **kwargs):
            criterion_value = func(params, *args, **kwargs)

            if database:
                c = next(counter)
                if n_gradient_evaluations is None:
                    status = c
                else:
                    status = (c % n_gradient_evaluations) / n_gradient_evaluations
                    status = 1 if status == 0 else status
                update_scalar_field(database, "gradient_status", status)

            return criterion_value

        return wrapper_log_gradient_status

    return decorator_log_gradient_status


def exception_handling(start_params, general_options):
    def decorator_exception_handling(func):
        @functools.wraps(func)
        def wrapper_exception_handling(x, *args, **kwargs):
            try:
                out = func(x, *args, **kwargs)
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                criterion_value = general_options["start_criterion_value"]
                constant, slope = general_options.get(
                    "criterion_exception_penalty", (None, None)
                )
                constant = 2 * criterion_value if constant is None else constant
                slope = 0.1 * criterion_value if slope is None else slope
                raise_exc = general_options.get("criterion_exception_raise", False)

                if raise_exc:
                    raise e
                else:
                    out = min(
                        MAX_CRITERION_PENALTY,
                        constant + slope * np.linalg.norm(x - start_params),
                    )

            return out

        return wrapper_exception_handling

    return decorator_exception_handling
