import functools

import pandas as pd

from estimagic.logging.update_database import append_rows
from estimagic.optimization.reparametrize import reparametrize_from_internal


def x_to_params(params, constraints=None):
    """Convert x to params."""

    def decorator_x_to_params(func):
        @functools.wraps(func)
        def wrapper_x_to_params(x, *args, **kwargs):
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

            return out

        return wrapper_x_to_params

    return decorator_x_to_params


def output_to_numpy(func):
    @functools.wraps(func)
    def wrapper_output_to_numpy(*args, **kwargs):
        out = func(*args, **kwargs)

        if isinstance(out, (pd.DataFrame, pd.Series)):
            out = out.to_numpy()

        return out

    return wrapper_output_to_numpy


def logging(database, tables):
    """Log rows to tables in a database."""

    def decorator_logging(func):
        @functools.wraps(func)
        def wrapper_logging(params, *args, **kwargs):
            criterion_value = func(params, *args, **kwargs)

            if database:
                adj_params = params.copy().set_index("name")["value"]
                append_rows(database, tables, [adj_params, {"value": criterion_value}])

            return criterion_value

        return wrapper_logging

    return decorator_logging
