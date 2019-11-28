import functools

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


def logging(database, tables):
    """Log rows to tables in a database."""

    def decorator_logging(func):
        @functools.wraps(func)
        def wrapper_logging(x, *args, **kwargs):
            out = func(x, *args, **kwargs)
            append_rows(database, tables, [x, out])

            return out

        return wrapper_logging

    return decorator_logging
