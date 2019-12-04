import functools
import itertools

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


def log_parameters_and_fitness(database, tables):
    """Log rows to tables in a database."""

    def decorator_log_parameters_and_fitness(func):
        @functools.wraps(func)
        def wrapper_log_parameters_and_fitness(params, *args, **kwargs):
            criterion_value = func(params, *args, **kwargs)

            if database:
                adj_params = params.copy().set_index("name")["value"]
                append_rows(database, tables, [adj_params, {"value": criterion_value}])

            return criterion_value

        return wrapper_log_parameters_and_fitness

    return decorator_log_parameters_and_fitness


def log_gradient(database, names):
    """Log rows to tables in a database."""

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
    """Log rows to tables in a database."""

    counter = itertools.count(1)

    def decorator_log_gradient_status(func):
        @functools.wraps(func)
        def wrapper_log_gradient_status(params, *args, **kwargs):
            if database:
                c = next(counter)
                status = (c % n_gradient_evaluations) / n_gradient_evaluations
                append_rows(database, ["gradient_status"], {"value": status})

            criterion_value = func(params, *args, **kwargs)

            return criterion_value

        return wrapper_log_gradient_status

    return decorator_log_gradient_status
