import functools
import itertools
import traceback

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


def negative_criterion(criterion):
    """Turn maximization into minimization."""

    @functools.wraps(criterion)
    def wrapper_negative_criterion(*args, **kwargs):
        out = criterion(*args, **kwargs)
        if np.isscalar(out):
            criterion_value = -out
            comparison_plot_data = pd.DataFrame({"value": [np.nan]})
        else:
            criterion_value, comparison_plot_data = -out[0], out[1]

        return criterion_value, comparison_plot_data

    return wrapper_negative_criterion


def log_evaluation(database, tables):
    """Log parameters and fitness values.

    The criterion function is allowed to have two returns:

    1. The criterion value which can be a scalar or an array, e.g., moments in MSM.
    2. Additional data for the comparison plot in the tidy data format.

    """

    def decorator_log_evaluation(func):
        @functools.wraps(func)
        def wrapper_log_evaluation(params, *args, **kwargs):
            out = func(params, *args, **kwargs)

            if np.isscalar(out):
                criterion_value = out
                cp_data = {"value": pd.DataFrame({"value": [np.nan]})}
            else:
                criterion_value, comparison_plot_data = out
                cp_data = {"value": comparison_plot_data["value"].to_numpy()}

            if database:
                adj_params = params.copy().set_index("name")["value"]
                crit_val = {"value": criterion_value}

                append_rows(database, tables, [adj_params, crit_val, cp_data])

            return criterion_value

        return wrapper_log_evaluation

    return decorator_log_evaluation


def aggregate_criterion_output(aggregation_func):
    """Aggregate the criterion output."""

    def decorator_aggregate_criterion_output(func):
        @functools.wraps(func)
        def wrapper_aggregate_criterion_output(params, *args, **kwargs):
            out = func(params, *args, **kwargs)

            if isinstance(out, np.ndarray):
                criterion_value = aggregation_func(out)
                comparison_plot_data = pd.DataFrame({"value": out})
            else:
                criterion_components, comparison_plot_data = out
                criterion_value = aggregation_func(criterion_components)

            return criterion_value, comparison_plot_data

        return wrapper_aggregate_criterion_output

    return decorator_aggregate_criterion_output


def log_gradient(database, names):
    """Log gradient."""

    def decorator_log_gradient(func):
        @functools.wraps(func)
        def wrapper_log_gradient(*args, **kwargs):
            gradient = func(*args, **kwargs)

            if database:
                data = [dict(zip(names, gradient))]
                append_rows(database, ["gradient_history"], data)

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
            out = func(params, *args, **kwargs)

            if np.isscalar(out):
                criterion_value = out
            else:
                criterion_value, _ = out

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


def handle_exceptions(database, params, constraints, start_params, general_options):
    def decorator_handle_exceptions(func):
        @functools.wraps(func)
        def wrapper_handle_exceptions(x, *args, **kwargs):
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
                    if database:
                        exception_info = traceback.format_exc()
                        p = reparametrize_from_internal(
                            internal=x,
                            fixed_values=params["_internal_fixed_value"].to_numpy(),
                            pre_replacements=params["_pre_replacements"]
                            .to_numpy()
                            .astype(int),
                            processed_constraints=constraints,
                            post_replacements=(
                                params["_post_replacements"].to_numpy().astype(int)
                            ),
                            processed_params=params,
                        )
                        msg = (
                            exception_info
                            + "\n\n"
                            + "The parameters are\n\n"
                            + p["value"].to_csv(sep="\t", header=True)
                        )
                        append_rows(database, "exceptions", {"value": msg})

                    out = min(
                        MAX_CRITERION_PENALTY,
                        constant + slope * np.linalg.norm(x - start_params),
                    )

            return out

        return wrapper_handle_exceptions

    return decorator_handle_exceptions
