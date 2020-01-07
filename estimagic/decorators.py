"""This module contains various decorators.

There are two kinds of decorators defined in this module which consists of either two or
three nested functions. The former are decorators without and the latter with arguments.

For more information on decorators, see this `guide`_ on https://realpython.com which
provides a comprehensive overview.

.. _guide:
    https://realpython.com/primer-on-python-decorators/

"""
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
    """Convert x to params.

    This decorator receives a NumPy array of parameters and converts it to a
    :class:`pandas.DataFrame` which can be handled by the user's criterion function.

    Args:
        params (pandas.DataFrame): See :ref:`params`.
        constraints (list of dict): Contains constraints.

    """

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

            criterion_value = func(p, *args, **kwargs)

            if isinstance(criterion_value, (pd.DataFrame, pd.Series)):
                criterion_value = criterion_value.to_numpy()

            return criterion_value

        return wrapper_numpy_interface

    return decorator_numpy_interface


def expand_criterion_output(criterion):
    """Handle one- or two-element criterion returns.

    There are three cases:

    1. The criterion function returns a scalar. Then, do not include any comparison plot
       data.
    2. If the criterion functions returns an array as with maximum likelihood estimation
       or while using POUNDERs, use the array as data for the comparison plot.
    3. If the criterion function returns a criterion value and the data for the
       comparison plot, the return is a tuple.

    """

    @functools.wraps(criterion)
    def wrappper_expand_criterion_output(*args, **kwargs):
        out = criterion(*args, **kwargs)
        if np.isscalar(out):
            criterion_value = out
            comparison_plot_data = pd.DataFrame({"value": [np.nan]})
        elif isinstance(out, np.ndarray):
            criterion_value = out
            comparison_plot_data = pd.DataFrame({"value": criterion_value})
        elif isinstance(out, tuple):
            criterion_value, comparison_plot_data = out[0], out[1]
        else:
            raise NotImplementedError

        return criterion_value, comparison_plot_data

    return wrappper_expand_criterion_output


def negative_criterion(criterion):
    """Turn maximization into minimization by switching the sign."""

    @functools.wraps(criterion)
    def wrapper_negative_criterion(*args, **kwargs):
        criterion_value, comparison_plot_data = criterion(*args, **kwargs)

        return -criterion_value, comparison_plot_data

    return wrapper_negative_criterion


def log_evaluation(database, tables):
    """Log parameters and fitness values."""

    def decorator_log_evaluation(func):
        @functools.wraps(func)
        def wrapper_log_evaluation(params, *args, **kwargs):
            criterion_value, comparison_plot_data = func(params, *args, **kwargs)

            if database:
                adj_params = params.copy().set_index("name")["value"]
                cp_data = {"value": comparison_plot_data["value"].to_numpy()}
                crit_val = {"value": criterion_value}

                append_rows(database, tables, [adj_params, crit_val, cp_data])

            return criterion_value

        return wrapper_log_evaluation

    return decorator_log_evaluation


def aggregate_criterion_output(aggregation_func):
    """Aggregate the return of of criterion functions with non-scalar output.

    This helper allows to conveniently alter the criterion function of the user for
    different purposes. For example, the criterion function for maximum likelihood
    estimation passed to :func:`~estimagic.estimation.estimate.maximize_log_likelihood`
    returns the log likelihood contributions. For the maximization, we need the mean log
    likelihood and the sum for the standard error calculations.

    """

    def decorator_aggregate_criterion_output(func):
        @functools.wraps(func)
        def wrapper_aggregate_criterion_output(params, *args, **kwargs):
            criterion_values, comparison_plot_data = func(params, *args, **kwargs)

            criterion_value = aggregation_func(criterion_values)

            return criterion_value, comparison_plot_data

        return wrapper_aggregate_criterion_output

    return decorator_aggregate_criterion_output


def log_gradient(database, names):
    """Log the gradient.

    The gradient is a vector containing the partial derivatives of the criterion
    function at each of the internal parameters.

    """

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

    The gradient status is between 0 and 1 and shows the current share of finished
    function evaluations to compute the gradients.

    """
    counter = itertools.count(1)

    def decorator_log_gradient_status(func):
        @functools.wraps(func)
        def wrapper_log_gradient_status(params, *args, **kwargs):
            criterion_value, _ = func(params, *args, **kwargs)

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
    """Handle exceptions in the criterion function.

    This decorator catches any exceptions raised inside the criterion function. If the
    exception is a :class:`KeyboardInterrupt` or a :class:`SystemExit`, the user wants
    to stop the optimization and the exception is raised

    For other exceptions, it is assumed that the optimizer proposed parameters which
    could not be handled by the criterion function. For example, the parameters formed
    an invalid covariance matrix which lead to an :class:`numpy.linalg.LinAlgError` in
    the matrix decompositions. Then, we calculate a penalty as a function of the
    criterion value at the initial parameters and some distance between the initial and
    the current parameters.

    """

    def decorator_handle_exceptions(func):
        @functools.wraps(func)
        def wrapper_handle_exceptions(x, *args, **kwargs):
            try:
                out = func(x, *args, **kwargs)
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                start_criterion_value = general_options["start_criterion_value"]
                constant, slope = general_options.get(
                    "criterion_exception_penalty", (None, None)
                )
                constant = 2 * start_criterion_value if constant is None else constant
                slope = 0.1 * start_criterion_value if slope is None else slope
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
