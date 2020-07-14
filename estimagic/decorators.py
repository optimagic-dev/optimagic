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
import warnings
from datetime import datetime as dt

try:
    from better_exceptions import format_exception
except ImportError:
    from traceback import format_exception

import sys

import jax.numpy as jnp
import numpy as np
import pandas as pd

from estimagic.config import MAX_CRITERION_PENALTY
from estimagic.logging.database_utilities import append_rows
from estimagic.logging.database_utilities import update_scalar_field
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


def negative_gradient(gradient):
    """Switch the sign of the gradient."""
    if gradient is None:
        wrapper_negative_gradient = None
    else:

        @functools.wraps(gradient)
        def wrapper_negative_gradient(*args, **kwargs):
            return -1 * gradient(*args, **kwargs)

    return wrapper_negative_gradient


def log_evaluation(func=None, *, database, tables):
    """Log parameters and fitness values.

    This decorator can be used with and without parentheses and accepts only keyword
    arguments.

    """

    def decorator_log_evaluation(func):
        @functools.wraps(func)
        def wrapper_log_evaluation(params, *args, **kwargs):
            criterion_value, comparison_plot_data = func(params, *args, **kwargs)

            if database:
                adj_params = params.copy().set_index("name")["value"]
                cp_data = {"value": comparison_plot_data["value"].to_numpy()}
                crit_val = {"value": criterion_value}
                timestamp = {"value": dt.now()}

                append_rows(
                    database=database,
                    tables=tables,
                    rows=[adj_params, crit_val, cp_data, timestamp],
                )

            return criterion_value

        return wrapper_log_evaluation

    if callable(func):
        return decorator_log_evaluation(func)
    else:
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


def log_gradient_status(func=None, *, database, n_gradient_evaluations):
    """Log the gradient status.

    The gradient status is between 0 and 1 and shows the current share of finished
    function evaluations to compute the gradients.

    This decorator can be used with and without parentheses and accepts only keyword
    arguments.

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

    if callable(func):
        return decorator_log_gradient_status(func)
    else:
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
                # Adjust the criterion value at the start.
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


def de_scalarize(x_was_scalar):
    """Create a function with non-scalar input and output.

    Examples:

    >>> @de_scalarize(True)
    ... def f(x):
    ...     return x

    >>> f(3)
    Traceback (most recent call last):
        ...
    TypeError: 'int' object is not subscriptable

    >>> f(np.array([3]))
    array([3])

    >>> @de_scalarize(True)
    ... def g(x):
    ...     return 3

    >>> g(np.ones(3))
    array([3])

    """

    def decorator_de_scalarize(func):
        @functools.wraps(func)
        def wrapper_de_scalarize(x, *args, **kwargs):
            x = x[0] if x_was_scalar else x
            return np.atleast_1d(func(x, *args, **kwargs))

        return wrapper_de_scalarize

    return decorator_de_scalarize


def hide_jax(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        args = [_to_jax(arg) for arg in args]
        res = func(*args, **kwargs)
        if isinstance(res, tuple):
            res = (_from_jax(obj) for obj in res)
        else:
            res = _from_jax(res)
        return res

    return wrapper


def _to_jax(obj):
    if isinstance(obj, (np.ndarray, pd.Series, pd.DataFrame, list)):
        obj = jnp.array(obj)
    return obj


def _from_jax(obj):
    if isinstance(obj, jnp.ndarray):
        obj = np.array(obj)
    return obj


def catch(
    func=None,
    *,
    exception=Exception,
    exclude=(KeyboardInterrupt, SystemExit),
    onerror=None,
    default=None,
    warn=False,
    reraise=False,
):
    """Catch and handle exceptions.

    This decorator can be used with and without additional arguments.

    Args:
        exception (Exception or tuple): One or several exceptions that
            are caught and handled. By default all Exceptions are
            caught and handled.
        exclude (Exception or tuple): One or several exceptionts that
            are not caught. By default those are KeyboardInterrupt and
            SystemExit.
        onerror (None or Callable): Callable that takes an Exception
            as only argument. This is called when an exception occurs.
        default: Value that is returned when as the output of func when
            an exception occurs. Can be one of the following:
            - a constant
            - "__traceback__", in this case a string with a traceback is returned.
            - callable with the same signature as func.
        warn (bool): If True, the exception is converted to a warning.
        reraise (bool): If True, the exception is raised after handling it.

    """

    def decorator_catch(func):
        @functools.wraps(func)
        def wrapper_catch(*args, **kwargs):
            try:
                res = func(*args, **kwargs)
            except exclude:
                raise
            except exception as e:

                if onerror is not None:
                    onerror(e)

                if reraise:
                    raise e

                exc_info = format_exception(*sys.exc_info())
                if isinstance(exc_info, list):
                    exc_info = "".join(exc_info)

                if warn:
                    msg = f"The following exception was caught:\n\n{exc_info}"
                    warnings.warn(msg)

                if default == "__traceback__":
                    res = exc_info
                elif callable(default):
                    res = default(*args, **kwargs)
                else:
                    res = default
            return res

        return wrapper_catch

    if callable(func):
        return decorator_catch(func)
    else:
        return decorator_catch


def unpack(func=None, symbol=None):
    def decorator_unpack(func):
        if symbol is None:

            @functools.wraps(func)
            def wrapper_unpack(arg):
                return func(arg)

        elif symbol == "*":

            @functools.wraps(func)
            def wrapper_unpack(arg):
                return func(*arg)

        elif symbol == "**":

            @functools.wraps(func)
            def wrapper_unpack(arg):
                return func(**arg)

        return wrapper_unpack

    if callable(func):
        return decorator_unpack(func)
    else:
        return decorator_unpack
