"""This module contains various decorators.

There are two kinds of decorators defined in this module which consists of either two or
three nested functions. The former are decorators without and the latter with arguments.

For more information on decorators, see this `guide`_ on https://realpython.com which
provides a comprehensive overview.

.. _guide:
    https://realpython.com/primer-on-python-decorators/

"""
import functools
import warnings

import numpy as np
import pandas as pd

from estimagic.exceptions import get_traceback
from estimagic.optimization.reparametrize import reparametrize_from_internal


def numpy_interface(params, constraints=None):
    """Convert x to params.

    This decorated function receives a NumPy array of parameters and converts it to a
    :class:`pandas.DataFrame` which can be handled by the user's criterion function.

    For convenience, the decorated function can also be called directly with a
    params DataFrame. In that case, the decorator does nothing.

    Args:
        params (pandas.DataFrame): See :ref:`params`.
        constraints (list of dict): Contains constraints.

    """

    def decorator_numpy_interface(func):
        @functools.wraps(func)
        def wrapper_numpy_interface(x, *args, **kwargs):
            if isinstance(x, pd.DataFrame) and "value" in x.columns:
                p = x
            elif constraints is None:
                p = params.copy()
                p["value"] = x
            elif isinstance(x, np.ndarray):
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
            else:
                raise ValueError(
                    "x must be a numpy array or DataFrame with 'value' column."
                )

            criterion_value = func(p, *args, **kwargs)

            if isinstance(criterion_value, (pd.DataFrame, pd.Series)):
                criterion_value = criterion_value.to_numpy()

            return criterion_value

        return wrapper_numpy_interface

    return decorator_numpy_interface


def catch(
    func=None,
    *,
    exception=Exception,
    exclude=(KeyboardInterrupt, SystemExit),
    onerror=None,
    default=None,
    warn=True,
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

                exc_info = get_traceback()

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
