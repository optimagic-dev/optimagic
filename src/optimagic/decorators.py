"""This module contains various decorators.

There are two kinds of decorators defined in this module which consists of either two or
three nested functions. The former are decorators without and the latter with arguments.

For more information on decorators, see this `guide
`_ on https://realpython.com

which
provides a comprehensive overview.

.. _guide:

https://realpython.com/primer-on-python-decorators/

"""

import functools
import warnings

from optimagic.exceptions import get_traceback


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

                tb = get_traceback()

                if warn:
                    msg = f"The following exception was caught:\n\n{tb}"
                    warnings.warn(msg)

                if default == "__traceback__":
                    res = tb
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


def deprecated(func, msg):
    def decorator_deprecated(func):
        @functools.wraps(func)
        def wrapper_deprecated(*args, **kwargs):
            warnings.warn(msg, FutureWarning)
            return func(*args, **kwargs)

        return wrapper_deprecated

    if callable(func):
        return decorator_deprecated(func)
    else:
        return decorator_deprecated
