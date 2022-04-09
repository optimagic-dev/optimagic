"""Process user provided functions."""
import inspect
import warnings
from functools import partial

from estimagic.exceptions import InvalidFunctionError
from estimagic.exceptions import InvalidKwargsError


def process_func_of_params(func, kwargs, name="your function"):

    kept, ignored = filter_kwargs(func, kwargs)

    if ignored:
        warnings.warn(
            f"The following user provided keyword arguments for {name} were ignored "
            " because they are not valid function arguments."
        )

    out = partial(func, **kept)

    unpartialled_args = get_unpartialled_arguments(out)
    no_default_args = get_arguments_without_default(out)

    no_free_argument_left = len(unpartialled_args) < 1

    if no_free_argument_left and kept:
        raise InvalidKwargsError(
            f"Too many keyword arguments for {name}. After applying all keyword "
            "arguments there must be at least one free argument (the params) left."
        )
    elif no_free_argument_left:
        raise InvalidFunctionError(f"{name} must have at least one free argument.")

    required_args = unpartialled_args.intersection(no_default_args)
    too_many_required_arguments = len(required_args) > 1

    if too_many_required_arguments:
        raise InvalidKwargsError(
            f"Too few keyword arguments for {name}. After applying all keyword "
            "arguments at most one required argument (the params) should remain. "
            f"in your case the following required arguments remain: {required_args}."
        )

    return out


def filter_kwargs(func, kwargs):
    valid = get_unpartialled_arguments(func)

    kept = {key: val for key, val in kwargs.items() if key in valid}

    ignored = {key: val for key, val in kwargs.items() if key not in valid}

    return kept, ignored


def get_unpartialled_arguments(func):
    unpartialled = set(inspect.signature(func).parameters)

    if isinstance(func, partial):
        partialed_in = set(func.keywords)
        unpartialled = unpartialled - partialed_in

    return unpartialled


def get_arguments_without_default(func):
    args = dict(inspect.signature(func).parameters)
    no_default = []
    for name, arg in args.items():
        if not hasattr(arg.default, "__len__"):
            if arg.default == inspect.Parameter.empty:
                no_default.append(name)

    no_default = set(no_default)
    return no_default
