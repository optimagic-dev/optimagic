import functools
import inspect
import warnings
from functools import partial

from estimagic.utilities import propose_alternatives
from estimagic.optimization.tranquilo.options import update_option_bundle


def get_component(
    name_or_func,
    component_name,
    func_dict=None,
    default_options=None,
    user_options=None,
    redundant_option_handling="ignore",
    redundant_argument_handling="ignore",
    mandatory_signature=None,
):
    """Process a function that represents an interchangeable component of tranquilo.

    The function is either a built in function or a user provided function. In all
    cases we run some checks that the signature of the function is correct and then
    partial all static options into the function.

    Args:
        name_or_func (str or callable): Name of a function or function.
        component_name (str): Name of the component. Used in error messages. Examples
            would be "subsolver" or "model".
        func_dict (dict): Dict with function names as keys and functions as values.
        default_options (NamedTuple): Default options as a dict or NamedTuple. The
            default options will be updated by the user options.
        user_options (NamedTuple, Dict or None): User options as a dict or NamedTuple.
            The default options will be updated by the user options.
        redundant_option_handling (str): How to handle redundant options. Can be
            "warn", "raise" or "ignore". Default "ignore".
        redundant_argument_handling (str): How to handle redundant arguments passed
            to the processed function at runtime. Can be "warn", "raise" or "ignore".
            Default "ignore".
        mandatory_signature (list): List or tuple of arguments that must be in the
            signature of all functions in `func_dict`. These can be options or
            arguments. Otherwise, a ValueError is raised.

    Returns:
        callable: The processed function.

    """

    _func, _name = _get_function_and_name(
        name_or_func=name_or_func,
        component_name=component_name,
        func_dict=func_dict,
    )

    _all_arguments = list(inspect.signature(_func).parameters)

    _valid_options = _get_valid_options(
        default_options=default_options,
        user_options=user_options,
        signature=_all_arguments,
        name=_name,
        component_name=component_name,
        redundant_option_handling=redundant_option_handling,
    )

    _fail_if_mandatory_argument_is_missing(
        mandatory_arguments=mandatory_signature,
        signature=_all_arguments,
        name=_name,
        component_name=component_name,
    )

    _partialled = partial(_func, **_valid_options)

    if redundant_argument_handling == "raise":
        out = _partialled
    else:
        out = _add_redundant_argument_handling(
            func=_partialled,
            signature=_all_arguments,
            warn=redundant_argument_handling == "warn",
        )

    return out


def _get_function_and_name(name_or_func, component_name, func_dict):
    """Get the function and its name.

    Args:
        name_or_func (str or callable): Name of a function or function.
        component_name (str): Name of the component. Used in error messages. Examples
            would be "subsolver" or "model".
        func_dict (dict): Dict with function names as keys and functions as values.

    Returns:
        tuple: The function and its name.

    """
    func_dict = {} if func_dict is None else func_dict
    if isinstance(name_or_func, str):
        if name_or_func in func_dict:
            _func = func_dict[name_or_func]
            _name = name_or_func
        else:
            _proposal = propose_alternatives(name_or_func, list(func_dict))
            msg = (
                f"If {component_name} is a string, it must be one of the built in "
                f"{component_name}s. Did you mean: {_proposal}?"
            )
            raise ValueError(msg)
    elif callable(name_or_func):
        _func = name_or_func
        _name = _func.__name__
    else:
        raise TypeError("name_or_func must be a string or a callable.")

    return _func, _name


def _get_valid_options(
    default_options,
    user_options,
    signature,
    name,
    component_name,
    redundant_option_handling,
):
    """Get the options that are valid for the function.

    Args:
        default_options (NamedTuple): Default options as a dict or NamedTuple. The
            default options will be updated by the user options.
        user_options (NamedTuple, Dict or None): User options as a dict or NamedTuple.
            The default options will be updated by the user options.
        signature (list): List of arguments that are present in the signature.
        name (str): Name of the function.
        component_name (str): Name of the component. Used in error messages. Examples
            would be "subsolver" or "model".
        redundant_option_handling (str): How to handle redundant options. Can be

    Returns:
        dict: Valid options.

    """
    _options = update_option_bundle(default_options, user_options=user_options)
    _options = _options._asdict()

    _valid_options = {k: v for k, v in _options.items() if k in signature}
    _redundant_options = {k: v for k, v in _options.items() if k not in signature}

    if redundant_option_handling == "warn" and _redundant_options:
        msg = (
            f"The following options are not supported by the {component_name} {name} "
            f"and will be ignored: {list(_redundant_options)}."
        )
        warnings.warn(msg)

    elif redundant_option_handling == "raise" and _redundant_options:
        msg = (
            f"The following options are not supported by the {component_name} {name}: "
            f"{list(_redundant_options)}."
        )
        raise ValueError(msg)

    return _valid_options


def _fail_if_mandatory_argument_is_missing(
    mandatory_arguments, signature, name, component_name
):
    """Check if any mandatory arguments are missing in the signature of the function.

    Args:
        mandatory_arguments (list): List of mandatory arguments.
        signature (list): List of arguments that are present in the signature.
        name (str): Name of the function.
        component_name (str): Name of the component. Used in error messages. Examples
            would be "subsolver" or "model".

    Returns:
        None

    Raises:
        ValueError: If any mandatory arguments are missing in the signature of the
            function.

    """
    mandatory_arguments = [] if mandatory_arguments is None else mandatory_arguments

    _missing = [arg for arg in mandatory_arguments if arg not in signature]

    if _missing:
        msg = (
            f"The following mandatory arguments are missing in the signature of the "
            f"{component_name} {name}: {_missing}."
        )
        raise ValueError(msg)


def _add_redundant_argument_handling(func, signature, warn):
    """Allow func to be called with arguments that are not in the signature.

    Args:
        func (callable): The function to be wrapped.
        signature (list): List of arguments that are supported by func.
        warn (bool): Whether to warn about redundant arguments.

    Returns:
        callable: The wrapped function.

    """

    @functools.wraps(func)
    def _wrapper_add_redundant_argument_handling(*args, **kwargs):
        _kwargs = {**dict(zip(signature[: len(args)], args)), **kwargs}

        _redundant = {k: v for k, v in _kwargs.items() if k not in signature}
        _valid = {k: v for k, v in _kwargs.items() if k in signature}

        if warn and _redundant:
            msg = (
                f"The following arguments are not supported by the function "
                f"{func.__name__} and will be ignored: {_redundant}."
            )
            warnings.warn(msg)

        out = func(**_valid)
        return out

    return _wrapper_add_redundant_argument_handling
