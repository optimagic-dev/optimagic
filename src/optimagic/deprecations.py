import warnings
from functools import wraps
from typing import Any, Callable, ParamSpec

from optimagic.optimization.fun_value import (
    LeastSquaresFunctionValue,
    LikelihoodFunctionValue,
    ScalarFunctionValue,
)
from optimagic.parameters.bounds import Bounds
from optimagic.typing import ProblemType


def throw_criterion_future_warning():
    msg = (
        "To align optimagic with scipy.optimize, the `criterion` argument has been "
        "renamed to `fun`. Please use `fun` instead of `criterion`. Using `criterion` "
        " will become an error in optimagic version 0.6.0 and later."
    )
    warnings.warn(msg, FutureWarning)


def throw_criterion_kwargs_future_warning():
    msg = (
        "To align optimagic with scipy.optimize, the `criterion_kwargs` argument has "
        "been renamed to `fun_kwargs`. Please use `fun_kwargs` instead of "
        "`criterion_kwargs`. Using `criterion_kwargs` will become an error in "
        "optimagic version 0.6.0 and later."
    )
    warnings.warn(msg, FutureWarning)


def throw_derivative_future_warning():
    msg = (
        "To align optimagic with scipy.optimize, the `derivative` argument has been "
        "renamed to `jac`. Please use `jac` instead of `derivative`. Using `derivative`"
        " will become an error in optimagic version 0.6.0 and later."
    )
    warnings.warn(msg, FutureWarning)


def throw_derivative_kwargs_future_warning():
    msg = (
        "To align optimagic with scipy.optimize, the `derivative_kwargs` argument has "
        "been renamed to `jac_kwargs`. Please use `jac_kwargs` instead of "
        "`derivative_kwargs`. Using `derivative_kwargs` will become an error in "
        "optimagic version 0.6.0 and later."
    )
    warnings.warn(msg, FutureWarning)


def throw_criterion_and_derivative_future_warning():
    msg = (
        "To align optimagic with scipy.optimize, the `criterion_and_derivative` "
        "argument has been renamed to `fun_and_jac`. Please use `fun_and_jac` "
        "instead of `criterion_and_derivative`. Using `criterion_and_derivative` "
        "will become an error in optimagic version 0.6.0 and later."
    )
    warnings.warn(msg, FutureWarning)


def throw_criterion_and_derivative_kwargs_future_warning():
    msg = (
        "To align optimagic with scipy.optimize, the `criterion_and_derivative_kwargs` "
        "argument has been renamed to `fun_and_jac_kwargs`. Please use "
        "`fun_and_jac_kwargs` instead of `criterion_and_derivative_kwargs`. Using "
        "`criterion_and_derivative_kwargs` will become an error in optimagic version "
        "0.6.0 and later."
    )
    warnings.warn(msg, FutureWarning)


def throw_scaling_options_future_warning():
    msg = (
        "Specifying scaling options via the argument `scaling_options` is deprecated "
        "and will be removed in optimagic version 0.6.0 and later. You can pass these "
        "options directly to the `scaling` argument instead."
    )
    warnings.warn(msg, FutureWarning)


def replace_and_warn_about_deprecated_algo_options(algo_options):
    if not isinstance(algo_options, dict):
        return algo_options

    algo_options = {k.replace(".", "_"): v for k, v in algo_options.items()}

    replacements = {
        "stopping_max_criterion_evaluations": "stopping_maxfun",
        "stopping_max_iterations": "stopping_maxiter",
        "convergence_absolute_criterion_tolerance": "convergence_ftol_abs",
        "convergence_relative_criterion_tolerance": "convergence_ftol_rel",
        "convergence_scaled_criterion_tolerance": "convergence_ftol_scaled",
        "convergence_absolute_params_tolerance": "convergence_xtol_abs",
        "convergence_relative_params_tolerance": "convergence_xtol_rel",
        "convergence_absolute_gradient_tolerance": "convergence_gtol_abs",
        "convergence_relative_gradient_tolerance": "convergence_gtol_rel",
        "convergence_scaled_gradient_tolerance": "convergence_gtol_scaled",
    }

    present = sorted(set(algo_options) & set(replacements))
    if present:
        msg = (
            "The following keys in `algo_options` are deprecated and will be removed "
            "in optimagic version 0.6.0 and later. Please replace them as follows:\n"
        )
        for k in present:
            msg += f"  {k} -> {replacements[k]}\n"

        warnings.warn(msg, FutureWarning)

    out = {k: v for k, v in algo_options.items() if k not in present}
    for k in present:
        out[replacements[k]] = algo_options[k]

    return out


def replace_and_warn_about_deprecated_bounds(
    lower_bounds,
    upper_bounds,
    bounds,
    soft_lower_bounds=None,
    soft_upper_bounds=None,
):
    old_bounds = {
        "lower": lower_bounds,
        "upper": upper_bounds,
        "soft_lower": soft_lower_bounds,
        "soft_upper": soft_upper_bounds,
    }

    old_present = [k for k, v in old_bounds.items() if v is not None]

    if old_present:
        substring = ", ".join(f"{b}_bound" for b in old_present)
        substring = substring.replace(", ", ", and ", -1)
        msg = (
            f"Specifying bounds via the arguments {substring} is "
            "deprecated and will be removed in optimagic version 0.6.0 and later. "
            "Please use the `bounds` argument instead."
        )
        warnings.warn(msg, FutureWarning)

    if bounds is None and old_present:
        bounds = Bounds(**old_bounds)

    return bounds


def convert_dict_to_function_value(candidate):
    """Convert the deprecated dictionary output to a suitable FunctionValue object.

    No warning is raised here because this function will be called repeatedly!

    """
    special_keys = ["value", "contributions", "root_contributions"]

    if is_dict_output(candidate):
        info = {k: v for k, v in candidate.items() if k not in special_keys}
        if "root_contributions" in candidate:
            out = LeastSquaresFunctionValue(candidate["root_contributions"], info)
        elif "contributions" in candidate:
            out = LikelihoodFunctionValue(candidate["contributions"], info)
        else:
            out = ScalarFunctionValue(candidate["value"], info)
    else:
        out = candidate

    return out


def is_dict_output(candidate):
    """Check if the output is a dictionary with special keys."""
    special_keys = ["value", "contributions", "root_contributions"]
    return isinstance(candidate, dict) and any(k in candidate for k in special_keys)


def throw_dict_output_warning():
    msg = (
        "Returning a dictionary with the special keys 'value', 'contributions', or "
        "'root_contributions' is deprecated and will be removed in optimagic version "
        "0.6.0 and later. Please use the optimagic.mark.scalar, optimagic.mark."
        "least_squares, or optimagic.mark.likelihood decorators to indicate the type "
        "of problem you are solving. Use optimagic.FunctionValue objects to return "
        "additional information for the logging."
    )
    warnings.warn(msg, FutureWarning)


def infer_problem_type_from_dict_output(output):
    if "root_contributions" in output:
        out = ProblemType.LEAST_SQUARES
    elif "contributions" in output:
        out = ProblemType.LIKELIHOOD
    else:
        out = ProblemType.SCALAR
    return out


P = ParamSpec("P")


def replace_dict_output(func: Callable[P, Any]) -> Callable[P, Any]:
    """Replace the deprecated dictionary output by a suitable FunctionValue.

    This has no effect if the function does not return a dictionary with at least one of
    the special keys "value", "contributions" or "root_contributions" or a tuple where
    the first entry is such a dictionary.

    This decorator does not add a warning because the function will be evaluated many
    times and the warning would pop up too often.

    """

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
        raw = func(*args, **kwargs)
        # fun and jac case
        if isinstance(raw, tuple):
            out = (convert_dict_to_function_value(raw[0]), raw[1])
        # fun case
        else:
            out = convert_dict_to_function_value(raw)
        return out

    return wrapper


def throw_key_warning_in_derivatives():
    msg = (
        "The `key` argument in first_derivative and second_derivative is deprecated "
        "and will be removed in optimagic version 0.6.0 and later. Please use the "
        "`unpacker` argument instead. While `key` was a string, `unpacker` is a "
        "callable that takes the output of `func` and returns the desired output that "
        "is then differentiated."
    )
    warnings.warn(msg, FutureWarning)
