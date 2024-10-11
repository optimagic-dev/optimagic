import logging
import warnings
from dataclasses import replace
from functools import wraps
from pathlib import Path
from typing import Any, Callable, ParamSpec, cast

from optimagic import mark
from optimagic.constraints import Constraint, InvalidConstraintError
from optimagic.logging.logger import (
    LogOptions,
    SQLiteLogOptions,
)
from optimagic.optimization.fun_value import (
    LeastSquaresFunctionValue,
    LikelihoodFunctionValue,
    ScalarFunctionValue,
)
from optimagic.parameters.bounds import Bounds
from optimagic.typing import AggregationLevel

_logger = logging.getLogger(__name__)


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
        " will become an error in optimagic version 0.6.0 and later. For more details "
        "see the documentation: "
        "https://optimagic.readthedocs.io/en/latest/how_to/how_to_derivatives.html"
    )
    warnings.warn(msg, FutureWarning)


def throw_derivative_kwargs_future_warning():
    msg = (
        "To align optimagic with scipy.optimize, the `derivative_kwargs` argument has "
        "been renamed to `jac_kwargs`. Please use `jac_kwargs` instead of "
        "`derivative_kwargs`. Using `derivative_kwargs` will become an error in "
        "optimagic version 0.6.0 and later. For more details see the documentation: "
        "https://optimagic.readthedocs.io/en/latest/how_to/how_to_derivatives.html"
    )
    warnings.warn(msg, FutureWarning)


def throw_criterion_and_derivative_future_warning():
    msg = (
        "To align optimagic with scipy.optimize, the `criterion_and_derivative` "
        "argument has been renamed to `fun_and_jac`. Please use `fun_and_jac` "
        "instead of `criterion_and_derivative`. Using `criterion_and_derivative` "
        "will become an error in optimagic version 0.6.0 and later. For more details "
        "see the documentation: "
        "https://optimagic.readthedocs.io/en/latest/how_to/how_to_derivatives.html"
    )
    warnings.warn(msg, FutureWarning)


def throw_criterion_and_derivative_kwargs_future_warning():
    msg = (
        "To align optimagic with scipy.optimize, the `criterion_and_derivative_kwargs` "
        "argument has been renamed to `fun_and_jac_kwargs`. Please use "
        "`fun_and_jac_kwargs` instead of `criterion_and_derivative_kwargs`. Using "
        "`criterion_and_derivative_kwargs` will become an error in optimagic version "
        "0.6.0 and later. For more details see the documentation: "
        "https://optimagic.readthedocs.io/en/latest/how_to/how_to_derivatives.html"
    )
    warnings.warn(msg, FutureWarning)


def throw_scaling_options_future_warning():
    msg = (
        "Specifying scaling options via the argument `scaling_options` is deprecated "
        "and will be removed in optimagic version 0.6.0 and later. You can pass these "
        "options directly to the `scaling` argument instead."
    )
    warnings.warn(msg, FutureWarning)


def throw_multistart_options_future_warning():
    msg = (
        "Specifying multistart options via the argument `multistart_options` is "
        "deprecated and will be removed in optimagic version 0.6.0 and later. You can "
        "pass these options directly to the `multistart` argument instead. For more "
        "details see the documentation: "
        "https://optimagic.readthedocs.io/en/latest/how_to/how_to_multistart.html"
    )
    warnings.warn(msg, FutureWarning)


def throw_derivatives_step_ratio_future_warning():
    msg = (
        "The `step_ratio` argument is deprecated and will be removed alongside "
        "Richardson extrapolation in optimagic version 0.6.0."
    )
    warnings.warn(msg, FutureWarning)


def throw_derivatives_n_steps_future_warning():
    msg = (
        "The `n_steps` argument is deprecated and will be removed alongside "
        "Richardson extrapolation in optimagic version 0.6.0."
    )
    warnings.warn(msg, FutureWarning)


def throw_derivatives_return_info_future_warning():
    msg = (
        "The `return_info` argument is deprecated and will be removed alongside "
        "Richardson extrapolation in optimagic version 0.6.0."
    )
    warnings.warn(msg, FutureWarning)


def throw_derivatives_return_func_value_future_warning():
    msg = (
        "The `return_func_value` argument is deprecated and will be removed in "
        "optimagic version 0.6.0."
    )
    warnings.warn(msg, FutureWarning)


def throw_numdiff_result_func_evals_future_warning():
    msg = (
        "The `func_evals` attribute is deprecated and will be removed in optimagic "
        "version 0.6.0."
    )
    warnings.warn(msg, FutureWarning)


def throw_numdiff_result_derivative_candidates_future_warning():
    msg = (
        "The `derivative_candidates` attribute is deprecated and will be removed in "
        "optimagic version 0.6.0."
    )
    warnings.warn(msg, FutureWarning)


def throw_numdiff_options_deprecated_in_estimate_ml_future_warning():
    msg = (
        "The argument `numdiff_options` is deprecated for `estimate_ml` and will be "
        "removed in estimagic version 0.6.0. Please use the `jacobian_numdiff_options` "
        "and `hessian_numdiff_options` arguments instead to specify the options for "
        "the first and second numerical derivative estimation."
    )
    warnings.warn(msg, FutureWarning)


def throw_numdiff_options_deprecated_in_estimate_msm_future_warning():
    msg = (
        "The argument `numdiff_options` is deprecated for `estimate_msm` and will be "
        "removed in estimagic version 0.6.0. Please use the `jacobian_numdiff_options` "
        "argument instead."
    )
    warnings.warn(msg, FutureWarning)


def throw_dict_access_future_warning(attribute, obj_name):
    msg = (
        f"The dictionary access for '{attribute}' is deprecated and will be removed "
        "in optimagic version 0.6.0. Please use the new attribute access instead: "
        f"`{obj_name}.{attribute}`."
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
        "additional information for the logging. Please see the documentation for more "
        "details: https://optimagic.readthedocs.io/en/latest/how_to/how_to_criterion_function.html"
    )
    warnings.warn(msg, FutureWarning)


def infer_problem_type_from_dict_output(output):
    if "root_contributions" in output:
        out = AggregationLevel.LEAST_SQUARES
    elif "contributions" in output:
        out = AggregationLevel.LIKELIHOOD
    else:
        out = AggregationLevel.SCALAR
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


def throw_dict_constraints_future_warning_if_required(
    constraints: list[dict[str, Any]] | dict[str, Any],
) -> None:
    replacements = {
        "fixed": "optimagic.FixedConstraint",
        "increasing": "optimagic.IncreasingConstraint",
        "decreasing": "optimagic.DecreasingConstraint",
        "equality": "optimagic.EqualityConstraint",
        "probability": "optimagic.ProbabilityConstraint",
        "pairwise_equality": "optimagic.PairwiseEqualityConstraint",
        "covariance": "optimagic.FlatCovConstraint",
        "sdcorr": "optimagic.FlatSDCorrConstraint",
        "linear": "optimagic.LinearConstraint",
        "nonlinear": "optimagic.NonlinearConstraint",
    }

    if not isinstance(constraints, list):
        constraints = [constraints]

    types = [
        constraint.get("type", None) if isinstance(constraint, dict) else None
        for constraint in constraints
    ]
    types = list(set(types) - {None})

    if types:
        msg = (
            "Specifying constraints as a dictionary is deprecated and will be removed "
            "in optimagic version 0.6.0. Please replace them using the new optimagic "
            "constraint objects:\n"
        )
        for t in types:
            msg += f"  {{'type': '{t}', ...}} -> {replacements[t]}(...)\n"
        msg += (
            "\nFor more details see the documentation: "
            "https://optimagic.readthedocs.io/en/latest/how_to/how_to_constraints.html"
        )

        warnings.warn(msg, FutureWarning)


def replace_and_warn_about_deprecated_multistart_options(options):
    """Replace deprecated multistart options and warn about them.

    Args:
        options (MultistartOptions): The multistart options to replace.

    Returns:
        MultistartOptions: The replaced multistart options.

    """
    replacements = {}

    if options.share_optimization is not None:
        msg = (
            "The `share_optimization` option is deprecated and will be removed in "
            "version 0.6.0. Use `stopping_maxopt` instead to specify the number of "
            "optimizations directly. For more details see the documentation: "
            "https://optimagic.readthedocs.io/en/latest/how_to/how_to_multistart.html"
        )
        warnings.warn(msg, FutureWarning)

    if options.convergence_relative_params_tolerance is not None:
        msg = (
            "The `convergence_relative_params_tolerance` option is deprecated and will "
            "be removed in version 0.6.0. Use `convergence_xtol_rel` instead. For more "
            "details see the documentation: "
            "https://optimagic.readthedocs.io/en/latest/how_to/how_to_multistart.html"
        )
        warnings.warn(msg, FutureWarning)
        if options.convergence_xtol_rel is None:
            replacements["convergence_xtol_rel"] = (
                options.convergence_relative_params_tolerance
            )

    if options.optimization_error_handling is not None:
        msg = (
            "The `optimization_error_handling` option is deprecated and will be "
            "removed in version 0.6.0. Setting this attribute also sets the error "
            "handling for exploration. Use the new `error_handling` option to set the "
            "error handling for both optimization and exploration. For more details "
            "see the documentation: "
            "https://optimagic.readthedocs.io/en/latest/how_to/how_to_multistart.html"
        )
        warnings.warn(msg, FutureWarning)
        if options.error_handling is None:
            replacements["error_handling"] = options.optimization_error_handling

    if options.exploration_error_handling is not None:
        msg = (
            "The `exploration_error_handling` option is deprecated and will be "
            "removed in version 0.6.0. Setting this attribute also sets the error "
            "handling for exploration. Use the new `error_handling` option to set the "
            "error handling for both optimization and exploration. For more details "
            "see the documentation: "
            "https://optimagic.readthedocs.io/en/latest/how_to/how_to_multistart.html"
        )
        warnings.warn(msg, FutureWarning)
        if options.error_handling is None:
            replacements["error_handling"] = options.exploration_error_handling

    return replace(options, **replacements)


def replace_and_warn_about_deprecated_base_steps(
    step_size,
    base_steps,
):
    if base_steps is not None:
        msg = (
            "The `base_steps` argument is deprecated and will be removed alongside "
            "Richardson extrapolation in optimagic version 0.6.0. To specify the "
            "step size use the `step_size` argument instead. For more details see the "
            "documentation: "
            "https://optimagic.readthedocs.io/en/latest/how_to/how_to_step_size.html"
        )
        warnings.warn(msg, FutureWarning)

        if step_size is None:
            step_size = base_steps

    return step_size


def replace_and_warn_about_deprecated_derivatives(candidate, name):
    msg = (
        f"Specifying a dictionary of {name} functions is deprecated and will be "
        "removed in optimagic version 0.6.0. Please specify a single function that "
        "returns the correct derivative for your optimizer or a list of functions that "
        "are decorated with the `mark.scalar`, `mark.likelihood` or "
        "`mark.least_squares` decorators. For more details see the documentation: "
        "https://optimagic.readthedocs.io/en/latest/how_to/how_to_derivatives.html"
    )
    warnings.warn(msg, FutureWarning)

    key_to_marker = {
        "value": mark.scalar,
        "contributions": mark.likelihood,
        "root_contributions": mark.least_squares,
    }

    out = []
    for key, func in candidate.items():
        if key in key_to_marker:
            out.append(key_to_marker[key](func))

    return out


def handle_log_options_throw_deprecated_warning(
    log_options: dict[str, Any], logger: str | Path | LogOptions | None
) -> str | Path | LogOptions | None:
    msg = (
        "Usage of the parameter log_options is deprecated "
        "and will be removed in a future version. "
        "Provide a LogOptions instance for the parameter `logging`, if you need to "
        "configure the logging."
    )
    warnings.warn(msg, FutureWarning)
    logging_is_path_or_string = isinstance(logger, str) or isinstance(logger, Path)
    log_options_is_dict = isinstance(log_options, dict)
    compatible_keys = {"fast_logging", "if_table_exists", "if_database_exists"}
    log_options_is_compatible = set(log_options.keys()).issubset(compatible_keys)

    if logging_is_path_or_string:
        if log_options_is_dict and log_options_is_compatible:
            warnings.warn(
                f"\nUsing {log_options=} to create an instance of SQLiteLogOptions. "
                f"This mechanism will be removed in the future.",
                FutureWarning,
            )
            if "if_table_exists" in log_options:
                warnings.warn(
                    "Found 'if_table_exists' in options dictionary. "
                    "This option is deprecated and setting it has no effect.",
                    FutureWarning,
                )
                log_options = {
                    k: v for k, v in log_options.items() if k != "if_table_exists"
                }
            return SQLiteLogOptions(cast(str | Path, logger), **log_options)
        elif not log_options_is_compatible:
            raise ValueError(
                f"Found string or path for logger argument, but parameter"
                f" {log_options=} is not compatible to {compatible_keys=}."
                f"Explicitly create a Logger instance for configuration."
            )

    return logger


def pre_process_constraints(
    constraints: list[Constraint | dict[str, Any]] | Constraint | dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Convert all ways of specifying constraints to a list of dictionaries.

    For the optimagic release 0.5.0 we only implemented the new constraint API, but have
    not overhauled the internal representation of constraints yet. As a result, we
    convert all ways of specifying constraints, and in particular the new interface, to
    the old format, that is, a list of dictionaries.

    Once we have refactor the internal representation of constraints, we will be able to
    go the other way, and convert all formats to the new one.

    """
    if constraints is None:
        return []

    if isinstance(constraints, dict | Constraint):
        constraints = [constraints]

    if isinstance(constraints, list):
        out = []
        invalid_types: list[type] = []
        for constr in constraints:
            if isinstance(constr, Constraint):
                out.append(constr._to_dict())
            elif isinstance(constr, dict):
                out.append(constr)
            else:
                invalid_types.append(type(constr))

        if invalid_types:
            msg = (
                f"Invalid constraint types: {set(invalid_types)}. Must be a constraint "
                "object imported from `optimagic`."
            )
            raise InvalidConstraintError(msg)

    else:
        msg = (
            f"Invalid constraint type: {type(constraints)}. Must be a constraint "
            "object or list thereof imported from `optimagic`. For more details see "
            "the documentation: "
            "https://optimagic.readthedocs.io/en/latest/how_to/how_to_constraints.html"
        )
        raise InvalidConstraintError(msg)

    return out
