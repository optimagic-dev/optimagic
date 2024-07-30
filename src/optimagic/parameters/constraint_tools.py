from optimagic.parameters.conversion import get_converter
from optimagic.deprecations import replace_and_warn_about_deprecated_bounds
from optimagic.parameters.bounds import pre_process_bounds


def count_free_params(
    params,
    constraints=None,
    bounds=None,
    # deprecated
    lower_bounds=None,
    upper_bounds=None,
):
    """Count the (free) parameters of an optimization problem.

    Args:
        params (pytree): The parameters.
        constraints (list): The constraints for the optimization problem. If constraints
            are provided, only the free parameters are counted.
        lower_bounds (pytree): Lower bounds for params.
        upper_bounds (pytree): Upper bounds for params.

    Returns:
        int: Number of (free) parameters

    """
    bounds = replace_and_warn_about_deprecated_bounds(
        bounds=bounds,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )

    bounds = pre_process_bounds(bounds)

    _, internal_params = get_converter(
        params=params,
        constraints=constraints,
        bounds=bounds,
        func_eval=3,
        primary_key="value",
        scaling=False,
        scaling_options={},
    )

    return int(internal_params.free_mask.sum())


def check_constraints(
    params,
    constraints,
    bounds=None,
    # deprecated
    lower_bounds=None,
    upper_bounds=None,
):
    """Raise an error if constraints are invalid or not satisfied in params.

    Args:
        params (pytree): The parameters.
        constraints (list): The constraints for the optimization problem.
        lower_bounds (pytree): Lower bounds for params.
        upper_bounds (pytree): Upper bounds for params.


    Raises:
        InvalidParamsError: If constraints are valid but not satisfied.
        InvalidConstraintError: If constraints are invalid.

    """
    bounds = replace_and_warn_about_deprecated_bounds(
        bounds=bounds,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )

    bounds = pre_process_bounds(bounds)

    get_converter(
        params=params,
        constraints=constraints,
        bounds=bounds,
        func_eval=3,
        primary_key="value",
        scaling=False,
        scaling_options={},
    )
