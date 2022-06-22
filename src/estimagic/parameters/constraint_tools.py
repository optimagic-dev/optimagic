from estimagic.parameters.conversion import get_converter


def count_free_params(params, constraints=None, lower_bounds=None, upper_bounds=None):
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
    _, internal_params = get_converter(
        params=params,
        constraints=constraints,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        func_eval=3,
        primary_key="value",
        scaling=False,
        scaling_options={},
    )

    return int(internal_params.free_mask.sum())


def check_constraints(params, constraints, lower_bounds=None, upper_bounds=None):
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
    get_converter(
        params=params,
        constraints=constraints,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        func_eval=3,
        primary_key="value",
        scaling=False,
        scaling_options={},
    )
