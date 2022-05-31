from estimagic.parameters.conversion import get_converter


def count_free_params(params, constraints=None):
    """Count the (free) parameters of an optimization problem.

    Args:
        params (pytree): The parameters.
        constraints (list): The constraints for the optimization problem. If constraints
            are provided, only the free parameters are counted.

    Returns:
        int: Number of (free) parameters

    """
    _, flat_params = get_converter(
        func=None,
        params=params,
        constraints=constraints,
        lower_bounds=None,
        upper_bounds=None,
        func_eval=3,
        primary_key="value",
        scaling=False,
        scaling_options={},
    )

    return int(flat_params.free_mask.sum())


def check_constraints(params, constraints):
    """Raise an error if constraints are invalid or not satisfied in params.

    Args:
        params (pytree): The parameters.
        constraints (list): The constraints for the optimization problem.

    Raises:
        InvalidParamsError: If constraints are valid but not satisfied.
        InvalidConstraintError: If constraints are invalid.

    """
    get_converter(
        func=None,
        params=params,
        constraints=constraints,
        lower_bounds=None,
        upper_bounds=None,
        func_eval=3,
        primary_key="value",
        scaling=False,
        scaling_options={},
    )
