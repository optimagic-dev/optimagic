"""High level functions to convert between parameter vectors.

High level means:

- Functions are save to use directly with user input (if applicable)
- Defaults are filled automatically (if applicable)
- Robust checks and error handling

"""
import functools

from estimagic.parameters.parameter_preprocessing import add_default_bounds_to_params
from estimagic.parameters.parameter_preprocessing import check_params_are_valid
from estimagic.parameters.process_constraints import process_constraints
from estimagic.parameters.reparametrize import convert_external_derivative_to_internal
from estimagic.parameters.reparametrize import post_replace_jacobian
from estimagic.parameters.reparametrize import pre_replace_jacobian
from estimagic.parameters.reparametrize import reparametrize_from_internal
from estimagic.parameters.reparametrize import reparametrize_to_internal


def get_reparametrize_functions(
    params,
    constraints,
    scaling_factor=None,
    scaling_offset=None,
    processed_params=None,
    processed_constraints=None,
):
    """Construct functions to map between internal and external parameters.

    All required information is partialed into the functions.

    Args:
        params (pandas.DataFrame): See :ref:`params`.
        constraints (list): List of constraint dictionaries.
        scaling_factor (np.ndarray or None): If None, no scaling factor is used.
        scaling_offset (np.ndarray or None): If None, no scaling offset is used.
        processed_params (pandas.DataFrame): Processed parameters.
        processed_constraints (list): Processed constraints.

    Returns:
        func: Function that maps an external parameter vector to an internal one
        func: Function that maps an internal parameter vector to an external one

    """
    if processed_params is None or processed_constraints is None:
        params = add_default_bounds_to_params(params)
        check_params_are_valid(params)

        processed_constraints, processed_params = process_constraints(
            constraints=constraints,
            params=params,
            scaling_factor=scaling_factor,
            scaling_offset=scaling_offset,
        )

    # get partialed reparametrize from internal
    pre_replacements = processed_params["_pre_replacements"].to_numpy()
    post_replacements = processed_params["_post_replacements"].to_numpy()
    fixed_values = processed_params["_internal_fixed_value"].to_numpy()

    # get partialed reparametrize to internal
    internal_free = processed_params["_internal_free"].to_numpy()

    partialed_to_internal = functools.partial(
        reparametrize_to_internal,
        internal_free=internal_free,
        processed_constraints=processed_constraints,
        scaling_factor=scaling_factor,
        scaling_offset=scaling_offset,
    )

    partialed_from_internal = functools.partial(
        reparametrize_from_internal,
        fixed_values=fixed_values,
        pre_replacements=pre_replacements,
        processed_constraints=processed_constraints,
        post_replacements=post_replacements,
        params=params,
        scaling_factor=scaling_factor,
        scaling_offset=scaling_offset,
    )

    return partialed_to_internal, partialed_from_internal


def get_derivative_conversion_function(
    params,
    constraints,
    scaling_factor=None,
    scaling_offset=None,
    processed_params=None,
    processed_constraints=None,
):
    """Construct functions to map between internal and external derivatives.

    All required information is partialed into the functions.

    Args:
        params (pandas.DataFrame): See :ref:`params`.
        constraints (list): List of constraint dictionaries.
        scaling_factor (np.ndarray or None): If None, no scaling factor is used.
        scaling_offset (np.ndarray or None): If None, no scaling offset is used.
        processed_params (pandas.DataFrame): Processed parameters.
        processed_constraints (list): Processed constraints.


    Returns:
        func: Function that converts an external derivative to an internal one

    """
    if processed_params is None or processed_constraints is None:
        params = add_default_bounds_to_params(params)
        check_params_are_valid(params)
        processed_constraints, processed_params = process_constraints(
            constraints=constraints,
            params=params,
            scaling_factor=scaling_factor,
            scaling_offset=scaling_offset,
        )

    pre_replacements = processed_params["_pre_replacements"].to_numpy()
    post_replacements = processed_params["_post_replacements"].to_numpy()
    fixed_values = processed_params["_internal_fixed_value"].to_numpy()

    dim_internal = int(processed_params["_internal_free"].sum())

    pre_replace_jac = pre_replace_jacobian(
        pre_replacements=pre_replacements, dim_in=dim_internal
    )
    post_replace_jac = post_replace_jacobian(post_replacements=post_replacements)

    convert_derivative = functools.partial(
        convert_external_derivative_to_internal,
        fixed_values=fixed_values,
        pre_replacements=pre_replacements,
        processed_constraints=processed_constraints,
        pre_replace_jac=pre_replace_jac,
        post_replace_jac=post_replace_jac,
        scaling_factor=scaling_factor,
        scaling_offset=scaling_offset,
    )

    return convert_derivative


def get_internal_bounds(params, constraints, scaling_factor=None, scaling_offset=None):
    params = add_default_bounds_to_params(params)
    check_params_are_valid(params)

    _, processed_params = process_constraints(
        constraints=constraints,
        params=params,
        scaling_factor=scaling_factor,
        scaling_offset=scaling_offset,
    )

    free = processed_params.query("_internal_free")
    lower_bounds = free["_internal_lower"].to_numpy()
    upper_bounds = free["_internal_upper"].to_numpy()
    return lower_bounds, upper_bounds
