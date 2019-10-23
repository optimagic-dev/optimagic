"""Handle constraints by reparametrizations."""
from numba import jit

import estimagic.optimization.kernel_transformations as kt


def reparametrize_to_internal(processed_params, processed_constraints):
    """Convert a params DataFrame into a numpy array of internal parameters.


    Args:
        processed_params (DataFrame): A processed params DataFrame. See :ref:`params`.
        processed_constraints (list): Processed and consolidated constraints.


    Returns:
        internal_params (np.ndarray): 1d numpy array of free reparametrized parameters.

    """
    pp = processed_params.copy()
    internal_values = pp["value"].to_numpy()
    for constr in processed_constraints:
        func = getattr(kt, f"{constr['type']}_to_internal")
        index = constr["index"]
        internal_values[index] = func(internal_values[index], constr)

    return internal_values[pp["_internal_free"]]


def reparametrize_from_internal(
    internal,
    fixed_values,
    pre_replacements,
    processed_constraints,
    post_replacements,
    processed_params,
):
    """Convert a numpy array of internal parameters to a params DataFrame.

    Args:
        internal (np.ndarray): 1d numpy array with internal parameters
        fixed_values (np.ndarray): 1d numpy array with internal fixed values
        pre_replacements (np.ndarray): 1d numpy array with positions of internal
            parameters that have to be copied before transformations are applied.
            Negative if no value has to be copied.
        processed_constraints (list): List of processed and consolidated constraint
            dictionaries. Can have the types "linear", "probability", "covariance"
            and "sdcorr".
        post_replacments (np.ndarray): 1d numpy array with parameter positions.
        processed_params (pd.DataFrame): See :ref:`params`

    Returns:
        updated_params (pd.DataFrame): Copy of processed_params with replaced values.

    """
    external_values = fixed_values.copy()
    external_values = _do_pre_replacements(internal, pre_replacements, external_values)
    for constr in processed_constraints:
        func = getattr(kt, f"{constr['type']}_from_internal")
        index = constr["index"]
        external_values[index] = func(external_values[index], constr)
    external_values = _do_post_replacements(post_replacements, external_values)

    external = processed_params.copy()
    external["value"] = external_values
    return external


@jit
def _do_pre_replacements(internal, pre_replacements, container):
    for external_pos, internal_pos in enumerate(pre_replacements):
        if internal_pos >= 0:
            container[external_pos] = internal[internal_pos]
    return container


@jit
def _do_post_replacements(post_replacements, container):
    for i, pos in enumerate(post_replacements):
        if pos >= 0:
            container[i] = container[pos]
    return container
