"""Handle pc by reparametrizations."""
import numpy as np

import estimagic.optimization.kernel_transformations as kt


def reparametrize_to_internal(external, internal_free, processed_constraints):
    """Convert a params DataFrame into a numpy array of internal parameters.

    Args:
        processed_params (DataFrame): A processed params DataFrame. See :ref:`params`.
        processed_constraints (list): Processed and consolidated pc.

    Returns:
        internal_params (numpy.ndarray): 1d numpy array of free reparametrized
            parameters.

    """
    internal_values = external.copy()
    for constr in processed_constraints:
        func = getattr(kt, f"{constr['type']}_to_internal")

        internal_values[constr["index"]] = func(external[constr["index"]], constr)

    return internal_values[internal_free]


def reparametrize_from_internal(
    internal, fixed_values, pre_replacements, processed_constraints, post_replacements,
):
    """Convert a numpy array of internal parameters to a params DataFrame.

    Args:
        internal (numpy.ndarray): 1d numpy array with internal parameters
        fixed_values (numpy.ndarray): 1d numpy array with internal fixed values
        pre_replacements (numpy.ndarray): 1d numpy array with positions of internal
            parameters that have to be copied before transformations are applied.
            Negative if no value has to be copied.
        processed_constraints (list): List of processed and consolidated constraint
            dictionaries. Can have the types "linear", "probability", "covariance"
            and "sdcorr".
        post_replacments (numpy.ndarray): 1d numpy array with parameter positions.

    Returns:
        numpy.ndarray: Array with external parameters

    """
    # do pre-replacements
    external_values = pre_replace(internal, fixed_values, pre_replacements)

    # do transformations
    for constr in processed_constraints:
        func = getattr(kt, f"{constr['type']}_from_internal")
        external_values[constr["index"]] = func(
            external_values[constr["index"]], constr
        )

    # do post-replacements
    external_values = post_replace(external_values, post_replacements)

    return external_values


def convert_external_derivative_to_internal(
    external_derivative,
    internal_values,
    fixed_values,
    pre_replacements,
    processed_constraints,
    post_replacements,
):
    """Compute the derivative of the criterion utilizing an external derivative.

    Denote by :math:`c` the criterion function which is evaluated on the full
    parameter set. Denote by :math:`g` the paramater transform which maps an
    internal to an external paramter, i.e :math:`g: x \mapsto g(x)`, with
    :math:`x` denoting the internal paramter vector and :math:`g(x)` the
    respective external parameter frame. We are interested in the derivative of
    the composition :math:`f := c \circ g` which maps an internal vector to the
    criterion value. The derivative can be computed using the chain rule, as

    .. math::
        \frac{\mathrm{d}f}{\mathrm{d}x}(x) =
            \frac{\mathrm{d}c}{\mathrm{d}g}(g(x)) \times
            \frac{\mathrm{d}g}{\mathrm{d}x}(x)

    We assume that the user provides the first part of the above product. The
    second part denotes the derivative of the paramter transform from inner
    to external.

    Args:
        external_derivative (numpy.ndarray): The external derivative evaluated at
            external values mapped from ``internal_values``.
        internal_values (numpy.ndarray): 1d numpy array with internal parameters
        fixed_values (numpy.ndarray): 1d numpy array with internal fixed values
        pre_replacements (numpy.ndarray): 1d numpy array with positions of internal
            parameters that have to be copied before transformations are applied.
            Negative if no value has to be copied.
        processed_constraints (list): List of processed and consolidated constraint
            dictionaries. Can have the types "linear", "probability", "covariance"
            and "sdcorr".
        post_replacments (numpy.ndarray): 1d numpy array with parameter positions.

    Returns:
        deriv (numpy.ndarray): The gradient or Jacobian.

    """
    reparametrize_jacobian = reparametrize_from_internal_jacobian(
        internal_values,
        fixed_values,
        pre_replacements,
        processed_constraints,
        post_replacements,
    )
    deriv = external_derivative @ reparametrize_jacobian
    return deriv


def reparametrize_from_internal_jacobian(
    internal_values,
    fixed_values,
    pre_replacements,
    processed_constraints,
    post_replacements,
):
    """Return derivative of parameter transform from internal to external.

    Returns the Jacobian matrix of the mapping from internal to external
    parameters given a specific internal value. See function
    ``reparametrize_from_internal`` for the mapping specification.

    Args:
        internal_values (numpy.ndarray): 1d numpy array with internal parameters
        fixed_values (numpy.ndarray): 1d numpy array with internal fixed values
        pre_replacements (numpy.ndarray): 1d numpy array with positions of internal
            parameters that have to be copied before transformations are applied.
            Negative if no value has to be copied.
        processed_constraints (list): List of processed and consolidated constraint
            dictionaries. Can have the types "linear", "probability", "covariance"
            and "sdcorr".
        post_replacments (numpy.ndarray): 1d numpy array with parameter positions.

    Returns:
        jacobian (numpy.ndarray): The Jacobian matrix.

    """
    dim_in = len(internal_values)
    dim_out = len(fixed_values)

    # jacobian of pre-replacement
    pre_replace_jac = pre_replace_jacobian(pre_replacements, dim_in, dim_out)
    pre_replaced = pre_replace(internal_values, fixed_values, pre_replacements)

    # jacobian of constraint transformation step
    transform_jac = transformation_jacobian(
        processed_constraints, pre_replaced, dim_out
    )

    # jacobian of post-replacement
    post_replace_jac = post_replace_jacobian(post_replacements, dim_out)

    jacobian = post_replace_jac @ transform_jac @ pre_replace_jac
    return jacobian


def pre_replace(internal_values, fixed_values, pre_replacements):
    """Return pre-replaced parameters.

    Args:
        internal (numpy.ndarray): 1d numpy array with internal parameter.
        fixed_values (numpy.ndarray): 1d numpy array with internal fixed values.
        pre_replacements (numpy.ndarray): 1d numpy array with positions of internal
            parameters that have to be copied before transformations are applied.
            Negative if no value has to be copied.

    Returns:
        pre_replaced (numpy.ndarray): 1d numpy array with pre-replaced params.

    """
    pre_replaced = fixed_values.copy()

    mask = pre_replacements >= 0
    positions = pre_replacements[mask]
    pre_replaced[mask] = internal_values[positions]
    return pre_replaced


def pre_replace_jacobian(pre_replacements, dim_in, dim_out):
    """Return Jacobian of pre-replacement step.

    Args:
        pre_replacements (numpy.ndarray): 1d numpy array with positions of internal
            parameters that have to be copied before transformations are applied.
            Negative if no value has to be copied.
        dim_in (int): Dimension of the internal parameters.
        dim_out (int): Dimensions of the external parameters.

    Returns:
        jacobian (np.ndarray): The jacobian.

    """
    mask = pre_replacements >= 0
    position_in = pre_replacements[mask]
    position_out = np.arange(dim_out)[mask]

    jacobian = np.zeros((dim_out, dim_in))
    jacobian[position_out, position_in] = 1
    return jacobian


def transformation_jacobian(processed_constraints, pre_replaced, dim):
    """Return Jacobian of constraint transformation step.

    The Jacobian of the constraint transformation step is build as a block matrix
    of either identity matrices, in the case when the external parameter equals
    the internal parameter, or, of the Jacobians of the specific kernel transforms,
    in case the external paramater is a transformed version of the internal.

    Args:
        processed_constraints (list): List of processed and consolidated constraint
            dictionaries. Can have the types "linear", "probability", "covariance"
            and "sdcorr".
        pre_replaced (numpy.ndarray): 1d numpy array with pre-replaced params.
        dim (int): The dimension of the external parameters.

    Returns:
        jacobian (numpy.ndarray): The Jacobian.

    """
    jacobian = np.eye(dim)

    for constr in processed_constraints:
        block_indices = constr["index"]
        jacobian_func = getattr(kt, f"{constr['type']}_from_internal_jacobian")
        jac = jacobian_func(pre_replaced[block_indices], constr)
        jacobian[np.ix_(block_indices, block_indices)] = jac

    return jacobian


def post_replace(external_values, post_replacements):
    """Return post-replaed parameters.

    Args:
        external_values (numpy.ndarray): 1d numpy array of external params.
        post_replacments (numpy.ndarray): 1d numpy array with parameter positions.

    Returns:
        post_replaced (numpy.ndarray): 1d numpy array with post-replaced params.
    """
    post_replaced = external_values.copy()

    mask = post_replacements >= 0
    positions = post_replacements[mask]
    post_replaced[mask] = post_replaced[positions]
    return post_replaced


def post_replace_jacobian(post_replacements, dim):
    """Return Jacobian of post-replacement step.

    Args:
        post_replacments (numpy.ndarray): 1d numpy array with parameter positions.
        dim (int): The dimension of the external parameters.

    Returns:
        jacobian (np.ndarray): The Jacobian.

    """
    mask = post_replacements >= 0
    positions_in = post_replacements[mask]
    positions_out = np.arange(dim)[mask]

    jacobian = np.eye(dim)
    jacobian[positions_out, :] *= 0
    jacobian[positions_out, positions_in] = 1
    return jacobian
