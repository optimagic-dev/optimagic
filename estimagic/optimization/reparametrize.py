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
    post_replacements=None,
    pre_replace_jac=None,
    post_replace_jac=None,
):
    r"""Compute the derivative of the criterion utilizing an external derivative.

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
        pre_replace_jac (np.ndarray): 2d Array with the jacobian of the
            pre-replacements.
        post_replacment_jacobian (np.ndarray): 2d Array with the jacobian of the
            post replacements.

    Returns:
        deriv (numpy.ndarray): The gradient or Jacobian.

    """
    dim_in = len(internal_values)
    dim_out = len(fixed_values)

    pre_replaced = pre_replace(internal_values, fixed_values, pre_replacements)

    if post_replacements is None and post_replace_jac is None:
        raise ValueError(
            "either post_replacements or post_replace_jac must be specified."
        )

    if pre_replace_jac is None:
        pre_replace_jac = pre_replace_jacobian(pre_replacements, dim_in, dim_out)

    if post_replace_jac is None:
        post_replace_jac = post_replace_jacobian(post_replacements, dim_out)

    transform_jac = transformation_jacobian(
        processed_constraints, pre_replaced, dim_out
    )

    if len(external_derivative.shape) == 1:
        external_derivative = external_derivative.reshape(1, -1)

    tall_external = external_derivative.shape[0] > external_derivative.shape[1]

    mat_list = [
        external_derivative,
        post_replace_jac,
        transform_jac,
        pre_replace_jac,
    ]

    if tall_external:
        deriv = _multiply_from_right(mat_list)
    else:
        deriv = _multiply_from_left(mat_list)

    return deriv


def _multiply_from_left(mat_list):
    """Multiply all matrices in the list, starting from the left.

    Note that this only affects the order in which the pairwise multiplications happen,
    not the actual result.

    """
    out = mat_list[0]
    for mat in mat_list[1:]:
        out = out @ mat
    return out


def _multiply_from_right(mat_list):
    """Multiply all matrices in the list, starting from the right.

    Note that this only affects the order in which the pairwise multiplications happen,
    not the actual result.

    """
    out = mat_list[-1]
    for mat in reversed(mat_list[:-1]):
        out = mat @ out
    return out


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

    Remark. The function ``pre_replace`` can have ``np.nan`` in its output. In
    this case we know from the underlying structure that the derivative of this
    output with respect to any of the inputs is zero. Here we use this additional
    knowledge; however, when the derivative is computed using a numerical
    differentiation technique this will not be the case. Thus the numerical
    derivative can differ from the derivative here in these cases.

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
