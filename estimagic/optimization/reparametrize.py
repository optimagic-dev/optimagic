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
    external_values = fixed_values.copy()

    # do pre-replacements
    mask = pre_replacements >= 0
    positions = pre_replacements[mask]
    external_values[mask] = internal[positions]

    # do transformations
    for constr in processed_constraints:
        func = getattr(kt, f"{constr['type']}_from_internal")
        external_values[constr["index"]] = func(
            external_values[constr["index"]], constr
        )

    # do post-replacements
    mask = post_replacements >= 0
    positions = post_replacements[mask]
    external_values[mask] = external_values[positions]

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
        internal_values (np.ndarray): 1d numpy array with internal parameters.
        external_derivative (np.ndarray): The external derivative evaluated at
            external values mapped from ``internal_values``.

    Returns:
        deriv (np.ndarray): The gradient or jacobian.

    """
    reparametrize_jacobian = _reparametrize_from_internal_jacobian(
        internal_values,
        fixed_values,
        pre_replacements,
        processed_constraints,
        post_replacements,
    )
    deriv = external_derivative @ reparametrize_jacobian
    return deriv


def _reparametrize_from_internal_jacobian(
    internal_values,
    fixed_values,
    pre_replacements,
    processed_constraints,
    post_replacements,
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
    pre_replaced = _pre_replace(internal_values, fixed_values, pre_replacements)
    transformed = _transform_constraint_part(processed_constraints, pre_replaced)
    post_replaced = _post_replace(transformed, post_replacements)
    return post_replaced


def _reparametrize_from_internal_jacobian_matrix(
    internal_values,
    fixed_values,
    pre_replacements,
    processed_constraints,
    post_replacements,
):
    dim_in = len(internal_values)
    dim_out = len(fixed_values)

    pre_replacer = _pre_replace_matrix(pre_replacements, dim_in, dim_out)
    pre_replaced = _pre_replace(internal_values, fixed_values, pre_replacements)
    transformer = _transformer_matrix(processed_constraints, pre_replaced, dim_out)
    post_replacer = _post_replace_matrix(post_replacements, dim_out)

    jacobian = post_replacer @ transformer @ pre_replacer
    return jacobian


def _pre_replace_matrix(pre_replacements, dim_in, dim_out):
    mask = pre_replacements >= 0
    positions = pre_replacements[mask]
    pre_replacer = np.zeros((dim_out, dim_in))
    pre_replacer[np.ix_(positions, positions)] = 1
    return pre_replacer


def _post_replace_matrix(post_replacements, dim):
    mask = post_replacements >= 0
    positions = post_replacements[mask]

    post_replacer = np.eye(dim)
    post_replacer[positions, ] *= 0
    post_replacer[np.ix_(positions, positions)] = 1
    return post_replacer


def _transformer_matrix(processed_constraints, pre_replaced, dim):
    transformer = np.eye(dim)

    for constr in processed_constraints:
        block_indices = constr["index"]
        jacobian = getattr(kt, f"{constr['type']}_from_internal_jacobian")
        jac = jacobian(pre_replaced[block_indices], constr)
        transformer[np.ix_(block_indices, block_indices)] = jac

    return transformer


def _pre_replace(internal_values, fixed_values, pre_replacements):
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
    pre_replaced = np.zeros_like(fixed_values)
    mask = pre_replacements >= 0
    positions = pre_replacements[mask]
    pre_replaced[mask] = internal_values[positions]
    return pre_replaced


def _post_replace(transformed, post_replacements):
    """Return post-replaced parameters.

    Args:
        transformed (numpy.ndarray): 1d numpy array with transformed parameters.
        fixed_values (numpy.ndarray): 1d numpy array with internal fixed values
        post_replacments (numpy.ndarray): 1d numpy array with parameter positions.
    
    Returns:
        post_replaced (numpy.ndarray): 1d numpy array with post_replaced params.
        
    """
    post_replaced = transformed.copy()
    mask = post_replacements >= 0
    positions = post_replacements[mask]
    post_replaced[mask] = post_replaced[positions]
    return post_replaced


def _transform_constraint_part(processed_constraints, pre_replaced):
    """Return constraint transformation derivative matrix.

    Args:
        processed_constraints (list): Processed and consolidated pc.
        pre_replaced (np.ndarrary): 1d array of internal paramaters with
            pre-replacements applied.

    Returns:
        transformed (np.ndarray): 1d arnp.arange(ray of transformed parameters.

    """
    transformed = pre_replaced.copy()

    for constr in processed_constraints:
        indices = constr["index"]
        jacobian = getattr(kt, f"{constr['type']}_from_internal_jacobian")
        jac = jacobian(pre_replaced[indices], constr)
        transformed[indices] = jac @ pre_replaced[indices]

    return transformed
