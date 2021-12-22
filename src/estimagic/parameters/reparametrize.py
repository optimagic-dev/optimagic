"""Handle constraints by reparametrizations.

The functions in this module allow to convert between internal and external parameter
vectors.


An external parameter vector is the parameter vector as it was specified by the user.
This external parameter vector might be subject to constraints, such as the condition
that the first two parameters are equal.


An internal parameter vector is an internal representation of the parameters in a
different space. The internal parameters are meaningless and have no direct
interpretation. However, the internal parameter vector has two important properties:
1. It is only subject to box constraints
2. `reparametrize_from_internal(internal_parameter)` always produces a valid external
parameter vector (i.e. one that fulfills all constraints.

For more background see :ref:`implementation_of_constraints`.

The reparametrization from internal can be broken down into three separate steps:

- Writing values from the internal parameter vector into an array that is as long as the
  external parameters and contains NaNs or values to which parameters have been fixed.
  We call this step `pre_replace`.
- Transforming slices of the resulting vector with kernel transformations. Note that
  this step does not change the length. All kernel transformations have as many input
  as output parameters and are invertible. We call this step `transformation`. The
  resulting vector might still contrain NaNs.
- Fill the NaNs by duplicating values of the transformed parameter vector. We call this
  step `post_replace`

In the following, let n_external be the length of th external parameter vector and
n_internal the length of the internal parameter vector.

"""
import estimagic.parameters.kernel_transformations as kt
import numpy as np
import pandas as pd


def reparametrize_to_internal(
    external,
    internal_free,
    processed_constraints,
    scaling_factor=None,
    scaling_offset=None,
):
    """Convert a params DataFrame into a numpy array of internal parameters.

    Args:
        external (np.ndarray or pandas.DataFrmae): 1d array with of external parameter
            values or params DataFrame.
        internal_free (np.ndarray): 1d array of lenth n_external that determines
            which parameters are free.
        processed_constraints (list): Processed and consolidated constraints. The
            processed constraints contain information on the transformations that have
            to be done.
        scaling_factor (np.ndarray or None): If None, no scaling factor is used.
        scaling_offset (np.ndarray or None): If None, no scaling offset is used.

    Returns:
        internal_params (numpy.ndarray): 1d numpy array of free reparametrized
            parameters.

    """
    if isinstance(external, pd.DataFrame):
        external = external["value"].to_numpy()

    with_internal_values = external.copy()

    for constr in processed_constraints:
        func = getattr(kt, f"{constr['type']}_to_internal")

        with_internal_values[constr["index"]] = func(external[constr["index"]], constr)

    internal = with_internal_values[internal_free]

    scaled = kt.scale_to_internal(
        internal, scaling_factor=scaling_factor, scaling_offset=scaling_offset
    )

    return scaled


def reparametrize_from_internal(
    internal,
    fixed_values,
    pre_replacements,
    processed_constraints,
    post_replacements,
    params,
    return_numpy=True,
    scaling_factor=None,
    scaling_offset=None,
):
    """Convert a numpy array of internal parameters to a params DataFrame.

    Args:
        internal (numpy.ndarray): 1d numpy array with internal parameters
        fixed_values (numpy.ndarray): 1d numpy array of length n_external. It contains
            NaN for parameters that are not fixed and an internal representation of the
            value to which a parameter has been fixed for all others.
        pre_replacements (numpy.ndarray): 1d numpy of length n_external. The i_th
            element in array contains the position of the internal parameter that has to
            be copied to the i_th position of the external parameter vector or -1 if no
            value has to be copied.
        params (pandas.DataFrame): See :ref:`params`.
        return_numpy (bool): If True, a 1d array with flattened parameters is returned.
        processed_constraints (list): List of processed and consolidated constraint
            dictionaries. Can have the types "linear", "probability", "covariance"
            and "sdcorr".
        post_replacements (numpy.ndarray): 1d numpy array of lenth n_external. The i_th
            element contains the position a parameter in the transformed parameter
            vector that has to be copied to duplicated and copied to the i_th position
            of the external parameter vector.
        scaling_factor (np.ndarray or None): If None, no scaling factor is used.
        scaling_offset (np.ndarray or None): If None, no scaling offset is used.

    Returns:
        numpy.ndarray: Array with external parameters

    """
    # undo scaling
    internal = kt.scale_from_internal(
        internal, scaling_factor=scaling_factor, scaling_offset=scaling_offset
    )

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

    if return_numpy:
        out = external_values
    else:
        out = params.copy()
        out["value"] = external_values

    return out


def convert_external_derivative_to_internal(
    external_derivative,
    internal_values,
    fixed_values,
    pre_replacements,
    processed_constraints,
    post_replacements=None,
    pre_replace_jac=None,
    post_replace_jac=None,
    scaling_factor=None,
    scaling_offset=None,
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
        fixed_values (numpy.ndarray): 1d numpy array of length n_external. It contains
            NaN for parameters that are not fixed and an internal representation of the
            value to which a parameter has been fixed for all others.
        pre_replacements (numpy.ndarray): 1d numpy of length n_external. The i_th
            element in array contains the position of the internal parameter that has to
            be copied to the i_th position of the external parameter vector or -1 if no
            value has to be copied.
        processed_constraints (list): List of processed and consolidated constraint
            dictionaries. Can have the types "linear", "probability", "covariance"
            and "sdcorr".
        post_replacements (numpy.ndarray): 1d numpy array of lenth n_external. The i_th
            element contains the position a parameter in the transformed parameter
            vector that has to be copied to duplicated and copied to the i_th position
            of the external parameter vector.
        pre_replace_jac (np.ndarray): 2d Array with the jacobian of pre_replace
        post_replacment_jacobian (np.ndarray): 2d Array with the jacobian post_replace
        scaling_factor (np.ndarray or None): If None, no scaling factor is used.
        scaling_offset (np.ndarray or None): If None, no_scaling_factor is used.


    Returns:
        deriv (numpy.ndarray): The gradient or Jacobian.

    """
    dim_in = len(internal_values)
    internal_unscaled_values = kt.scale_from_internal(
        internal_values,
        scaling_factor=scaling_factor,
        scaling_offset=scaling_offset,
    )

    pre_replaced = pre_replace(internal_unscaled_values, fixed_values, pre_replacements)

    if post_replacements is None and post_replace_jac is None:
        raise ValueError(
            "either post_replacements or post_replace_jac must be specified."
        )

    if pre_replace_jac is None:
        pre_replace_jac = pre_replace_jacobian(pre_replacements, dim_in)

    if post_replace_jac is None:
        post_replace_jac = post_replace_jacobian(post_replacements)

    transform_jac = transformation_jacobian(processed_constraints, pre_replaced)

    external_derivative = np.atleast_2d(external_derivative)
    tall_external = external_derivative.shape[0] > external_derivative.shape[1]

    mat_list = [
        external_derivative,
        post_replace_jac,
        transform_jac,
        pre_replace_jac,
    ]
    if scaling_factor is not None:
        mat_list.append(np.diag(scaling_factor))

    if tall_external:
        deriv = _multiply_from_right(mat_list)
    else:
        deriv = _multiply_from_left(mat_list)

    # return gradient with shape (len(params),)
    if deriv.shape[0] == 1:
        deriv = deriv.flatten()
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
        fixed_values (numpy.ndarray): 1d numpy array of length n_external. It contains
            NaN for parameters that are not fixed and an internal representation of the
            value to which a parameter has been fixed for all others.
        pre_replacements (numpy.ndarray): 1d numpy of length n_external. The i_th
            element in array contains the position of the internal parameter that has to
            be copied to the i_th position of the external parameter vector or -1 if no
            value has to be copied.

    Returns:
        pre_replaced (numpy.ndarray): 1d numpy array with pre-replaced params.


    Examples:

        >>> internal_values = np.array([1., 2.])
        >>> fixed_values = np.array([np.nan, 0, np.nan])
        >>> pre_replacements = np.array([1, -1, 0])
        >>> pre_replace(internal_values, fixed_values, pre_replacements)
        array([2., 0., 1.])

    """
    pre_replaced = fixed_values.copy()

    mask = pre_replacements >= 0
    positions = pre_replacements[mask]
    pre_replaced[mask] = internal_values[positions]
    return pre_replaced


def pre_replace_jacobian(pre_replacements, dim_in):
    """Return Jacobian of pre-replacement step.

    Remark. The function ``pre_replace`` can have ``np.nan`` in its output. In
    this case we know from the underlying structure that the derivative of this
    output with respect to any of the inputs is zero. Here we use this additional
    knowledge; however, when the derivative is computed using a numerical
    differentiation technique this will not be the case. Thus the numerical
    derivative can differ from the derivative here in these cases.

    Args:
        pre_replacements (numpy.ndarray): 1d numpy of length n_external. The i_th
            element in array contains the position of the internal parameter that has to
            be copied to the i_th position of the external parameter vector or -1 if no
            value has to be copied.
        dim_in (int): Dimension of the internal parameters.

    Returns:
        jacobian (np.ndarray): The jacobian.

    Examples:
        >>> # Note: The example is the same as in the doctest of pre_replace
        >>> pre_replacements = np.array([1, -1, 0])
        >>> pre_replace_jacobian(pre_replacements, 2)
        array([[0., 1.],
               [0., 0.],
               [1., 0.]])

    """
    dim_out = len(pre_replacements)
    mask = pre_replacements >= 0
    position_in = pre_replacements[mask]
    position_out = np.arange(dim_out)[mask]

    jacobian = np.zeros((dim_out, dim_in))
    jacobian[position_out, position_in] = 1
    return jacobian


def transformation_jacobian(processed_constraints, pre_replaced):
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
    dim = len(pre_replaced)
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
        post_replacements (numpy.ndarray): 1d numpy array of lenth n_external. The i_th
            element contains the position a parameter in the transformed parameter
            vector that has to be copied to duplicated and copied to the i_th position
            of the external parameter vector.

    Returns:
        post_replaced (numpy.ndarray): 1d numpy array with post-replaced params.

    Examples:
        >>> external_values = np.array([3., 4., np.nan])
        >>> post_replacements = np.array([-1, -1, 1])
        >>> post_replace(external_values, post_replacements)
        array([3., 4., 4.])

    """
    post_replaced = external_values.copy()

    mask = post_replacements >= 0
    positions = post_replacements[mask]
    post_replaced[mask] = post_replaced[positions]
    return post_replaced


def post_replace_jacobian(post_replacements):
    """Return Jacobian of post-replacement step.

    Args:
        post_replacements (numpy.ndarray): 1d numpy array of lenth n_external. The i_th
            element contains the position a parameter in the transformed parameter
            vector that has to be copied to duplicated and copied to the i_th position
            of the external parameter vector.
        dim (int): The dimension of the external parameters.

    Returns:
        jacobian (np.ndarray): The Jacobian.

    Examples:
        >>> # Note: the example is the same as in the doctest of post_replace
        >>> post_replacements = np.array([-1, -1, 1])
        >>> post_replace_jacobian(post_replacements)
        array([[1., 0., 0.],
               [0., 1., 0.],
               [0., 1., 0.]])



    """
    dim = len(post_replacements)
    mask = post_replacements >= 0
    positions_in = post_replacements[mask]
    positions_out = np.arange(dim)[mask]

    jacobian = np.eye(dim)
    jacobian[positions_out, :] *= 0
    jacobian[positions_out, positions_in] = 1
    return jacobian
