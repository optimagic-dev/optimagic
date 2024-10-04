"""Import common objective functions in several optimagic compatible versions.

All implemented functions accept arbitrary pytrees as parameters. If possible they are
implemented as scalar and least-squares versions.

"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pybaum import tree_just_flatten, tree_unflatten

from optimagic import mark
from optimagic.optimization.fun_value import (
    FunctionValue,
)
from optimagic.parameters.block_trees import matrix_to_block_tree
from optimagic.parameters.tree_registry import get_registry
from optimagic.typing import PyTree

REGISTRY = get_registry(extended=True)


@mark.scalar
def trid_scalar(params: PyTree) -> float:
    """Implement Trid function: https://www.sfu.ca/~ssurjano/trid.html."""
    x = _get_x(params)
    return ((x - 1) ** 2).sum() - (x[1:] * x[:-1]).sum()


@mark.scalar
def trid_gradient(params: PyTree) -> PyTree:
    """Calculate gradient of trid function."""
    x = _get_x(params)
    l1 = np.insert(x, 0, 0)
    l1 = np.delete(l1, [-1])
    l2 = np.append(x, 0)
    l2 = np.delete(l2, [0])
    flat = 2 * (x - 1) - l1 - l2
    return _unflatten_gradient(flat, params)


@mark.scalar
def trid_fun_and_gradient(params: PyTree) -> tuple[float, PyTree]:
    """Implement Trid function and calculate gradient."""
    val = trid_scalar(params)
    grad = trid_gradient(params)
    return val, grad


@mark.scalar
def rhe_scalar(params: PyTree) -> float:
    """Implement Rotated Hyper Ellipsoid function.

    Function description: https://www.sfu.ca/~ssurjano/rothyp.html.

    """
    return (rhe_ls(params) ** 2).sum()


@mark.scalar
def rhe_gradient(params: PyTree) -> PyTree:
    """Calculate gradient of rotated_hyper_ellipsoid function."""
    x = _get_x(params)
    flat = np.arange(2 * len(x), 0, -2) * x
    return _unflatten_gradient(flat, params)


@mark.scalar
def rhe_fun_and_gradient(params: PyTree) -> tuple[float, PyTree]:
    """Implement Rotated Hyper Ellipsoid function and calculate gradient."""
    val = rhe_scalar(params)
    grad = rhe_gradient(params)
    return val, grad


@mark.least_squares
def rhe_ls(params: PyTree) -> NDArray[np.float64]:
    """Compute least-squares version of the Rotated Hyper Ellipsoid function."""
    x = _get_x(params)
    dim = len(params)
    out = np.zeros(dim)
    for i in range(dim):
        out[i] = np.sqrt((x[: i + 1] ** 2).sum())
    return out


@mark.least_squares
def rhe_function_value(params: PyTree) -> FunctionValue:
    """FunctionValue version of Rotated Hyper Ellipsoid function."""
    contribs = rhe_ls(params)
    out = FunctionValue(contribs)
    return out


@mark.scalar
def rosenbrock_scalar(params: PyTree) -> float:
    """Rosenbrock function: https://www.sfu.ca/~ssurjano/rosen.html."""
    return (rosenbrock_ls(params) ** 2).sum()


@mark.scalar
def rosenbrock_gradient(params: PyTree) -> PyTree:
    """Calculate gradient of rosenbrock function."""
    x = _get_x(params)
    l1 = np.delete(x, [-1])
    l1 = np.append(l1, 0)
    l2 = np.insert(x, 0, 0)
    l2 = np.delete(l2, [1])
    l3 = np.insert(x, 0, 0)
    l3 = np.delete(l3, [-1])
    l4 = np.delete(x, [0])
    l4 = np.append(l4, 0)
    l5 = np.full((len(x) - 1), 2)
    l5 = np.append(l5, 0)
    flat = 100 * (4 * (l1**3) + 2 * l2 - 2 * (l3**2) - 4 * (l4 * x)) + 2 * l1 - l5
    return _unflatten_gradient(flat, params)


@mark.scalar
def rosenbrock_fun_and_gradient(params: PyTree) -> tuple[float, PyTree]:
    """Implement rosenbrock function and calculate gradient."""
    return rosenbrock_scalar(params), rosenbrock_gradient(params)


@mark.least_squares
def rosenbrock_ls(params: PyTree) -> NDArray[np.float64]:
    """Least-squares version of the rosenbrock function."""
    x = _get_x(params)
    dim = len(params)
    out = np.zeros(dim)
    for i in range(dim - 1):
        out[i] = np.sqrt(((x[i + 1] - x[i] ** 2) ** 2) * 100 + ((x[i] - 1) ** 2))
    return out


@mark.least_squares
def rosenbrock_function_value(params: PyTree) -> FunctionValue:
    """FunctionValue version of the rosenbrock function."""
    return FunctionValue(rosenbrock_ls(params))


@mark.least_squares
def sos_ls(params: PyTree) -> NDArray[np.float64]:
    """Least-squares version of the sum of squares or sphere function."""
    return _get_x(params)


@mark.least_squares
def sos_ls_with_pd_objects(params: PyTree) -> pd.Series[float]:
    """Least-squares version of the sphere function returning pandas objects."""
    return pd.Series(sos_ls(params))


@mark.scalar
def sos_scalar(params: PyTree) -> float:
    """Sum of squares or sphere function."""
    return (_get_x(params) ** 2).sum()


@mark.scalar
def sos_gradient(params: PyTree) -> PyTree:
    """Calculate the gradient of the sum of squares function."""
    flat = 2 * _get_x(params)
    return _unflatten_gradient(flat, params)


@mark.likelihood
def sos_likelihood(params: PyTree) -> NDArray[np.float64]:
    return _get_x(params) ** 2


@mark.likelihood
def sos_likelihood_jacobian(params: PyTree) -> PyTree:
    """Calculate the likelihood Jacobian of the sum of squares function."""
    x = _get_x(params)
    out_mat = np.diag(2 * x)
    out_tree = matrix_to_block_tree(out_mat, x, params)
    return out_tree


@mark.least_squares
def sos_ls_jacobian(params: PyTree) -> PyTree:
    """Calculate the least-squares Jacobian of the sum of squares function."""
    x = _get_x(params)
    out_mat = np.eye(len(x))
    out_tree = matrix_to_block_tree(out_mat, x, params)
    return out_tree


@mark.scalar
def sos_fun_and_gradient(params: PyTree) -> tuple[float, PyTree]:
    """Calculate sum of squares criterion value and gradient."""
    return sos_scalar(params), sos_gradient(params)


@mark.likelihood
def sos_likelihood_fun_and_jac(
    params: PyTree,
) -> tuple[NDArray[np.float64], PyTree]:
    """Calculate sum of squares criterion value and Jacobian."""
    return sos_likelihood(params), sos_likelihood_jacobian(params)


@mark.least_squares
def sos_ls_fun_and_jac(
    params: PyTree,
) -> tuple[NDArray[np.float64], PyTree]:
    """Calculate sum of squares criterion value and Jacobian."""
    return sos_ls(params), sos_ls_jacobian(params)


sos_derivatives = [sos_gradient, sos_likelihood_jacobian, sos_ls_jacobian]


def _get_x(params: PyTree) -> NDArray[np.float64]:
    if isinstance(params, np.ndarray) and params.ndim == 1:
        x = params.astype(float)
    else:
        registry = get_registry(extended=True)
        x = np.array(tree_just_flatten(params, registry=registry), dtype=np.float64)
    return x


def _unflatten_gradient(flat: NDArray[np.float64], params: PyTree) -> PyTree:
    out = tree_unflatten(params, flat.tolist(), registry=REGISTRY)
    return out
