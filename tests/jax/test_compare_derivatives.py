"""Compare first and second derivative behavior to that of jax.

This test module only runs if jax is installed.

"""
import numpy as np
import pytest
from estimagic.config import IS_JAX_INSTALLED
from estimagic.differentiation.derivatives import first_derivative
from estimagic.differentiation.derivatives import second_derivative
from numpy.testing import assert_array_almost_equal as aaae
from pybaum import tree_equal
from pybaum import tree_map

if not IS_JAX_INSTALLED:
    pytestmark = pytest.mark.skip(reason="jax is not installed.")
else:
    import jax
    import jax.numpy as jnp


TOLERANCE = 1.5 * 10e-7


def hessian(f):
    return jax.jacfwd(jax.jacrev(f))


def _convert_leaves_to_numpy(tree):
    tree = tree_map(lambda x: np.array(x), tree)
    return tree


def _tree_equal_numpy_leaves(tree1, tree2):
    equality_checkers = {np.ndarray: lambda x, y: aaae(x, y, decimal=TOLERANCE)}
    tree_equal(tree1, tree2, equality_checkers=equality_checkers)


def _compute_testable_estimagic_and_jax_derivatives(func, params, func_jax=None):
    """

    Computes first and second derivative using estimagic and jax. Then converts leaves
    of jax output to numpy so that we can use numpy.testing. For higher dimensional
    output we need to define two function, one with numpy array output and one with
    jax.numpy array output.

    """
    func_jax = func if func_jax is None else func_jax

    estimagic_jac = first_derivative(func, params)["derivative"]
    jax_jac = jax.jacobian(func_jax)(params)
    jax_jac = _convert_leaves_to_numpy(jax_jac)

    estimagic_hess = second_derivative(func, params)["derivative"]
    jax_hess = hessian(func_jax)(params)
    jax_hess = _convert_leaves_to_numpy(jax_hess)

    out = {
        "jac": {"est": estimagic_jac, "jax": jax_jac},
        "hess": {"est": estimagic_hess, "jax": jax_hess},
    }
    return out


def test_scalar_input_scalar_output():
    def func(params):
        return params**2

    params = 1.0

    result = _compute_testable_estimagic_and_jax_derivatives(func, params)
    _tree_equal_numpy_leaves(result["jac"]["est"], result["jac"]["jax"])
    _tree_equal_numpy_leaves(result["hess"]["est"], result["hess"]["jax"])


def test_array_input_scalar_output():
    def func(params):
        return params @ params

    params = np.array([1.0, 2, 3])

    result = _compute_testable_estimagic_and_jax_derivatives(func, params)
    _tree_equal_numpy_leaves(result["jac"]["est"], result["jac"]["jax"])
    _tree_equal_numpy_leaves(result["hess"]["est"], result["hess"]["jax"])


def test_dict_input_scalar_output():
    def func(params):
        return params["a"] * params["b"]

    params = {"a": 1.0, "b": 2.0}

    result = _compute_testable_estimagic_and_jax_derivatives(func, params)
    _tree_equal_numpy_leaves(result["jac"]["est"], result["jac"]["jax"])
    _tree_equal_numpy_leaves(result["hess"]["est"], result["hess"]["jax"])


def test_array_dict_input_scalar_output():
    def func(params):
        return params["a"].sum() * params["b"].prod()

    params = {
        "a": np.array([1.0, 2, 3]),
        "b": np.arange(9, dtype=np.float64).reshape(3, 3),
    }

    result = _compute_testable_estimagic_and_jax_derivatives(func, params)
    _tree_equal_numpy_leaves(result["jac"]["est"], result["jac"]["jax"])
    _tree_equal_numpy_leaves(result["hess"]["est"], result["hess"]["jax"])


def test_array_input_array_output():
    def func(params):
        return np.array([params.sum(), params.prod()])

    def func_jax(params):
        return jnp.array([params.sum(), params.prod()])

    params = np.array([1.0, 2, 3])

    result = _compute_testable_estimagic_and_jax_derivatives(func, params, func_jax)
    _tree_equal_numpy_leaves(result["jac"]["est"], result["jac"]["jax"])

    # for the batch hessian case we return a different object than jax at the moment
    res = np.array(list(result["hess"]["est"].values()))
    _tree_equal_numpy_leaves(res, result["hess"]["jax"])


def test_array_dict_input_array_array_output():
    """

    Jax handles functions with multi-dimensional output different than estimagic. They
    create a block matrix in which the leaves are the 3-dimensional hessian. In
    estimagic we create a named dictionary in which each entry corresponds to one
    output dimension and the value is the standard block-matrix representation of the
    hessian. Interestingly jax only does this if the output contains an array, but not
    if its a pytree that collapes to an array.

    """

    def func(params):
        return params["b"] * np.array([params["a"].sum(), params["a"].prod()])

    def func_jax(params):
        return params["b"] * jnp.array([params["a"].sum(), params["a"].prod()])

    params = {"a": np.array([1.0, 2, 3]), "b": 2.0}

    result = _compute_testable_estimagic_and_jax_derivatives(func, params, func_jax)
    _tree_equal_numpy_leaves(result["jac"]["est"], result["jac"]["jax"])
    # comparison of hessians remains to be implemented


def test_array_dict_input_array_dict_output():
    def func(params):
        value = params["b"] * np.array([params["a"].sum(), params["a"].prod()])
        return [value[0], {"c": 0.0, "d": value[1]}]

    def func_jax(params):
        value = params["b"] * jnp.array([params["a"].sum(), params["a"].prod()])
        return [value[0], {"c": 0.0, "d": value[1]}]

    params = {"a": np.array([1.0, 2, 3]), "b": 2.0}

    result = _compute_testable_estimagic_and_jax_derivatives(func, params, func_jax)
    _tree_equal_numpy_leaves(result["jac"]["est"], result["jac"]["jax"])
    _tree_equal_numpy_leaves(result["hess"]["est"], result["hess"]["jax"])
