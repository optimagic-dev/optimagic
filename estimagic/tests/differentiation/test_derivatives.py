from functools import partial
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal as aaae
from pandas.testing import assert_frame_equal
from scipy.optimize._numdiff import approx_derivative

from estimagic.differentiation.derivatives import _consolidate_one_step_derivatives
from estimagic.differentiation.derivatives import _convert_evaluation_data_to_tidy_frame
from estimagic.differentiation.derivatives import _nan_skipping_batch_evaluator
from estimagic.differentiation.derivatives import first_derivative
from estimagic.examples.numdiff_example_functions_np import logit_loglike
from estimagic.examples.numdiff_example_functions_np import logit_loglike_gradient
from estimagic.examples.numdiff_example_functions_np import logit_loglikeobs
from estimagic.examples.numdiff_example_functions_np import logit_loglikeobs_jacobian
from estimagic.optimization.utilities import namedtuple_from_kwargs


@pytest.fixture
def binary_choice_inputs():
    fix_path = Path(__file__).resolve().parent / "binary_choice_inputs.pickle"
    inputs = pd.read_pickle(fix_path)
    return inputs


methods = ["forward", "backward", "central"]


@pytest.mark.parametrize("method", methods)
def test_first_derivative_jacobian(binary_choice_inputs, method):
    fix = binary_choice_inputs
    func = partial(logit_loglikeobs, y=fix["y"], x=fix["x"])

    calculated = first_derivative(
        func=func,
        method=method,
        params=fix["params_np"],
        n_steps=1,
        base_steps=None,
        lower_bounds=np.full(fix["params_np"].shape, -np.inf),
        upper_bounds=np.full(fix["params_np"].shape, np.inf),
        min_steps=1e-8,
        step_ratio=2.0,
        f0=func(fix["params_np"]),
        n_cores=1,
    )

    expected = logit_loglikeobs_jacobian(fix["params_np"], fix["y"], fix["x"])

    aaae(calculated, expected, decimal=6)


def test_first_derivative_jacobian_works_at_defaults(binary_choice_inputs):
    fix = binary_choice_inputs
    func = partial(logit_loglikeobs, y=fix["y"], x=fix["x"])
    calculated = first_derivative(func=func, params=fix["params_np"], n_cores=1)
    expected = logit_loglikeobs_jacobian(fix["params_np"], fix["y"], fix["x"])
    aaae(calculated, expected, decimal=6)


@pytest.mark.parametrize("method", methods)
def test_first_derivative_gradient(binary_choice_inputs, method):
    fix = binary_choice_inputs
    func = partial(logit_loglike, y=fix["y"], x=fix["x"])

    calculated = first_derivative(
        func=func,
        method=method,
        params=fix["params_np"],
        n_steps=1,
        f0=func(fix["params_np"]),
        n_cores=1,
    )

    expected = logit_loglike_gradient(fix["params_np"], fix["y"], fix["x"])

    aaae(calculated, expected, decimal=4)


@pytest.mark.parametrize("method", methods)
def test_first_derivative_scalar(method):
    def f(x):
        return x ** 2

    calculated = first_derivative(f, 3.0, n_cores=1)
    expected = 6.0
    assert calculated == expected


@pytest.mark.parametrize("method", methods)
def test_first_derivative_scalar_with_return_func_value(method):
    def f(x):
        return x ** 2

    calculated = first_derivative(f, 3.0, return_func_value=True, n_cores=1)
    expected = (6.0, {"func_value": 9.0})
    assert calculated == expected


def test_nan_skipping_batch_evaluator():
    arglist = [np.nan, np.ones(2), np.array([3, 4]), np.nan, np.array([1, 2])]
    expected = [
        np.full(2, np.nan),
        np.ones(2),
        np.array([9, 16]),
        np.full(2, np.nan),
        np.array([1, 4]),
    ]
    calculated = _nan_skipping_batch_evaluator(
        func=lambda x: x ** 2,
        arguments=arglist,
        n_cores=1,
        error_handling="continue",
        batch_evaluator="joblib",
    )
    for arr_calc, arr_exp in zip(calculated, expected):
        if np.isnan(arr_exp).all():
            assert np.isnan(arr_calc).all()
        else:
            aaae(arr_calc, arr_exp)


def test_consolidate_one_step_derivatives():
    forward = np.ones((1, 4, 3))
    forward[:, :, 0] = np.nan
    backward = np.zeros_like(forward)

    calculated = _consolidate_one_step_derivatives(
        {"forward": forward, "backward": backward}, ["forward", "backward"]
    )
    expected = np.array([[0, 1, 1]] * 4)
    aaae(calculated, expected)


@pytest.fixture()
def example_function_gradient_fixtures():
    def f(x):
        """f:R^3 -> R"""
        x1, x2, x3 = x[0], x[1], x[2]
        y1 = np.sin(x1) + np.cos(x2) + x3 - x3
        return y1

    def fprime(x):
        """Gradient(f)(x):R^3 -> R^3"""
        x1, x2, x3 = x[0], x[1], x[2]
        grad = np.array([np.cos(x1), -np.sin(x2), x3 - x3])
        return grad

    return {"func": f, "func_prime": fprime}


@pytest.fixture()
def example_function_jacobian_fixtures():
    def f(x):
        """f:R^3 -> R^2"""
        x1, x2, x3 = x[0], x[1], x[2]
        y1, y2 = np.sin(x1) + np.cos(x2), np.exp(x3)
        return np.array([y1, y2])

    def fprime(x):
        """Jacobian(f)(x):R^3 -> R^(2x3)"""
        x1, x2, x3 = x[0], x[1], x[2]
        jac = np.array([[np.cos(x1), -np.sin(x2), 0], [0, 0, np.exp(x3)]])
        return jac

    return {"func": f, "func_prime": fprime}


def test_first_derivative_gradient_richardson(example_function_gradient_fixtures):
    f = example_function_gradient_fixtures["func"]
    fprime = example_function_gradient_fixtures["func_prime"]

    true_fprime = fprime(np.ones(3))
    scipy_fprime = approx_derivative(f, np.ones(3))
    our_fprime = first_derivative(f, np.ones(3), n_steps=3, method="central", n_cores=1)

    aaae(scipy_fprime, our_fprime)
    aaae(true_fprime, our_fprime)


def test_first_derivative_jacobian_richardson(example_function_jacobian_fixtures):
    f = example_function_jacobian_fixtures["func"]
    fprime = example_function_jacobian_fixtures["func_prime"]

    true_fprime = fprime(np.ones(3))
    scipy_fprime = approx_derivative(f, np.ones(3))
    our_fprime = first_derivative(f, np.ones(3), n_steps=3, method="central", n_cores=1)

    aaae(scipy_fprime, our_fprime)
    aaae(true_fprime, our_fprime)


def test_convert_evaluation_data_to_tidy_frame():
    arr = np.arange(4).reshape(2, 2)
    arr2 = arr.reshape(2, 1, 2)
    steps = namedtuple_from_kwargs(pos=arr, neg=-arr)
    evals = namedtuple_from_kwargs(pos=arr2, neg=-arr2)
    expected = """sign,step_number,dim_x,dim_f,step,eval
    1,0,0,0,0,0
    1,0,1,0,1,1
    1,1,0,0,2,2
    1,1,1,0,3,3
    -1,0,0,0,0,0
    -1,0,1,0,1,-1
    -1,1,0,0,2,-2
    -1,1,1,0,3,-3
    """
    expected = pd.read_csv(StringIO(expected))
    got = _convert_evaluation_data_to_tidy_frame(steps, evals)
    assert_frame_equal(expected, got.reset_index(), check_dtype=False)
