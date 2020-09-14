import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from estimagic.optimization.estimagic_optimizers import _estimagic_bhhh


def sum_of_squares(x):
    return x ** 2


def sum_of_squares_jac(x):
    return np.diag(2 * x)


@pytest.mark.xfail(reason="linalg error when start values contain 0")
def test_linalg_erro_with_zero_in_start_values():
    res = _estimagic_bhhh(
        func=sum_of_squares, x0=np.arange(3), jacobian=sum_of_squares_jac,
    )
    aaae(res["solution_x"], np.zeros(3))


def test_extremely_simple_case_closed_form_jac():
    res = _estimagic_bhhh(
        func=sum_of_squares, x0=np.ones(2), jacobian=sum_of_squares_jac,
    )
    aaae(res["solution_x"], np.zeros(2))


def test_extremely_simple_case_num_jac():
    res = _estimagic_bhhh(func=sum_of_squares, x0=np.ones(2),)
    aaae(res["solution_x"], np.zeros(2))


@pytest.mark.xfail(reason="bad start values.")
def test_more_difficult_closed_form_jac():
    res = _estimagic_bhhh(
        func=sum_of_squares, x0=np.arange(5) + 1, jacobian=sum_of_squares_jac,
    )
    aaae(res["solution_x"], np.zeros(5))


@pytest.mark.xfail(reason="bad start values.")
def test_more_difficult_num_jac():
    res = _estimagic_bhhh(func=sum_of_squares, x0=np.arange(5) + 1,)
    aaae(res["solution_x"], np.zeros(5))
