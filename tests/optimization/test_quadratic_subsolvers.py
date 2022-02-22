"""Test various solvers for quadratic trust-region subproblems."""
from functools import partial

import numpy as np
import pytest
from estimagic.optimization.quadratic_subsolvers import (
    solve_trustregion_subproblem,
)
from numpy.testing import assert_array_almost_equal as aaae

linear_terms = np.array([-0.0005429824695352, -0.1032556117176, -0.06816855282091])
square_terms = np.array(
    [
        0.02057140765905,
        0.7581823895471,
        0.9000502790194,
        0.7581823895471,
        62.58679924171,
        42.00966482215,
        0.9000502790194,
        42.00966482215,
        40.38108583657,
    ]
).reshape(3, 3, order="F")
main_model = {"linear_terms": linear_terms, "square_terms": square_terms}


def _evaluate_main_model(
    x,
    linear_terms,
    square_terms,
):
    """Evaluate the criterion and derivative of the main model.

    Args:
        x (np.ndarray): Parameter vector of zeros.
        linear_terms (np.ndarray): Linear terms of the main model of shape (n,).
        square_terms (np.ndarray): Square terms of the main model of shape (n, n).

    Returns:
        Tuple:
        - criterion (float): Criterion value of the main model.
        - derivative (np.ndarray): Derivative of the main model of shape (n,).
    """
    criterion = np.dot(linear_terms, x) + 0.5 * np.dot(np.dot(x, square_terms), x)
    derivative = linear_terms + np.dot(square_terms, x)

    return criterion, derivative


evaluate_main_model = partial(
    _evaluate_main_model,
    **main_model,
)


@pytest.fixture
def expected_inputs():
    p_expected = np.array(
        [
            -0.9994584757179,
            -0.007713730538474,
            0.03198833730482,
        ]
    )
    q_min_expected = -0.001340933981148

    return p_expected, q_min_expected


def test_trustregion_subsolver(expected_inputs):
    p_expected, q_min_expected = expected_inputs

    result = solve_trustregion_subproblem(evaluate_main_model)

    aaae(result["p_solution"], p_expected)
    aaae(result["q_min"], q_min_expected)
