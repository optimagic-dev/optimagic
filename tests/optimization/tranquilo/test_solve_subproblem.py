from collections import namedtuple

import numpy as np
import pytest
from estimagic.optimization.tranquilo.models import ScalarModel
from estimagic.optimization.tranquilo.solve_subproblem import get_subsolver
from numpy.testing import assert_array_almost_equal as aaae

solvers = ["bntr", "bntr_fast", "gqtpar", "gqtpar_fast"]


@pytest.mark.parametrize("solver_name", solvers, ids=solvers)
def test_without_bounds(solver_name):
    """Note: For bntr this only checks that it runs, not that results are correct."""
    linear_terms = np.array([-0.0005429824695352, -0.1032556117176, -0.06816855282091])
    quadratic_terms = np.array(
        [
            [2.05714077e-02, 7.58182390e-01, 9.00050279e-01],
            [7.58182390e-01, 6.25867992e01, 4.20096648e01],
            [9.00050279e-01, 4.20096648e01, 4.03810858e01],
        ]
    )

    expected_x = np.array(
        [
            -0.9994584757179,
            -0.007713730538474,
            0.03198833730482,
        ]
    )

    model = ScalarModel(
        intercept=0, linear_terms=linear_terms, square_terms=quadratic_terms
    )

    trustregion = namedtuple("Trustregion", ["center", "radius"])(
        center=np.zeros(3),
        radius=1,
    )

    solve_subproblem = get_subsolver(solver_name)

    calculated = solve_subproblem(
        model=model,
        trustregion=trustregion,
    )

    if solver_name in ["gqtpar", "gqtpar_fast"]:
        aaae(calculated["x"], expected_x)
