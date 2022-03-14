"""Test various solvers for quadratic trust-region subproblems."""
import numpy as np
import pytest
from estimagic.optimization.bounded_newton_trustregion import minimize_trust_bntr
from estimagic.optimization.quadratic_subsolvers import (
    solve_trustregion_subproblem,
)
from estimagic.optimization.trustregion_conjugate_gradient import minimize_trust_cg
from numpy.testing import assert_array_almost_equal as aaae


@pytest.mark.parametrize(
    "linear_terms, square_terms, x_expected, q_min_expected",
    [
        (
            np.array([-0.0005429824695352, -0.1032556117176, -0.06816855282091]),
            np.array(
                [
                    [2.05714077e-02, 7.58182390e-01, 9.00050279e-01],
                    [7.58182390e-01, 6.25867992e01, 4.20096648e01],
                    [9.00050279e-01, 4.20096648e01, 4.03810858e01],
                ]
            ),
            np.array(
                [
                    -0.9994584757179,
                    -0.007713730538474,
                    0.03198833730482,
                ]
            ),
            -0.001340933981148,
        )
    ],
)
def test_trustregion_subsolver(linear_terms, square_terms, x_expected, q_min_expected):
    main_model = {"linear_terms": linear_terms, "square_terms": square_terms}

    result = solve_trustregion_subproblem(main_model)

    aaae(result["x_solution"], x_expected)
    aaae(result["q_min"], q_min_expected)


@pytest.mark.parametrize(
    "x, model_gradient, model_hessian, lower_bound, upper_bound, expected",
    [
        (
            np.zeros(3),
            np.array([0.0002877431832243, 0.00763968126032, 0.01217268029151]),
            np.array(
                [
                    [
                        4.0080360351800763e00,
                        1.6579091056425378e02,
                        1.7322297746691254e02,
                    ],
                    [
                        1.6579091056425378e02,
                        1.6088016292793940e04,
                        1.1041403355728811e04,
                    ],
                    [
                        1.7322297746691254e02,
                        1.1041403355728811e04,
                        9.2992625728417297e03,
                    ],
                ]
            ),
            -np.ones(3),
            np.ones(3),
            (np.array([0.000122403, 3.92712e-06, -8.2519e-06]), 2),
        ),
        (
            np.zeros(3),
            np.array([7.898833044695e-06, 254.9676549378, 0.0002864050095122]),
            np.array(
                [
                    [3.97435226e00, 1.29126446e02, 1.90424789e02],
                    [1.29126446e02, 1.08362658e04, 9.05024598e03],
                    [1.90424789e02, 9.05024598e03, 1.06395102e04],
                ]
            ),
            np.array([-1.0, 0, -1.0]),
            np.ones(3),
            (np.array([-4.89762e-06, 0.0, 6.0738e-08]), 1),
        ),
    ],
)
def test_bounded_newton_trustregion(
    x,
    model_gradient,
    model_hessian,
    lower_bound,
    upper_bound,
    expected,
):
    options = {"gatol": 1e-8, "grtol": 1e-8}
    x_expected, niter_expected = expected

    x_out, niter_out = minimize_trust_bntr(
        x, model_gradient, model_hessian, lower_bound, upper_bound, options
    )

    aaae(x_out, x_expected)
    assert niter_out == niter_expected


def test_trustregion_conjugate_gradient():
    model_gradient = np.array([0.00028774, 0.00763968, 0.01217268])
    model_hessian = np.array(
        [
            [4.00803604e00, 1.65790911e02, 1.73222977e02],
            [1.65790911e02, 1.60880163e04, 1.10414034e04],
            [1.73222977e02, 1.10414034e04, 9.29926257e03],
        ]
    )

    trustregion_radius = 9.5367431640625e-05

    x_expected = np.array([9.50204689e-05, 3.56030822e-06, -7.30627902e-06])

    x_out = minimize_trust_cg(model_gradient, model_hessian, trustregion_radius)

    aaae(x_out, x_expected)
