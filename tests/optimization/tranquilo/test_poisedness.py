import numpy as np
import pytest
from estimagic.optimization.tranquilo.poisedness import build_interpolation_matrix
from estimagic.optimization.tranquilo.poisedness import get_lagrange_polynomial
from estimagic.optimization.tranquilo.poisedness import get_lambda_poisedness_constant
from numpy.testing import assert_allclose
from numpy.testing import assert_array_almost_equal


def aaae(x, y, case=None):
    tolerance = {
        None: 7,
        "hessian": 3,
    }
    assert_array_almost_equal(x, y, decimal=tolerance[case])


TEST_CASES = [
    (np.array([[1, 0.0], [0.0, 1]]), 1.0),
    (
        np.array(
            [
                [0.024, -0.4994],
                [-0.468, -0.177],
                [-0.313, 0.39],
                [0.482, -0.132],
            ]
        ),
        3.226777830135947,
    ),
]


@pytest.mark.parametrize("sample, expected", TEST_CASES)
def test_poisedness_constant(sample, expected):
    n_params = 2

    lower_bounds = -1 * np.ones(n_params)
    upper_bounds = 1 * np.ones(n_params)

    got = get_lambda_poisedness_constant(sample, lower_bounds, upper_bounds)

    assert_allclose(got, expected, rtol=1e-2, atol=1e-8)


TEST_CASES = [
    (
        np.array(
            [
                [-0.45, -0.4],
                [-0.4, -0.45],
                [0.45, 0.4],
                [0.4, 0.45],
                [0.35, 0.35],
            ]
        ),
        1179.497744338786,
    )
]


@pytest.mark.parametrize("sample, expected", TEST_CASES)
def test_poisedness_constant_imprecise(sample, expected):
    n_params = 2

    lower_bounds = -1 * np.ones(n_params)
    upper_bounds = 1 * np.ones(n_params)

    got = get_lambda_poisedness_constant(sample, lower_bounds, upper_bounds)

    assert 1178 <= got <= 1180


TEST_CASES = [
    (
        np.array([[0.0, 0.0], [3.2, -0.1]]),
        np.array([2.2, -0.1]),
        [
            [np.array([-0.0, 10.0]), np.zeros((2, 2))],
            [np.array([1.0, 22.0]), np.zeros((2, 2))],
        ],
    ),
    (
        np.array([[0.0, 0.0], [1.0, 0.0], [-0.1, 0.0]]),
        np.array([0.1, 0.9]),
        [
            [
                np.array([7.0, -2.0]),
                np.array(
                    [
                        [-2.00000000e01, -7.10542736e-15],
                        [-7.10542736e-15, 0.00000000e00],
                    ]
                ),
            ],
            [
                np.array([0.27272727, -0.02020202]),
                np.array([[1.81818182, 0.0], [0.0, 0.0]]),
            ],
            [
                np.array([-7.27272727, 0.90909091]),
                np.array(
                    [[1.81818182e01, 3.55271368e-15], [3.55271368e-15, 2.84217094e-14]]
                ),
            ],
        ],
    ),
    (
        np.array([[0.0, 0.0], [1.0, 0.0], [-0.1, 0.0], [-0.1, 2.0]]),
        np.array([0.1, 0.9]),
        [
            [
                np.array([6.44186047, -0.46511628]),
                np.array([[-19.96554694, -0.31007752], [-0.31007752, 6.20155039]]),
            ],
            [
                np.array([0.2670895, -0.00469814]),
                np.array([[1.81852983, -0.0031321], [-0.0031321, 0.06264192]]),
            ],
            [
                np.array([-6.8794926, -0.17230444]),
                np.array([[18.15754444, 0.21846371], [0.21846371, -4.36927414]]),
            ],
            [
                np.array([-0.13953488, 0.38372093]),
                np.array([[0.00861326, -0.07751938], [-0.07751938, 1.5503876]]),
            ],
        ],
    ),
    (
        np.array([[0.0, 0.0], [1.0, 0.0], [-0.1, 0.0], [-0.1, 2.0], [-1.1, 1.0]]),
        np.array([0.1, 0.9]),
        [
            [
                np.array([-10.37692308, -3.52307692]),
                np.array([[-20.0, -19.30769231], [-19.30769231, -3.38461538]]),
            ],
            [
                np.array([1.02027972, 0.13224553]),
                np.array([[1.81818182, 0.83061383], [0.83061383, 0.33877234]]),
            ],
            [
                np.array([9.31433566, 2.77202797]),
                np.array([[18.18181818, 18.43006993], [18.43006993, 4.13986014]]),
            ],
            [
                np.array([0.03461538, 0.41538462]),
                np.array(
                    [[1.23788147e-15, 3.84615385e-02], [3.84615385e-02, 9.23076923e-01]]
                ),
            ],
            [
                np.array([-0.76153846, -0.13846154]),
                np.array(
                    [
                        [5.11125253e-16, -8.46153846e-01],
                        [-8.46153846e-01, -3.07692308e-01],
                    ]
                ),
            ],
        ],
    ),
]


@pytest.mark.parametrize("sample, center, expected", TEST_CASES)
def test_get_lagrange_polynomial(sample, center, expected):

    n_samples, n_params = sample.shape
    sample_centered = sample - center

    interpolation_mat = build_interpolation_matrix(
        sample_centered.T, n_params, n_samples
    )

    for index in range(n_samples):
        linear_terms, square_terms = get_lagrange_polynomial(
            sample,
            interpolation_mat,
            index,
        )
        aaae(linear_terms, expected[index][0])
        aaae(square_terms, expected[index][1])


TEST_CASES = [
    (np.array([[-2.2, 0.1], [1.0, 0.0]]), np.array([[-2.2, 0.1], [1.0, 0.0]])),
    (
        np.array([[-0.1, -0.9], [0.9, -0.9], [-0.2, -0.9]]),
        np.array(
            [
                [0.3362, 0.2592, 0.34445, -0.1, -0.9],
                [0.2592, 1.3122, 0.19845, 0.9, -0.9],
                [0.34445, 0.19845, 0.36125, -0.2, -0.9],
                [-0.1, 0.9, -0.2, 0.0, 0.0],
                [-0.9, -0.9, -0.9, 0.0, 0.0],
            ]
        ),
    ),
    (
        np.array([[-0.1, -0.9], [0.9, -0.9], [-0.2, -0.9], [-0.2, 1.1]]),
        np.array(
            [
                [0.3362, 0.2592, 0.34445, 0.47045, -0.1, -0.9],
                [0.2592, 1.3122, 0.19845, 0.68445, 0.9, -0.9],
                [0.34445, 0.19845, 0.36125, 0.45125, -0.2, -0.9],
                [0.47045, 0.68445, 0.45125, 0.78125, -0.2, 1.1],
                [-0.1, 0.9, -0.2, -0.2, 0.0, 0.0],
                [-0.9, -0.9, -0.9, 1.1, 0.0, 0.0],
            ]
        ),
    ),
]


@pytest.mark.parametrize("sample, expected", TEST_CASES)
def test_get_interpolation_matrix(sample, expected):
    sample_t = sample.T
    n_params, n_samples = sample_t.shape

    mat = build_interpolation_matrix(sample_t, n_params, n_samples)
    aaae(mat, expected)
