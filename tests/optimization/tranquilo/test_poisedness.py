import numpy as np
import pytest
from estimagic.optimization.tranquilo.poisedness import _reshape_coef_to_square_terms
from estimagic.optimization.tranquilo.poisedness import get_poisedness_constant
from estimagic.optimization.tranquilo.poisedness import improve_poisedness
from estimagic.optimization.tranquilo.poisedness import lagrange_poly_matrix
from numpy.testing import assert_array_almost_equal as aaae


def evaluate_scalar_model(x, intercept, linear_terms, square_terms):
    return intercept + linear_terms.T @ x + 0.5 * x.T @ square_terms @ x


# ======================================================================================
# Improve poisedness
# ======================================================================================


def test_improve_poisedness():
    sample = np.array(
        [
            [-0.98, -0.96],
            [-0.96, -0.98],
            [0, 0],
            [0.98, 0.96],
            [0.96, 0.98],
            [0.94, 0.94],
        ]
    )
    expected_sample = np.array(
        [
            [0.99974443, -0.02260675],
            [-0.96, -0.98],
            [-0.02131938, 0.03287205],
            [0.98, 0.96],
            [-0.52862931, 0.84885279],
            [0.2545369, -0.96706306],
        ]
    )
    expected_lambdas = [
        5324.240935366314,
        36.87996947175511,
        11.090857556966462,
        1.3893207179888898,
        1.0016763267639168,
    ]

    got_sample, got_lambdas = improve_poisedness(sample)

    aaae(got_sample, expected_sample)
    aaae(got_lambdas, expected_lambdas)


# ======================================================================================
# Lambda poisedness constant
# ======================================================================================

TEST_CASES = [
    (
        np.array(
            [
                [-0.98, -0.96],
                [-0.96, -0.98],
                [0, 0],
                [0.98, 0.96],
                [0.96, 0.98],
                [0.94, 0.94],
            ]
        ),
        5324,
    ),
    (
        np.array(
            [
                [-0.98, -0.96],
                [-0.96, -0.98],
                [0, 0],
                [0.98, 0.96],
                [0.96, 0.98],
                [0.707, -0.707],
            ]
        ),
        36.88,
    ),
    (
        np.array(
            [
                [-0.967, 0.254],
                [-0.96, -0.98],
                [0, 0],
                [0.98, 0.96],
                [-0.199, 0.979],
                [0.707, -0.707],
            ]
        ),
        1.001,
    ),
]


@pytest.mark.parametrize("sample, expected", TEST_CASES)
def test_poisedness_scaled_precise(sample, expected):
    """Test cases are taken from :cite:`Conn2009` p. 99."""

    got, *_ = get_poisedness_constant(sample)
    assert np.allclose(got, expected, rtol=1e-2)


TEST_CASES = [
    (
        np.array(
            [
                [0.848, 0.528],
                [-0.96, -0.98],
                [0, 0],
                [0.98, 0.96],
                [-0.96, -0.98],
                [0.707, -0.707],
            ]
        ),
        15.66,
    ),
    (
        np.array(
            [
                [-0.848, 0.528],
                [-0.96, -0.98],
                [0, 0],
                [0.98, 0.96],
                [-0.89, 0.996],
                [0.707, -0.707],
            ]
        ),
        1.11,
    ),
    (
        np.array(
            [
                [-0.967, 0.254],
                [-0.96, -0.98],
                [0, 0],
                [0.98, 0.96],
                [-0.89, 0.996],
                [0.707, -0.707],
            ]
        ),
        1.01,
    ),
]


@pytest.mark.xfail(reason="Imprecise results, but expected decrease in lambda.")
@pytest.mark.parametrize("sample, expected", TEST_CASES)
def test_poisedness_scaled_imprecise(sample, expected):
    """Test cases are taken from :cite:`Conn2009` p. 99."""

    got, *_ = get_poisedness_constant(sample)
    assert np.allclose(got, expected, rtol=1e-2)


TEST_CASES = [
    (
        np.array(
            [
                [0.524, 0.0006],
                [0.032, 0.323],
                [0.187, 0.890],
                [0.5, 0.5],
                [0.982, 0.368],
                [0.774, 0.918],
            ]
        ),
        1,
    )
]


@pytest.mark.parametrize("sample, expected", TEST_CASES)
def test_poisedness_unscaled_precise(sample, expected):
    """This test case is taken from :cite:`Conn2009` p. 45."""
    n_params = sample.shape[1]

    radius = 0.5
    center = 0.5 * np.ones(n_params)
    sample_centered = (sample - center) / radius

    got, *_ = get_poisedness_constant(sample_centered)
    assert np.allclose(got, expected, rtol=1e-2)


TEST_CASES = [
    (
        np.array(
            [
                [0.05, 0.1],
                [0.1, 0.05],
                [0.5, 0.5],
                [0.95, 0.9],
                [0.9, 0.95],
                [0.85, 0.85],
            ]
        ),
        440,
    ),
    (
        np.array(
            [
                [0.01, 0.02],
                [0.02, 0.01],
                [0.5, 0.5],
                [0.99, 0.98],
                [0.98, 0.99],
                [0.97, 0.97],
            ]
        ),
        21296,
    ),
    (
        np.array(
            [
                [0.524, 0.0006],
                [0.032, 0.323],
                [0.187, 0.890],
                [0.854, 0.853],
                [0.982, 0.368],
                [0.774, 0.918],
            ]
        ),
        524982,
    ),
]


@pytest.mark.xfail(reason="Lambda is scale-sensitive.")
@pytest.mark.parametrize("sample, expected", TEST_CASES)
def test_poisedness_unscaled_imprecise(sample, expected):
    """Test cases are taken from :cite:`Conn2009` p. 43ff."""
    n_params = sample.shape[1]

    radius = 0.5
    center = 0.5 * np.ones(n_params)
    sample_centered = (sample - center) / radius

    got, *_ = get_poisedness_constant(sample_centered)
    assert np.allclose(got, expected, rtol=1e-2)


# ======================================================================================
# Lagrange polynomials
# ======================================================================================

TEST_CASES = [
    (
        np.array([[0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2], [0.5, 0.5]]),
        np.array(
            [
                [
                    1,
                    -1.5,
                    -1.5,
                    1,
                    1,
                    1,
                ],
                [
                    0,
                    5 / 3,
                    -1 / 3,
                    -1.64705882e00,
                    -7.64705882e-01,
                    3.52941176e-01,
                ],
                [
                    0,
                    -1 / 3,
                    5 / 3,
                    3.52941176e-01,
                    -7.64705882e-01,
                    -1.64705882e00,
                ],
                [
                    0,
                    -5 / 12,
                    1 / 12,
                    9.11764706e-01,
                    -5.88235294e-02,
                    -8.82352941e-02,
                ],
                [
                    -0,
                    -1 / 6,
                    -1 / 6,
                    1.76470588e-01,
                    1.11764706e00,
                    1.76470588e-01,
                ],
                [
                    0,
                    1 / 12,
                    -5 / 12,
                    -8.82352941e-02,
                    -5.88235294e-02,
                    9.11764706e-01,
                ],
                [
                    0,
                    2 / 3,
                    2 / 3,
                    -7.05882353e-01,
                    -4.70588235e-01,
                    -7.05882353e-01,
                ],
            ]
        ),
        np.array([1, 0.84, 0.84, 0.99, 0.96, 0.99, 0.37]),
    )
]


@pytest.mark.parametrize("sample, expected_lagrange_mat, expected_critval", TEST_CASES)
def test_lagrange_poly_matrix(sample, expected_lagrange_mat, expected_critval):
    """This test case is taken from :cite:`Conn2009` p. 62."""
    sample = np.array([[0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2], [0.5, 0.5]])
    n_params = sample.shape[1]

    lagrange_mat = lagrange_poly_matrix(sample)
    aaae(lagrange_mat, expected_lagrange_mat)

    for idx, lagrange_poly in enumerate(lagrange_mat):
        intercept = lagrange_poly[0]
        linear_terms = lagrange_poly[1 : n_params + 1]
        _coef_square_terms = lagrange_poly[n_params + 1 :]
        square_terms = _reshape_coef_to_square_terms(_coef_square_terms, n_params)

        got = evaluate_scalar_model(sample[idx], intercept, linear_terms, square_terms)
        aaae(got, expected_critval[idx], decimal=2)
