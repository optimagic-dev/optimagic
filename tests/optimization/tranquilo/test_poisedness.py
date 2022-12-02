import numpy as np
import pytest
from estimagic.optimization.tranquilo.poisedness import _evaluate_scalar_model
from estimagic.optimization.tranquilo.poisedness import _reshape_coef_to_square_terms
from estimagic.optimization.tranquilo.poisedness import get_lagrange_poly_matrix
from numpy.testing import assert_array_almost_equal as aaae


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

    lagrange_mat = get_lagrange_poly_matrix(sample)
    aaae(lagrange_mat, expected_lagrange_mat)

    for idx, lagrange_poly in enumerate(lagrange_mat):
        intercept = lagrange_poly[0]
        linear_terms = lagrange_poly[1 : n_params + 1]
        _coef_square_terms = lagrange_poly[n_params + 1 :]
        square_terms = _reshape_coef_to_square_terms(_coef_square_terms, n_params)

        got = _evaluate_scalar_model(sample[idx], intercept, linear_terms, square_terms)
        aaae(got, expected_critval[idx], decimal=2)
