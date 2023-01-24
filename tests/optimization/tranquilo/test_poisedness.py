import numpy as np
import pytest
from estimagic.optimization.tranquilo.poisedness import _get_minimize_options
from estimagic.optimization.tranquilo.poisedness import _lagrange_poly_matrix
from estimagic.optimization.tranquilo.poisedness import _reshape_coef_to_square_terms
from estimagic.optimization.tranquilo.poisedness import get_poisedness_constant
from estimagic.optimization.tranquilo.poisedness import improve_poisedness
from numpy.testing import assert_array_almost_equal as aaae


def evaluate_scalar_model(x, intercept, linear_terms, square_terms):
    return intercept + linear_terms.T @ x + 0.5 * x.T @ square_terms @ x


# ======================================================================================
# Improve poisedness
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
        "sphere",
        5,
        [
            5324.240935366314,
            36.87996947175511,
            11.090857556966462,
            1.3893207179888898,
            1.0016763267639168,
        ],
    ),
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
        "cube",
        10,
        [
            10648.478006222356,
            49.998826793338836,
            13.145227394549012,
            1.0313287779903457,
            1.008398336326099,
            1.0306831620836225,
            1.0019247733166188,
            1.0044418474330754,
            1.0024393102571791,
            1.0017007017773365,
        ],
    ),
    (
        np.array(
            [
                [-0.98, -0],
                [-0.96, -0.01],
                [0, 0],
                [-0.02, 0.98],
                [0.03, -0.96],
                [0.94, 0.06],
            ]
        ),
        "sphere",
        5,
        [
            50.83088699521032,
            1.4010345122261196,
            1.109469103188152,
            1.0614725892080803,
            1.0368961283088556,
        ],
    ),
    (
        np.array(
            [
                [-0.98, 0.0],
                [-0.56, -0.01],
                [-0.3, -0.07],
                [0.98, 0.02],
                [0.46, 0.03],
                [0.94, 0.06],
            ]
        ),
        "sphere",
        5,
        [
            687.9333361325548,
            22.830295678507802,
            11.89595397927371,
            1.590858593504958,
            1.1143219029197806,
        ],
    ),
]


@pytest.mark.parametrize("sample, shape, maxiter, expected", TEST_CASES)
def test_improve_poisedness(sample, shape, maxiter, expected):
    _, got_lambdas = improve_poisedness(sample=sample, shape=shape, maxiter=maxiter)
    aaae(got_lambdas[-5:], expected[-5:], decimal=2)


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
        5324.241743151584,
    ),
    (
        np.array(
            [
                [-0.98, -0.96],
                [-0.96, -0.98],
                [0.0, 0.0],
                [0.98, 0.96],
                [0.96, 0.98],
                [-0.70710678, 0.70710678],
            ]
        ),
        36.87996947175511,
    ),
    (
        np.array(
            [
                [-0.98, -0.96],
                [-0.96, -0.98],
                [0.0, 0.0],
                [0.84885278, -0.52862932],
                [0.96, 0.98],
                [-0.70710678, 0.70710678],
            ]
        ),
        11.090857500607644,
    ),
    (
        np.array(
            [
                [-0.98, -0.96],
                [-0.02260674, 0.99974443],
                [0.0, 0.0],
                [0.84885278, -0.52862932],
                [0.96, 0.98],
                [-0.70710678, 0.70710678],
            ]
        ),
        1.3893205660280858,
    ),
    (
        np.array(
            [
                [-0.98, -0.96],
                [-0.02260674, 0.99974443],
                [0.0, 0.0],
                [0.84885278, -0.52862932],
                [0.96, 0.98],
                [-0.96706306, 0.2545369],
            ]
        ),
        1.0016763272061744,
    ),
]


@pytest.mark.parametrize("sample, expected", TEST_CASES)
def test_poisedness_constant_scaled(sample, expected):
    """Test cases are modified versions from :cite:`Conn2009` p.

    99.

    """

    got, *_ = get_poisedness_constant(sample, shape="sphere")
    assert np.allclose(got, expected)


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
                [0.0, 0.0],
                [0.98, 0.96],
                [0.96, 0.98],
                [-0.707, 0.707],
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
def test_poisedness_constant_textbook_scaled(sample, expected):
    """Test cases are taken from :cite:`Conn2009` p.

    99.

    """

    got, *_ = get_poisedness_constant(sample, shape="sphere")
    assert np.allclose(got, expected, rtol=1e-3)


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
def test_poisedness_constant_textbook_unscaled(sample, expected):
    """This test case is taken from :cite:`Conn2009` p.

    45.

    """
    n_params = sample.shape[1]

    radius = 0.5
    center = 0.5 * np.ones(n_params)
    sample_scaled = (sample - center) / radius

    got, *_ = get_poisedness_constant(sample_scaled, shape="sphere")
    assert np.allclose(got, expected, rtol=1e-3)


def test_invalid_shape_argument():
    with pytest.raises(ValueError):
        assert _get_minimize_options(shape="ellipse", n_params=10)


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
    """This test case is taken from :cite:`Conn2009` p.

    62.

    """
    sample = np.array([[0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2], [0.5, 0.5]])
    n_params = sample.shape[1]

    lagrange_mat = _lagrange_poly_matrix(sample)
    aaae(lagrange_mat, expected_lagrange_mat)

    for idx, lagrange_poly in enumerate(lagrange_mat):
        intercept = lagrange_poly[0]
        linear_terms = lagrange_poly[1 : n_params + 1]
        _coef_square_terms = lagrange_poly[n_params + 1 :]
        square_terms = _reshape_coef_to_square_terms(_coef_square_terms, n_params)

        got = evaluate_scalar_model(sample[idx], intercept, linear_terms, square_terms)
        aaae(got, expected_critval[idx], decimal=2)
