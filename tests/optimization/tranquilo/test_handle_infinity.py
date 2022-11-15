import numpy as np
from estimagic.optimization.tranquilo.handle_infinity import get_infinity_handler
from numpy.testing import assert_array_almost_equal as aaae


def test_clip_relative():
    func = get_infinity_handler("relative")

    fvecs = np.array([[1, np.inf, 3, 1], [-np.inf, 0, 1, 2], [-1, 5, 6, 3]])

    got = func(fvecs)

    expected = np.array([[1, 16, 3, 1], [-6, 0, 1, 2], [-1, 5, 6, 3]])

    aaae(got, expected)
