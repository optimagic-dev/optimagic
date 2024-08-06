import numpy as np
import pytest
from optimagic import mark, maximize


def test_error_when_maximizing_least_squares():
    @mark.least_squares
    def f(x):
        return x

    with pytest.raises(ValueError):
        maximize(
            f,
            np.arange(3),
            algorithm="scipy_ls_lm",
        )
