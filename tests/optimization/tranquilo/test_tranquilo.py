import numpy as np
from estimagic.optimization.tranquilo.tranquilo import _tranquilo
from numpy.testing import assert_array_almost_equal as aaae


def _scalar_sphere(x):
    return x @ x


def test_internal_tranquilo_with_scalar_sphere():
    res = _tranquilo(
        criterion=_scalar_sphere,
        x=np.ones(5),
        functype="scalar",
    )

    aaae(res["solution_x"], np.zeros(5))
