import numpy as np
import pytest
from estimagic.optimization.tranquilo.tranquilo import _tranquilo
from numpy.testing import assert_array_almost_equal as aaae


@pytest.mark.parametrize("functype", ["scalar", "least_squares"])
def test_internal_tranquilo_with_sphere_at_defaults(functype):
    func_dict = {
        "scalar": lambda x: x @ x,
        "least_squares": lambda x: x,
    }

    res = _tranquilo(
        criterion=func_dict[functype],
        x=np.arange(5),
        functype=functype,
    )
    aaae(res["solution_x"], np.zeros(5), decimal=5)
