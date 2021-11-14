import numpy as np
import pytest
from estimagic.examples.cartis_roberts import CARTIS_ROBERTS_PROBLEMS
from estimagic.examples.cartis_roberts import get_start_points_bdvalues
from numpy.testing import assert_array_almost_equal


@pytest.mark.parametrize("name, specification", list(CARTIS_ROBERTS_PROBLEMS.items()))
def test_cratis_roberts_function_at_start_x(name, specification):
    _criterion = specification["criterion"]
    _x = specification["start_x"]
    _contributions = _criterion(_x)
    calculated = _contributions @ _contributions
    expected = specification["start_criterion"]
    assert np.allclose(calculated, expected)


def test_get_start_points_bdvalues():
    expected = np.array([-0.1389, -0.2222, -0.2500, -0.2222, -0.1389])
    result = get_start_points_bdvalues(5)
    assert_array_almost_equal(expected, result, decimal=4)
