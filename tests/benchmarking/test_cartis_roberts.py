import numpy as np
import pytest
from estimagic.benchmarking.cartis_roberts import CARTIS_ROBERTS_PROBLEMS
from estimagic.benchmarking.cartis_roberts import get_start_points_bdvalues
from estimagic.benchmarking.cartis_roberts import get_start_points_msqrta
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


def test_get_start_points_msqrta():
    matlab_mat = np.array(
        [
            [0.8415, -0.7568, 0.4121, -0.2879, -0.1324],
            [-0.9918, -0.9538, 0.9200, -0.6299, -0.5064],
            [0.9988, -0.4910, -0.6020, 0.9395, -0.9301],
            [-0.9992, -0.0265, -0.4041, 0.2794, -0.8509],
            [0.9235, 0.1935, 0.9365, -0.8860, 0.1760],
        ]
    )
    expected = 0.2 * matlab_mat.flatten()
    result = get_start_points_msqrta(5)
    assert_array_almost_equal(result, expected, decimal=4)
