import itertools

import numpy as np
import pandas as pd
import pytest
from estimagic.estimation.msm_weighting import assemble_block_diagonal_matrix
from estimagic.estimation.msm_weighting import get_weighting_matrix
from numpy.testing import assert_array_almost_equal as aaae


@pytest.fixture
def expected_values():
    values = np.array([[1, 2, 0, 0], [3, 4, 0, 0], [0, 0, 5, 6], [0, 0, 7, 8]])
    return values


cov_np = np.diag([1, 2, 3])
cov_pd = pd.DataFrame(cov_np)

test_cases = itertools.product([cov_np, cov_pd], ["diagonal", "optimal"])


@pytest.mark.parametrize("moments_cov, method", test_cases)
def test_get_weighting_matrix(moments_cov, method):
    calculated = get_weighting_matrix(moments_cov, method)

    if isinstance(moments_cov, pd.DataFrame):
        assert calculated.index.equals(moments_cov.index)
        assert calculated.columns.equals(moments_cov.columns)
        calculated = calculated.to_numpy()

    expected = np.diag(1 / np.array([1, 2, 3]))
    aaae(calculated, expected)


def test_assemble_block_diagonal_matrix_pd(expected_values):
    matrices = [
        pd.DataFrame([[1, 2], [3, 4]]),
        pd.DataFrame([[5, 6], [7, 8]], columns=[2, 3], index=[2, 3]),
    ]
    calculated = assemble_block_diagonal_matrix(matrices)
    assert isinstance(calculated, pd.DataFrame)
    assert calculated.index.equals(calculated.columns)
    assert calculated.index.tolist() == [0, 1, 2, 3]
    aaae(calculated, expected_values)


def test_assemble_block_diagonal_matrix_mixed(expected_values):
    matrices = [pd.DataFrame([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
    calculated = assemble_block_diagonal_matrix(matrices)
    assert isinstance(calculated, np.ndarray)
    aaae(calculated, expected_values)
