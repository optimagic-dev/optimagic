import itertools

import numpy as np
import pandas as pd
import pytest
from estimagic.estimation.msm_weighting import _assemble_block_diagonal_matrix
from estimagic.estimation.msm_weighting import get_moments_cov
from estimagic.estimation.msm_weighting import get_weighting_matrix
from estimagic.parameters.block_trees import block_tree_to_matrix
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
    if isinstance(moments_cov, np.ndarray):
        fake_emp_moms = np.ones(len(moments_cov))
    else:
        fake_emp_moms = pd.Series(np.ones(len(moments_cov)), index=moments_cov.index)
    calculated = get_weighting_matrix(moments_cov, method, fake_emp_moms)

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
    calculated = _assemble_block_diagonal_matrix(matrices)
    assert isinstance(calculated, pd.DataFrame)
    assert calculated.index.equals(calculated.columns)
    assert calculated.index.tolist() == [0, 1, 2, 3]
    aaae(calculated, expected_values)


def test_assemble_block_diagonal_matrix_mixed(expected_values):
    matrices = [pd.DataFrame([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
    calculated = _assemble_block_diagonal_matrix(matrices)
    assert isinstance(calculated, np.ndarray)
    aaae(calculated, expected_values)


def test_get_moments_cov_runs_with_pytrees():
    np.random.seed(1234)
    data = np.random.normal(scale=[10, 5, 1], size=(100, 3))
    data = pd.DataFrame(data=data)

    def calc_moments(data, keys):
        means = data.mean()
        means.index = keys
        return means.to_dict()

    moment_kwargs = {"keys": ["a", "b", "c"]}

    calculated = get_moments_cov(
        data=data,
        calculate_moments=calc_moments,
        moment_kwargs=moment_kwargs,
        bootstrap_kwargs={"n_draws": 100},
    )

    fake_tree = {"a": 1, "b": 2, "c": 3}
    cov = block_tree_to_matrix(calculated, fake_tree, fake_tree)
    assert cov.shape == (3, 3)

    assert cov[0, 0] > cov[1, 1] > cov[2, 2]
