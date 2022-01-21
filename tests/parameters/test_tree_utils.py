import numpy as np
import pytest
from estimagic.parameters.tree_utils import tree_flatten
from estimagic.parameters.tree_utils import tree_map
from estimagic.parameters.tree_utils import tree_multimap
from estimagic.parameters.tree_utils import tree_unflatten
from numpy.testing import assert_array_almost_equal as aaae


@pytest.fixture
def example_tree():
    return ([1, np.array([2, 3]), {"a": np.array([4, 5])}], 6)


@pytest.fixture
def example_flat():
    return [1, np.array([2, 3]), np.array([4, 5]), 6]


@pytest.fixture
def example_treedef():
    return (["*", "*", {"a": "*"}], "*")


def test_tree_flatten(example_tree, example_flat, example_treedef):
    flat, treedef = tree_flatten(example_tree)
    assert treedef == example_treedef
    _assert_list_with_arrays_is_equal(flat, example_flat)


def test_tree_unflatten(example_flat, example_treedef):
    unflat = tree_unflatten(example_treedef, example_flat)

    assert isinstance(unflat, tuple)
    assert isinstance(unflat[0], list)
    assert isinstance(unflat[1], int)
    assert len(unflat) == 2
    assert unflat[1] == 6
    assert unflat[0][0] == 1
    aaae(unflat[0][1], np.array([2, 3]))
    assert isinstance(unflat[0][2], dict)
    aaae(unflat[0][2]["a"], np.array([4, 5]))


def test_tree_map():
    tree = [{"a": 1, "b": 2, "c": {"d": 3, "e": 4}}]
    calculated = tree_map(lambda x: x * 2, tree)
    expected = [{"a": 2, "b": 4, "c": {"d": 6, "e": 8}}]
    assert calculated == expected


def test_tree_multimap():
    tree = [{"a": 1, "b": 2, "c": {"d": 3, "e": 4}}]
    mapped = tree_map(lambda x: x ** 2, tree)
    multimapped = tree_multimap(lambda x, y: x * y, tree, tree)
    assert mapped == multimapped


def _assert_list_with_arrays_is_equal(list1, list2):
    for first, second in zip(list1, list2):
        if isinstance(first, np.ndarray):
            aaae(first, second)
        else:
            assert first == second
