import numpy as np
import pandas as pd
from estimagic.parameters.block_trees import matrix_to_block_tree
from pybaum import tree_equal


def test_matrix_to_block_tree_array_and_scalar():
    t = {"a": 1.0, "b": np.arange(2)}
    calculated = matrix_to_block_tree(np.arange(9).reshape(3, 3), t, t)

    expected = {
        "a": {"a": np.array(0), "b": np.array([1, 2])},
        "b": {"a": np.array([3, 6]), "b": np.array([[4, 5], [7, 8]])},
    }

    assert tree_equal(calculated, expected)


def test_matrix_to_block_tree_only_params_dfs():
    tree = {
        "a": pd.DataFrame(index=["a", "b"]).assign(value=[1, 2]),
        "b": pd.DataFrame(index=["j", "k", "l"]).assign(value=[3, 4, 5]),
    }

    calculated = matrix_to_block_tree(np.arange(25).reshape(5, 5), tree, tree)

    expected = {
        "a": {
            "a": pd.DataFrame([[0, 1], [5, 6]], columns=["a", "b"], index=["a", "b"]),
            "b": pd.DataFrame(
                [[2, 3, 4], [7, 8, 9]], columns=["j", "k", "l"], index=["a", "b"]
            ),
        },
        "b": {
            "a": pd.DataFrame(
                [[10, 11], [15, 16], [20, 21]],
                index=["j", "k", "l"],
                columns=["a", "b"],
            ),
            "b": pd.DataFrame(
                [[12, 13, 14], [17, 18, 19], [22, 23, 24]],
                index=["j", "k", "l"],
                columns=["j", "k", "l"],
            ),
        },
    }
    assert tree_equal(calculated, expected)


# one params df (make sure we don't get a list back)
# dataframe and scalar
# tests against jax
