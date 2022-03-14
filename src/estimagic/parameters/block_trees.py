"""Functions to convert between array and block-tree representations of a matrix."""
import numpy as np
import pandas as pd
from estimagic.parameters.tree_registry import get_registry
from pybaum import tree_flatten
from pybaum import tree_just_flatten as tree_leaves
from pybaum import tree_unflatten


def matrix_to_block_tree(matrix, tree1, tree2):
    """Convert a matrix (2-dimensional array) to block-tree.

    A block tree most often arises when one applies an operation to a function that maps
    between two trees. For certain functions this results in a 2-dimensional data array.
    Two main examples are the Jacobian of the function f : tree2 -> tree1, which results
    in a block tree structure, or the covariance matrix of a tree, in which case tree1 =
    tree2.

    Args:
        matrix (numpy.ndarray): 2d representation of the block tree. Has shape (m, n).
        tree1: A pytree. If flattened to scalars has length m.
        tree2: A pytree. If flattened to scalars has length n.

    Returns:
        block_tree: A (block) pytree.

    """
    _check_dimensions_matrix(matrix, tree1, tree2)

    flat1, treedef1 = tree_flatten(tree1)
    flat2, treedef2 = tree_flatten(tree2)

    flat1_np = [_convert_pandas_objects_to_numpy(leaf) for leaf in flat1]
    flat2_np = [_convert_pandas_objects_to_numpy(leaf) for leaf in flat2]

    shapes1 = [np.shape(a) for a in flat1_np]
    shapes2 = [np.shape(a) for a in flat2_np]

    block_bounds1 = np.cumsum([int(np.product(s)) for s in shapes1[:-1]])
    block_bounds2 = np.cumsum([int(np.product(s)) for s in shapes2[:-1]])

    blocks = []
    for leaf1, s1, submat in zip(
        flat1, shapes1, np.split(matrix, block_bounds1, axis=0)
    ):
        row = []
        for leaf2, s2, block_values in zip(
            flat2, shapes2, np.split(submat, block_bounds2, axis=1)
        ):
            raw_block = block_values.reshape((*s1, *s2))
            block = _convert_raw_block_to_pandas(raw_block, leaf1, leaf2)
            row.append(block)

        blocks.append(row)

    block_tree = tree_unflatten(
        treedef1, [tree_unflatten(treedef2, row) for row in blocks]
    )

    return block_tree


def hessian_to_block_tree(hessian, f0, params):
    """Convert a Hessian array to block-tree format.

    Args:
        hessian (np.ndarray): The Hessian, 2- or 3-dimensional array representation of
            the resulting block-tree.
        f0 (pytree): The function evaluated at params.
        params (pytree): The params.

    Returns:
        hessian_block_tree (pytree): The pytree

    """
    _check_dimensions_hessian(hessian, f0, params)

    if hessian.ndim == 2:
        hessian = hessian[np.newaxis]

    flat_f, treedef_f = tree_flatten(f0)
    flat_p, treedef_p = tree_flatten(params)

    flat_f_np = [_convert_pandas_objects_to_numpy(leaf) for leaf in flat_f]
    flat_p_np = [_convert_pandas_objects_to_numpy(leaf) for leaf in flat_p]

    shapes_f = [np.shape(a) for a in flat_f_np]
    shapes_p = [np.shape(a) for a in flat_p_np]

    block_bounds_f = np.cumsum([int(np.product(s)) for s in shapes_f[:-1]])
    block_bounds_p = np.cumsum([int(np.product(s)) for s in shapes_p[:-1]])

    sub_block_trees = []
    for s0, subarr in zip(shapes_f, np.split(hessian, block_bounds_f, axis=0)):
        blocks = []
        for leaf1, s1, submat in zip(
            flat_p, shapes_p, np.split(subarr, block_bounds_p, axis=1)
        ):
            row = []
            for leaf2, s2, block_values in zip(
                flat_p, shapes_p, np.split(submat, block_bounds_p, axis=2)
            ):
                raw_block = block_values.reshape(((*s0, *s1, *s2)))
                raw_block = np.squeeze(raw_block)
                block = _convert_raw_block_to_pandas(raw_block, leaf1, leaf2)
                row.append(block)
            blocks.append(row)
        block_tree = tree_unflatten(
            treedef_p, [tree_unflatten(treedef_p, row) for row in blocks]
        )
        sub_block_trees.append(block_tree)

    hessian_block_tree = tree_unflatten(treedef_f, sub_block_trees)
    return hessian_block_tree


def block_tree_to_matrix(block_tree, tree1, tree2):
    """Convert a block tree to a matrix.

    A block tree most often arises when one applies an operation to a function that maps
    between two trees. Two main examples are the Jacobian of the function
    f : tree2 -> tree1, which results in a block tree structure, or the covariance
    matrix of a tree, in which case tree1 = tree2.

    Args:
        block_tree: A (block) pytree, must match dimensions of tree1 and tree2.
        tree1: A pytree.
        tree2: A pytree.

    Returns:
        matrix (np.ndarray): 2d array containing information stored in block_tree.

    """
    if len(block_tree) != len(tree1):
        raise ValueError("First dimension of block_tree does not match that of tree1.")

    selector_first_element = list(block_tree)[0] if isinstance(block_tree, dict) else 0
    if len(block_tree[selector_first_element]) != len(tree2):
        raise ValueError("Second dimension of block_tree does not match that of tree2.")

    flat1 = tree_leaves(tree1)
    flat2 = tree_leaves(tree2)
    flat_block_tree = tree_leaves(block_tree)

    flat1_np = [_convert_pandas_objects_to_numpy(leaf) for leaf in flat1]
    flat2_np = [_convert_pandas_objects_to_numpy(leaf) for leaf in flat2]

    size1 = [np.size(a) for a in flat1_np]
    size2 = [np.size(a) for a in flat2_np]

    n_blocks1 = len(size1)
    n_blocks2 = len(size2)

    block_rows_raw = [
        flat_block_tree[n_blocks2 * i : n_blocks2 * (i + 1)] for i in range(n_blocks1)
    ]

    block_rows = []
    for s1, row in zip(size1, block_rows_raw):
        shapes = [(s1, s2) for s2 in size2]
        row_np = [_convert_pandas_objects_to_numpy(leaf) for leaf in row]
        row_reshaped = _reshape_list(row_np, shapes)
        row_concatenated = np.concatenate(row_reshaped, axis=1)
        block_rows.append(row_concatenated)

    matrix = np.concatenate(block_rows, axis=0)

    _check_dimensions_matrix(matrix, tree1, tree2)
    return matrix


def block_tree_to_hessian(block_hessian, f0, params):
    """Convert a block tree to a Hessian array.

    See :func:`hessian_to_block_tree` for a detailed description.

    Args:
        block_hessian: A (block) pytree, must match dimensions of f0 and params.
        f0 (pytree): The function evaluated at params.
        params (pytree): The params.

    Returns:
        matrix (np.ndarray): 2d array containing information stored in block_tree.

    """
    flat_f = tree_leaves(f0)
    flat_p = tree_leaves(params)
    flat_block_tree = tree_leaves(block_hessian)

    flat_f_np = [_convert_pandas_objects_to_numpy(leaf) for leaf in flat_f]
    flat_p_np = [_convert_pandas_objects_to_numpy(leaf) for leaf in flat_p]

    size_f = [np.size(a) for a in flat_f_np]
    size_p = [np.size(a) for a in flat_p_np]

    n_blocks_f = len(size_f)
    n_blocks_p = len(size_p)

    block_rows_raw = [
        flat_block_tree[(n_blocks_p**2) * i : (n_blocks_p**2) * (i + 1)]
        for i in range(n_blocks_f)
    ]

    registry = get_registry(extended=True)
    n_p = len(tree_leaves(params, registry=registry))
    inner_treedef = matrix_to_block_tree(np.zeros((n_p, n_p)), params, params)

    inner_block_trees = [
        tree_unflatten(inner_treedef, block_row) for block_row in block_rows_raw
    ]
    inner_block_matrices = [
        block_tree_to_matrix(block_tree, params, params)[np.newaxis]
        for block_tree in inner_block_trees
    ]
    hessian = np.concatenate(inner_block_matrices, axis=0)

    _check_dimensions_hessian(hessian, f0, params)
    return hessian


def _convert_pandas_objects_to_numpy(obj):
    if not isinstance(obj, (pd.Series, pd.DataFrame)):
        return obj
    elif isinstance(obj, pd.Series):
        out = obj.to_numpy()
    elif "value" in obj.columns:
        out = obj["value"].to_numpy()
    else:
        out = obj.to_numpy()
    return out


def _convert_raw_block_to_pandas(raw_block, leaf1, leaf2):
    if np.ndim(raw_block) not in (1, 2):
        return raw_block

    if not _is_pd_object(leaf1) and not _is_pd_object(leaf2):
        return raw_block

    index1 = None if not _is_pd_object(leaf1) else leaf1.index
    index2 = None if not _is_pd_object(leaf2) else leaf2.index

    # can only happen if one leaf is a scalar and the other a pandas
    # object that is interpreted as one-dimensional. We want to convert
    # the block to a series wtih the index of the pandas object
    if np.ndim(raw_block) == 1:
        out = pd.Series(raw_block, index=_select_non_none(index1, index2))

    # can happen in two cases
    elif np.ndim(raw_block) == 2:
        # case 1: one leaf is scalar and the other is a DataFrame
        # without value column. We want to convert the block to a DataFrame
        # with same index and columns as original DataFrame
        if np.isscalar(leaf1) or np.isscalar(leaf2):
            if np.isscalar(leaf1):
                index, columns = leaf2.index, leaf2.columns
            elif np.isscalar(leaf2):
                index, columns = leaf1.index, leaf1.columns
            out = pd.DataFrame(raw_block, index=index, columns=columns)
        # case 2: both 1d Data structures and at least one of them is
        # a pandas object. We want to convert the result to a DataFrame
        # with index=index1 and columns=index2
        else:
            out = pd.DataFrame(raw_block, index=index1, columns=index2)

    return out


def _select_non_none(first, second):
    if first is None and second is None:
        raise ValueError()
    elif first is not None and second is not None:
        raise ValueError()
    elif first is None:
        out = second
    elif second is None:
        out = first
    return out


def _reshape_list(list_to_reshape, shapes):
    """Reshape list of numpy.ndarray according to list of shapes.

    Args:
        list_to_reshape (list): List containing numpy.ndarray's.
        shapes (list): List of shape tuples.

    Returns:
        reshaped (list): List containing the reshaped numpy.ndarray's.

    """
    if len(list_to_reshape) != len(shapes):
        raise ValueError("Arguments must have the same number of elements.")
    reshaped = [a.reshape(shape) for a, shape in zip(list_to_reshape, shapes)]
    return reshaped


def _is_pd_object(obj):
    return isinstance(obj, (pd.Series, pd.DataFrame))


def _check_dimensions_matrix(matrix, tree1, tree2):
    extended_registry = get_registry(extended=True)
    flat1 = tree_leaves(tree1, registry=extended_registry)
    flat2 = tree_leaves(tree2, registry=extended_registry)

    if matrix.shape[0] != len(flat1):
        raise ValueError("First dimension of matrix does not match that of tree1.")
    if matrix.shape[1] != len(flat2):
        raise ValueError("Second dimension of matrix does not match that of tree2.")


def _check_dimensions_hessian(hessian, f0, params):
    extended_registry = get_registry(extended=True)
    flat_f = tree_leaves(f0, registry=extended_registry)
    flat_p = tree_leaves(params, registry=extended_registry)

    if len(flat_f) == 1:
        if np.squeeze(hessian).ndim == 0:
            if len(flat_p) != 1:
                raise ValueError("Hessian dimension does not match those of params.")
        elif np.squeeze(hessian).ndim == 2:
            if np.squeeze(hessian).shape != 2 * (len(flat_p),):
                raise ValueError("Hessian dimension does not match those of params.")
        else:
            raise ValueError("Hessian must be 0- or 2-d if f0 is scalar-valued.")
    else:
        if hessian.ndim != 3:
            raise ValueError("Hessian must be 3d if f0 is multidimensional.")
        if hessian.shape[0] != len(flat_f):
            raise ValueError("First Hessian dimension does not match that of f0.")
        if hessian.shape[1:] != 2 * (len(flat_p),):
            raise ValueError(
                "Last two Hessian dimensions do not match those of params."
            )
