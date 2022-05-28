import numpy as np
import pandas as pd
from estimagic.parameters.tree_registry import get_registry
from pybaum import leaf_names


def get_params_groups(params, flat_params, max_group_size=8):
    """Create parameter groups.

    Args:
        params (pytree): parameters as supplied by the user.
        flat_params (FlatParams): processed parameters.

    Returns:
        groups (list): list of strings and None. For each entry in flat params the key
            of the group to which the parameter belongs. None if the parameter is not
            free.
        names (list): list of the parameter names to be displayed in the dashboard.

    """
    sep = "$$$+++"
    registry = get_registry(extended=True)
    paths = leaf_names(params, registry=registry, separator=sep)
    split_paths = [path.split(sep) for path in paths]
    group_and_names = [
        _get_group_and_name(path_list, is_free)
        for path_list, is_free in zip(split_paths, flat_params.free_mask)
    ]
    groups = np.array([group for group, _ in group_and_names])
    # !!! parameter names for the display in the dashboard are not used right now.
    # instead the full names are shown in the legend

    counts = pd.value_counts(groups)
    to_be_split = counts[counts > max_group_size]
    for group_name, n_occurrences in to_be_split.items():
        split_group_names = _split_long_group(
            group_name=group_name,
            n_occurrences=n_occurrences,
            max_group_size=max_group_size,
        )
        groups[groups == group_name] = split_group_names
    return groups.tolist()


def _get_group_and_name(path_list, is_free):
    """Create group and name from the path_list of a parameter.

    Args:
        path_list (list):
        is_free (bool): if True the parameter is free. If False, the parameter is fixed
            and won't change during the

    Returns:
        out (tuple): Tuple of length 2. The 1st entry is the group name of the
            parameter, the 2nd entry is the "first" name of the parameter (i.e.
            without its group).

    """
    if is_free:
        if len(path_list) == 1:
            out = (path_list[0], path_list[0])
        else:
            group_name = ", ".join(path_list[:-1])
            out = (group_name, path_list[-1])
    else:
        out = (None, "_".join(path_list))
    return out


def _split_long_group(group_name, n_occurrences, max_group_size=8):
    """Create new names that split a long group into chunks.

    Args:
        group_name (str): name of the group with too many members
        n_occurrences (int): number of occurrences of the group name
        max_group_size (int, optional): maximal number parameters that should be in a
            group.

    Returns:
        new_names (list): list of strings with length n_occurrences. Each is a new
            group name of the format "{group_name}_1", "{group_name}_2" etc. Each new
            group name occurs  at most max_group_size times.

    """
    quot, _ = divmod(n_occurrences, max_group_size)
    split = np.array_split([group_name] * n_occurrences, quot + 1)
    new_names = []
    for i, arr in enumerate(split):
        new_names += [f"{group_name}, {i + 1}"] * len(arr)
    return new_names
