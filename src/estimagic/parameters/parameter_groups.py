def get_params_groups(flat_params):
    """Create parameter groups.

    Args:
        flat_params (FlatParams)

    Returns:
        groups (list): list of strings. For each entry in flat params the key of the
            group to which the parameter belongs.

    """
    group_candidates = _determine_group_candidates(flat_params)
    groups = _determine_groups(flat_params, group_candidates)
    return groups


def _determine_group_candidates(flat_params):
    """Determine the group names from the given parameters.

    Only free parameters are used to generate group name candidates. Not every name must
    have a group name to which it belongs. Any name that is just a number would not
    have any group name generated for it.

    Args:
        flat_params (FlatParams)

    Returns:
        group_candidates (set): set of names for the different groups.

    """
    group_candidates = set()
    names_to_use = [
        name
        for name, is_free in zip(flat_params.names, flat_params.free_mask)
        if is_free
    ]
    for name in names_to_use:
        for component in name.split("_"):
            if not component.isdigit():
                group_candidates.add(component)
    return group_candidates


def _determine_groups(flat_params, group_candidates):
    """Given parameters and group candidates determine the group of each.

    Args:
        flat_params (FlatParams)
        group_candidates (set): set of names for the different groups. Not every name
            must have a group name to which it belongs. Any name that is just a number
            would not have any group name generated for it.

        groups (list): list of strings. For each entry in flat params the key of the
            group to which the parameter belongs.

    """
    groups = []
    for param_name, is_free in zip(flat_params.names, flat_params.free_mask):
        if is_free:
            groups.append(_determine_param_group(param_name, group_candidates))
        else:
            # Setting the group of non free parameters to the empty string ensures that
            # only free parameters are plotted
            groups.append("")
    return groups


def _determine_param_group(param_name, group_candidates):
    """Determine which group a parameter belongs to based on its name.

    Args:
        param_name (str): name of the parameter
        group_candidates (list): possible groups it can belong to.

    Returns:
        group (str): Either one entry of **group_candidates**. If none fits, the
            parameter's name is returned as its group name.

    """
    candidate_groups = [g for g in group_candidates if g in param_name.split("_")]
    if len(candidate_groups) == 0:
        return param_name
    else:  # if one or more candidate, return the longest candidate as "best" group name
        return max(candidate_groups, key=len)
