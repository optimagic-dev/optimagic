import numpy as np
import pandas as pd

from estimagic.optimization.utilities import number_of_triangular_elements_to_dimension


def consolidate_constraints(constraints, params):
    """Consolidate constraints with each other and remove redundant constraints.

    Args:
        constraints (list): List with constraint dictionaries. It is assumed that
            the selectors are already processed, sum constraints have been rewritten
            as linear constraints and pairwise_equality constraints have been rewritten
            as equality constraints.
        params (pd.DataFrame): see :ref:`params`.

    Returns:
        consolidated_constraints (list): This contains processed version of all
            constraints that require an actual kernel transformation. The information
            on all other constraints is subsumed in processed_params.
        processed_params (pd.DataFrame)

    """
    raw_eq, others = _split_constraints(constraints, "equality")
    eq_constraints = _consolidate_equality_constraints(raw_eq)

    fixes, others = _split_constraints(others, "fixed")
    fixed_value = _consolidate_fixes_with_equality_constraints(
        fixes, eq_constraints, params
    )

    pp = params.copy(deep=True)
    pp["_fixed_value"] = fixed_value
    pp["_is_fixed_to_value"] = pp["_fixed_value"].notnull()

    others = [c for c in others if not pp.iloc[c["index"]]["_is_fixed_to_value"].all()]

    others, pp = simplify_covariance_and_sdcorr_constraints(others, pp)

    pp = _consolidate_bounds_with_equality_constraints(eq_constraints, pp)

    others, pp = _plug_equality_constraints_into_selectors(eq_constraints, others, pp)

    linear_constraints, others = _split_constraints(others, "linear")

    if len(linear_constraints) > 0:
        linear_constraints = _consolidate_linear_constraints(linear_constraints, pp)

    consolidated_constraints = others + linear_constraints

    return consolidated_constraints, pp


def _consolidate_equality_constraints(equality_constraints):
    """Consolidate equality constraints as far as possible.

    Since equality is a transitive conditions we can consolidate any two equality
    constraints that have at least one parameter in common into one condition. Besides
    being faster, this also ensures that the result remains unchanged if equality
    constraints are are split into several different constraints or if they are
    specified in a different order.

    The index in the consolidated equality constraints is sorted in the same order
    as the index of params. This is no problem because order is irrelevant for
    equality constraints.

    Args:
        equality_constraints (list): List of dictionaries where each dictionary is a
            constraint. It is assumed that the selectors in the constraints were already
            processed.

    Returns:
        consolidated (list): List of consolidated equality constraints.
    """

    candidates = [constr["index"] for constr in equality_constraints]
    # drop constraints that just restrict one parameter to be equal to itself
    candidates = [c for c in candidates if len(c) >= 2]
    merged = _join_overlapping_lists(candidates)
    consolidated = [{"index": sorted(index), "type": "equality"} for index in merged]

    return consolidated


def _join_overlapping_lists(candidates):
    """Bundle all candidates with with non-empty intersection.

    Args:
        candidates (list): List of potentially overlapping lists.

    Returns:
        bundles (list): List of lists where all overlapping lists have been joined
            and sorted.

    """
    bundles = []

    while len(candidates) > 0:
        new_candidates = _unite_first_with_all_intersecting_elements(candidates)
        if len(candidates) == len(new_candidates):
            bundles.append(sorted(candidates[0]))
            candidates = candidates[1:]
        else:
            candidates = new_candidates

    return bundles


def _unite_first_with_all_intersecting_elements(indices):
    """Helper function to bundle overlapping indices.

    Args:
        indices (list): A list lists with indices.

    """
    first = set(indices[0])
    new_first = first
    new_others = []
    for idx in indices[1:]:
        if len(first.intersection(idx)) > 0:
            new_first = new_first.union(idx)
        else:
            new_others.append(idx)

    return [new_first] + new_others


def _consolidate_fixes_with_equality_constraints(
    fixed_constraints, equality_constraints, params
):
    """Consolidate fixes with equality constraints.

    If any equality constrained parameter is fixed, all of the parameters that are
    equal to it have to be fixed to the same value.

    Args:
        fixed_constraints (list): List of constraints of type "fixed".
        equality_constraints (list): List of constraints of type "equality".
        params (pd.DataFrame): see :ref:`params`

    Returns:
        fixed_value (pd.Series): Series with the fixed value for all parameters that
            are fixed and np.nan everywhere else. Has the same index as params.

    """
    fixed_value = pd.Series(index=params.index, data=np.nan)
    for fix in fixed_constraints:
        if "value" in fix:
            fixed_value.iloc[fix["index"]] = fix["value"]
        else:
            fixed_value.iloc[fix["index"]] = params["value"].iloc[fix["index"]]

    for eq in equality_constraints:
        if fixed_value.iloc[eq["index"]].notnull().any():
            valcounts = fixed_value.iloc[eq["index"]].value_counts(dropna=True)
            assert (
                len(valcounts) == 1
            ), "Equality constrained parameters cannot be fixed to different values."
            fixed_value.iloc[eq["index"]] = valcounts.idex[0]

    return fixed_value


def _consolidate_bounds_with_equality_constraints(equality_constraints, params):
    """consolidate bounds with equality constraints.

    Check that there are no incompatible bounds on equality constrained parameters and
    set the bounds for equal parameters to the strictest bound encountered on any of
    them.

    Args:
        equality_constraints (list): List of constraints of type "equality".
        params (pd.DataFrame): see :ref:`param`.

    Returns:
        processed_params (pd.DataFrame): Copy of params with stricter bounds.

    """
    pp = params.copy()
    lower = pp["lower"].copy()
    upper = pp["upper"].copy()
    for eq in equality_constraints:
        lower.iloc[eq["index"]] = lower.iloc[eq["index"]].max()
        upper.iloc[eq["index"]] = upper.iloc[eq["index"]].min()

    pp["lower"] = lower
    pp["upper"] = upper

    return pp


def _split_constraints(constraints, type_):
    filtered = [c for c in constraints if c["type"] == type_]
    rest = [c for c in constraints if c["type"] != type_]
    return filtered, rest


def simplify_covariance_and_sdcorr_constraints(constraints, params):
    cov_constraints, others = _split_constraints(constraints, "covariance")
    sdcorr_constraints, others = _split_constraints(others, "sdcorr")
    to_simplify = cov_constraints + sdcorr_constraints
    pp = params.copy()
    lower = pp["lower"].copy()
    upper = pp["upper"].copy()

    not_simplifyable = []
    for constr in to_simplify:
        dim = number_of_triangular_elements_to_dimension(len(constr["index"]))
        if constr["type"] == "covariance":
            diag_positions = [0] + np.cumsum(range(2, dim + 1)).tolist()
            diag_indices = np.array(constr["index"])[diag_positions].tolist()
            off_indices = [i for i in constr["index"] if i not in diag_positions]
        if constr["type"] == "sdcorr":
            diag_indices = constr["index"][:dim]
            off_indices = constr["index"][dim:]

        uncorrelated = False
        if params.iloc[off_indices]["_is_fixed_to_value"].all():
            if (params.iloc[off_indices]["_fixed_value"] == 0).all():
                uncorrelated = True

        if uncorrelated:
            lower.iloc[diag_indices] = np.maximum(0, lower.iloc[diag_indices])
        elif dim <= 2:
            lower.iloc[diag_indices] = np.maximum(0, lower.iloc[diag_indices])
            lower.iloc[off_indices] = -1
            upper.iloc[off_indices] = 1
        else:
            not_simplifyable.append(constr)

    pp["lower"] = lower
    pp["upper"] = upper

    return others + not_simplifyable, pp


def _plug_equality_constraints_into_selectors(
    equality_constraints, other_constraints, params
):
    """Rewrite all constraint in terms of free parameters.

    Only one parameter from a set of equality constrained parameters will actually
    be free. Which one is not important. We take the one with the lowest iloc.

    Then all other constraints have to be rewritten in terms of the free parameters.
    Once that is done, redundant constraints can be filtered out.

    Args:
        equality_constraints (list): List of constraints of type "equality".
        other_constraints (list): All other constraints.
        params (pd.DataFrame): see :ref:`params`.

    Returns:
        processed_constraints (list): List of processed non-equality constraints.
        processed_params (pd.DataFrame):

    """
    pp = params.copy()
    is_equal_to = pd.Series(index=params.index, data=-1, dtype=int)
    for eq in equality_constraints:
        is_equal_to.iloc[sorted(eq["index"][1:])] = eq["index"][0]
    pp["_post_replacements"] = is_equal_to
    pp["_is_fixed_to_other"] = is_equal_to >= 0
    helper = pp["_post_replacements"].reset_index(drop=True)
    replace_dict = helper[helper >= 0].to_dict()

    plugged_in = []
    for constr in other_constraints:
        new = constr.copy()
        new["index"] = pd.Series(constr["index"]).replace(replace_dict).tolist()
        plugged_in.append(new)

    linear_constraints, others = _split_constraints(plugged_in, "linear")

    processed_constraints = []
    for constr in others:
        if not _is_redundant(constr, processed_constraints):
            processed_constraints.append(constr)

    processed_constraints += linear_constraints

    return processed_constraints, pp


def _consolidate_linear_constraints(linear_constraints, processed_params):
    weights, right_hand_side = _transform_linear_constraints_to_pandas_objects(
        linear_constraints, processed_params
    )

    weights = _plug_equality_constraints_into_linear_weights(weights, processed_params)
    weights, right_hand_side = _plug_fixes_into_linear_weights_and_rhs(
        weights, right_hand_side, processed_params
    )

    involved_parameters = []
    for _, w in weights.iterrows():
        involved_parameters.append(set(w[w != 0].index))

    bundled_indices = _join_overlapping_lists(involved_parameters)

    constraints = []
    for involved_parameters in bundled_indices:
        w = weights[involved_parameters][
            (weights[involved_parameters] != 0).any(axis=1)
        ].copy(deep=True)
        rhs = right_hand_side.loc[w.index].copy(deep=True)
        w, rhs = _express_bounds_as_linear_constraints(w, rhs, processed_params)
        w, rhs = _rescale_linear_constraints(w, rhs)
        w, rhs = _drop_redundant_linear_constraints(w, rhs)
        _check_consolidated_weights(w, processed_params)
        rhs = _set_rhs_index(w, rhs, processed_params)
        to_internal, from_internal = _get_kernel_transformation_matrices(
            w, processed_params
        )
        constr = {
            "index": list(w.columns),
            "type": "linear",
            "to_internal": to_internal,
            "from_internal": from_internal,
            "right_hand_side": rhs,
        }
        constraints.append(constr)

    return constraints


def _transform_linear_constraints_to_pandas_objects(linear_constraints, params):
    """Collect information from the linear constraint dictionaries into pandas objects.

    Args:
        linear_constraints (list): List of constraints of type "linear".
        params (pd.DataFrame): see :ref:`params`

    Returns:
        weights (pd.DataFrame): DataFrame with one row per constraint and one column
            per parameter. Columns names are the ilocs of the parameters in params.
        rhs (pd.DataFrame): DataFrame with the columns "value", "lower" and
            "upper" that collects the right hand sides of the constraints.

    """
    all_weights, all_values, all_lbs, all_ubs = [], [], [], []
    for constr in linear_constraints:
        all_weights.append(constr["weights"])
        all_values.append(constr.get("value", np.nan))
        all_lbs.append(constr.get("lower", -np.inf))
        all_ubs.append(constr.get("upper", np.inf))

    weights = pd.concat(all_weights, axis=1).T.reset_index()
    weights = weights.reindex(columns=params.index).fillna(0)
    weights.columns = np.arange(len(weights.columns))
    values = pd.Series(all_values, name="value")
    lbs = pd.Series(all_lbs, name="lower")
    ubs = pd.Series(all_ubs, name="upper")
    rhs = pd.concat([values, lbs, ubs], axis=1)

    return weights, rhs


def _plug_equality_constraints_into_linear_weights(weights, processed_params):
    """Sum the weights of equality constrained parameters.

    The sum of the weights is then the new weight of the equality constrained parameter
    that is actually free. The weights of the other parameters are set to zero.

    """
    w = weights.T
    plugged_iloc = processed_params["_post_replacements"].reset_index(drop=True)
    plugged_iloc = plugged_iloc.where(plugged_iloc >= 0, np.arange(len(plugged_iloc)))
    w["plugged_iloc"] = plugged_iloc

    plugged_weights = w.groupby("plugged_iloc").sum()
    plugged_weights = plugged_weights.reindex(w.index).fillna(0).T

    return plugged_weights


def _plug_fixes_into_linear_weights_and_rhs(weights, right_hand_side, processed_params):
    """Set weights of fixed parameters to 0 and adjust right hand sides accordingly."""
    ilocs = pd.Series(data=range(len(processed_params)), index=processed_params.index)
    fixed_ilocs = ilocs[processed_params["_is_fixed_to_value"]].tolist()
    new_rhs = right_hand_side.copy()
    new_weights = weights.copy()

    if len(fixed_ilocs) > 0:
        fixed_values = processed_params.iloc[fixed_ilocs]["_fixed_value"].to_numpy()
        to_add = weights[fixed_ilocs] @ fixed_values
        for column in ["lower", "upper", "value"]:
            new_rhs[column] = new_rhs[column] + to_add
        for i in fixed_ilocs:
            new_weights[i] = 0

    return new_weights, new_rhs


def _express_bounds_as_linear_constraints(weights, right_hand_side, params):
    """Express bounds of linearly constrained params as linear constraint.

    In general it is easier to keep bounds separately from the constraints
    but in the case of linearly constrained parameters we need to express them as
    additional linear constraints to check compatibility and to choose the correct
    reparametrization.

    Args:
        weights (pd.DataFrame)
        right_hand_side (pd.DataFrame)

    Returns:
        extended_weights (pd.DataFrame)
        extended_rhs (pd.DataFrame)

    """
    additional_constraints = []
    for i in weights.columns:
        new = {}
        if np.isfinite(params.iloc[i]["lower"]):
            new["lower"] = params.iloc[i]["lower"]
        if np.isfinite(params.iloc[i]["upper"]):
            new["upper"] = params.iloc[i]["upper"]
        if new != {}:
            new["weights"] = pd.Series([1], name="w", index=params.iloc[[i]].index)
            additional_constraints.append(new)

    if len(additional_constraints) > 0:

        new_weights, new_rhs = _transform_linear_constraints_to_pandas_objects(
            additional_constraints, params
        )
        new_weights = new_weights[weights.columns]

        extended_weights = pd.concat([weights, new_weights]).reset_index(drop=True)
        extended_rhs = pd.concat([right_hand_side, new_rhs]).reset_index(drop=True)
    else:
        extended_weights, extended_rhs = weights, right_hand_side

    return extended_weights, extended_rhs


def _rescale_linear_constraints(weights, right_hand_side):
    """Rescale rows in weights such that the first nonzero element equals one.

    This will make it easier to detect redundant rows.

    """
    first_nonzero = weights.replace(0, np.nan).bfill(1).iloc[:, 0]
    scaling_factor = 1 / first_nonzero.to_numpy().reshape(-1, 1)
    weights = scaling_factor * weights
    scaled_rhs = scaling_factor * right_hand_side
    rhs = scaled_rhs.copy()
    rhs["lower"] = scaled_rhs["lower"].where(
        scaling_factor.flatten() > 0, scaled_rhs["upper"]
    )
    rhs["upper"] = scaled_rhs["upper"].where(
        scaling_factor.flatten() > 0, scaled_rhs["lower"]
    )

    return weights, rhs


def _drop_redundant_linear_constraints(weights, right_hand_side):
    weights["dupl_group"] = weights.groupby(list(weights.columns)).grouper.group_info[0]
    right_hand_side["dupl_group"] = weights["dupl_group"]
    weights.set_index("dupl_group", inplace=True)

    new_weights = weights.drop_duplicates()

    def _consolidate_fix(x):
        vc = x.value_counts(dropna=True)
        if len(vc) == 0:
            return np.nan
        elif len(vc) == 1:
            return vc.index[0]
        else:
            raise ValueError

    ub = right_hand_side.groupby("dupl_group")["upper"].min()
    lb = right_hand_side.groupby("dupl_group")["lower"].max()
    fix = right_hand_side.groupby("dupl_group")["value"].apply(_consolidate_fix)

    # remove the bounds for fixed parameters
    ub = ub.where(fix.isnull(), np.inf)
    lb = lb.where(fix.isnull(), -np.inf)

    new_rhs = pd.concat([lb, ub, fix], axis=1, names=["lower", "upper", "value"])
    new_rhs = new_rhs.reindex(weights.index)

    return new_weights, new_rhs


def _check_consolidated_weights(weights, processed_params):
    n_constraints, n_params = weights.shape

    msg_too_many = (
        "Too many linear constraints. There can be at most as many linear constraints"
        "as involved parameters with non-zero weights.\n"
    )

    msg_rank = "The weights for linear constraints must be linearly independent.\n"

    msg_general = (
        "The error occurred for constraints on the following parameters:\n{}\n with "
        "weighting matrix:\n{}\nIt is possible that you did not specify those "
        "constraints as linear constraints but as bounds, fixes, increasing or "
        "decreasing constraints."
    )

    ind = processed_params.iloc[weights.columns].index

    if n_constraints > n_params:
        raise ValueError(msg_too_many + msg_general.format(ind, weights))

    if np.linalg.matrix_rank(weights) < n_constraints:
        raise ValueError(msg_rank + msg_general.format(ind, weights))


def _set_rhs_index(weights, right_hand_side, params):
    ind = params.iloc[weights.columns[-len(weights) :]].index
    new_rhs = pd.DataFrame(
        right_hand_side.to_numpy(), columns=right_hand_side.columns, index=ind
    )

    return new_rhs


def _get_kernel_transformation_matrices(weights, params):
    n_constraints, n_params = weights.shape

    identity = np.eye(n_params)

    i = 0
    filled_weights = weights
    while len(filled_weights) < n_params:
        candidate = np.vstack([identity[i], filled_weights])
        if np.linalg.matrix_rank(candidate) == len(candidate):
            filled_weights = candidate
        i += 1

    k = n_params - n_constraints

    filled_weights[:k] = filled_weights[:k][::-1]

    to_internal = filled_weights
    from_internal = np.linalg.inv(to_internal)

    return to_internal, from_internal


def _is_redundant(candidate, others):
    if len(others) == 0:
        is_redundant = False
    else:
        same_type = _split_constraints(others, candidate["type"])
        duplicates = [c for c in same_type if c["index"] == candidate["index"]]
        is_redundant = len(duplicates) > 0

    return is_redundant
