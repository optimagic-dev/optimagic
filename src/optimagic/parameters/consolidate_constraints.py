"""Functions to consolidate user provided constraints.

Consolidation means that redundant constraints are dropped and other constraints are
collected in meaningful bundles.

Check the module docstring of process_constraints for naming conventions.

"""

from dataclasses import dataclass

import numpy as np

from optimagic.exceptions import InvalidConstraintError
from optimagic.utilities import (
    fast_numpy_full,
    number_of_triangular_elements_to_dimension,
)


@dataclass(frozen=True)
class LinearRightHandSide:
    """Right hand side of a consolidated linear constraint.

    All arrays have one entry per consolidated constraint, aligned with the rows of
    the consolidated weights.

    Attributes:
        values: Value at which the weighted sum is fixed; nan where it is not fixed.
        lower_bounds: Lower bound on the weighted sum; -inf where there is none.
        upper_bounds: Upper bound on the weighted sum; inf where there is none.

    """

    values: np.ndarray
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray

    def __len__(self) -> int:
        return len(self.values)


def consolidate_constraints(
    constraints, parvec, lower_bounds, upper_bounds, param_names
):
    """Consolidate constraints with each other and remove redundant ones.

    Args:
        constraints (list): List with constraint dictionaries. It is assumed that
            the selectors are already processed, increasing and decreasing
            constraints have been rewritten as linear constraints and
            pairwise_equality constraints have been rewritten as equality constraints.
        parvec (np.ndarray): 1d numpy array with parameters.
        lower_bounds (np.ndarray | None): 1d numpy array with lower_bounds
        upper_bounds (np.ndarray | None): 1d numpy array with upper_bounds
        param_names (list): Names of parameters. Used for error messages.

    Returns:
        list: This contains processed version of all
            constraints that require an actual kernel transformation. The information
            on all other constraints is subsumed in pp.
        dict: Dict of 1d numpy arrays with information about non-transforming
            constraints.

    """
    # None-valued bounds are handled by instantiating them as an -inf and inf array. In
    # the future, this should be handled more gracefully.
    if lower_bounds is None:
        lower_bounds = fast_numpy_full(len(parvec), fill_value=-np.inf)
    if upper_bounds is None:
        upper_bounds = fast_numpy_full(len(parvec), fill_value=np.inf)

    raw_eq, other_constraints = _split_constraints(constraints, "equality")
    equality_constraints = _consolidate_equality_constraints(raw_eq)

    fixed_constraints, other_constraints = _split_constraints(
        other_constraints, "fixed"
    )
    fixed_value = _consolidate_fixes_with_equality_constraints(
        fixed_constraints, equality_constraints, parvec
    )

    constr_info = {
        "fixed_values": fixed_value,
        "is_fixed_to_value": np.isfinite(fixed_value),
    }

    other_constraints = [
        c
        for c in other_constraints
        if not constr_info["is_fixed_to_value"][c["index"]].all()
    ]

    (
        other_constraints,
        lower_bounds,
        upper_bounds,
    ) = simplify_covariance_and_sdcorr_constraints(
        constraints=other_constraints,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        is_fixed_to_value=constr_info["is_fixed_to_value"],
        fixed_value=constr_info["fixed_values"],
    )

    lower_bounds, upper_bounds = _consolidate_bounds_with_equality_constraints(
        equality_constraints,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )

    constr_info["lower_bounds"] = lower_bounds
    constr_info["upper_bounds"] = upper_bounds

    (
        other_constraints,
        post_replacements,
        is_fixed_to_other,
    ) = _plug_equality_constraints_into_selectors(
        equality_constraints, other_constraints, n_params=len(parvec)
    )

    constr_info["post_replacements"] = post_replacements
    constr_info["is_fixed_to_other"] = is_fixed_to_other

    linear_constraints, other_constraints = _split_constraints(
        other_constraints, "linear"
    )

    if len(linear_constraints) > 0:
        linear_constraints = _consolidate_linear_constraints(
            params_vec=parvec,
            linear_constraints=linear_constraints,
            constr_info=constr_info,
            param_names=param_names,
        )

    constraints = other_constraints + linear_constraints

    return constraints, constr_info


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
            constraint. It is assumed that the selectors were already processed.

    Returns:
        list: List of consolidated equality constraints.

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
            bundles.append(sorted(new_candidates[0]))
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

    return [new_first, *new_others]


def _consolidate_fixes_with_equality_constraints(
    fixed_constraints, equality_constraints, parvec
):
    """Consolidate fixes with equality constraints.

    If any equality constrained parameter is fixed, all of the parameters that are
    equal to it have to be fixed to the same value.

    Args:
        fixed_constraints (list): List of constrains of type "fixed".
        equality_constraints (list): List of constraints of type "equality".
        parvec (np.ndarray): 1d numpy array with parameters.

    Returns:
        fixed_value (pd.Series): Series with the fixed value for all parameters that
            are fixed and np.nan everywhere else. Has the same index as params.

    """
    fixed_value = np.full(len(parvec), np.nan)
    for fix in fixed_constraints:
        fixed_value[fix["index"]] = fix.get("value", parvec[fix["index"]])

    for eq in equality_constraints:
        if np.isfinite(fixed_value[eq["index"]]).any():
            valcounts = _unique_values(fixed_value[eq["index"]])
            assert len(valcounts) == 1, (
                "Equality constrained parameters cannot be fixed to different values."
            )
            fixed_value[eq["index"]] = valcounts[0]

    return fixed_value


def _consolidate_bounds_with_equality_constraints(
    equality_constraints, lower_bounds, upper_bounds
):
    """Consolidate bounds with equality constraints.

    Check that there are no incompatible bounds on equality constrained parameters and
    set the bounds for equal parameters to the strictest bound encountered on any of
    them.

    Args:
        equality_constraints (list): List of constraints of type "equality".
        lower_bounds (np.ndarray): Lower bounds for parameters.
        upper_bounds (np.ndarray): Upper bounds for parameters.

    Returns:
        np.ndarray: 1d array with lower bounds
        np.ndarray: 1d array with upper bounds

    """
    lower = lower_bounds.copy()
    upper = upper_bounds.copy()
    for eq in equality_constraints:
        lower[eq["index"]] = lower[eq["index"]].max()
        upper[eq["index"]] = upper[eq["index"]].min()

    return lower, upper


def _split_constraints(constraints, type_):
    """Split list of constraints in two list.

    The first list contains all constraints of type and the second the rest.

    """
    filtered = [c for c in constraints if c["type"] == type_]
    rest = [c for c in constraints if c["type"] != type_]
    return filtered, rest


def simplify_covariance_and_sdcorr_constraints(
    constraints,
    lower_bounds,
    upper_bounds,
    is_fixed_to_value,
    fixed_value,
):
    """Enforce covariance and sdcorr constraints by bounds if possible.

    This is possible if the dimension is <= 2 or all covariances are fexd to 0.

    """
    cov_constraints, others = _split_constraints(constraints, "covariance")
    sdcorr_constraints, others = _split_constraints(others, "sdcorr")
    to_simplify = cov_constraints + sdcorr_constraints
    lower = lower_bounds.copy()
    upper = upper_bounds.copy()

    not_simplifyable = []
    for constr in to_simplify:
        dim = number_of_triangular_elements_to_dimension(len(constr["index"]))
        if constr["type"] == "covariance":
            diag_positions = [0, *np.cumsum(range(2, dim + 1)).tolist()]
            diag_indices = np.array(constr["index"])[diag_positions].tolist()
            off_indices = np.delete(constr["index"], diag_positions).tolist()
        if constr["type"] == "sdcorr":
            diag_indices = constr["index"][:dim]
            off_indices = constr["index"][dim:]

        uncorrelated = False
        if is_fixed_to_value[off_indices].all():
            if (fixed_value[off_indices] == 0).all():
                uncorrelated = True

        if uncorrelated:
            lower[diag_indices] = np.maximum(0, lower[diag_indices])
        elif dim <= 2 and constr["type"] == "sdcorr":
            lower[diag_indices] = np.maximum(0, lower[diag_indices])
            lower[off_indices] = -1
            upper[off_indices] = 1
        else:
            not_simplifyable.append(constr)

    return others + not_simplifyable, lower, upper


def _plug_equality_constraints_into_selectors(
    equality_constraints, other_constraints, n_params
):
    """Rewrite all constraint in terms of free parameters.

    Only one parameter from a set of equality constrained parameters will actually
    be free. Which one is not important. We take the one with the lowest iloc.

    Then all other constraints have to be rewritten in terms of the free parameters.
    Once that is done, redundant constraints can be filtered out.

    Args:
        equality_constraints (list): List of constraints of type "equality".
        other_constraints (list): All other constraints.
        n_params (int): Number of parameters.

    Returns:
        list: List of processed non-equality constraints.
        np.ndarray: post_replacements
        np.ndarray: is_fixed_to_other

    """
    is_equal_to = np.full(n_params, -1)
    for eq in equality_constraints:
        is_equal_to[sorted(eq["index"])[1:]] = sorted(eq["index"])[0]
    post_replacements = is_equal_to.astype(int)
    is_fixed_to_other = is_equal_to >= 0

    plugged_in = []
    for constr in other_constraints:
        # linear constraints keep their original index; their equality plugging
        # sums the weights of equal parameters in _consolidate_linear_constraints,
        # which requires the weights at their original positions
        if constr["type"] == "linear":
            plugged_in.append(constr)
            continue
        new = constr.copy()
        index = np.asarray(constr["index"])
        replacements = post_replacements[index]
        new["index"] = np.where(replacements >= 0, replacements, index).tolist()
        plugged_in.append(new)

    linear_constraints, others = _split_constraints(plugged_in, "linear")

    pc = []
    for constr in others:
        if not _is_redundant(constr, pc):
            pc.append(constr)

    pc += linear_constraints

    return pc, post_replacements, is_fixed_to_other


def _consolidate_linear_constraints(
    params_vec, linear_constraints, constr_info, param_names
):
    """Consolidate linear constraints.

    Consolidation entails the following steps:
    - Plugging fixes and equality constraints into the linear constraints
    - Collect weights of those constraints that overlap into weight DataFrames
    - Collect corresponding right hand sides (bounds or values) in DataFrames
    - Express box constraints of parameters involved in linear constraints as
      additional linear constraints.
    - Rescale the weights for easier detection of linear dependence
    - Drop redundant constraints
    - Check compatibility of constraints
    - Construct a list of consolidated constraint dictionaries that contain
        all matrices needed for the kernel transformations.

    Args:
        params_vec (np.ndarray): 1d numpy array wtih parameters
        linear_constraints (list): Linear constraints that already have processed
            weights and selector fields.
        constr_info (dict): Dict with information about constraints.
        param_names (list): Parameter names. Used for error messages.

    Returns:
        list: Processed and consolidated linear constraints.

    """
    weights, values, lbs, ubs = _build_linear_system(
        linear_constraints, n_params=len(params_vec)
    )

    weights = _plug_equalities_into_linear_weights(
        weights, constr_info["post_replacements"]
    )
    weights, values, lbs, ubs = _plug_fixes_into_linear_system(
        weights,
        values,
        lbs,
        ubs,
        constr_info["is_fixed_to_value"],
        constr_info["fixed_values"],
    )

    involved_parameters = [set(np.flatnonzero(row).tolist()) for row in weights]

    bundled_indices = _join_overlapping_lists(involved_parameters)

    pc = []
    for bundle in bundled_indices:
        columns = [int(i) for i in bundle]
        row_mask = (weights[:, columns] != 0).any(axis=1)
        w = weights[row_mask][:, columns]
        v, lb, ub = values[row_mask], lbs[row_mask], ubs[row_mask]
        w, v, lb, ub = _append_bound_rows(
            w,
            v,
            lb,
            ub,
            columns,
            constr_info["lower_bounds"],
            constr_info["upper_bounds"],
        )
        w, v, lb, ub = _rescale_rows(w, v, lb, ub)
        w, v, lb, ub = _drop_redundant_rows(w, v, lb, ub)
        _check_consolidated_weights(w, columns, param_names)
        to_internal, from_internal = _get_kernel_transformation_matrices(w)
        constr = {
            "index": columns,
            "type": "linear",
            "to_internal": to_internal,
            "from_internal": from_internal,
            "right_hand_side": LinearRightHandSide(
                values=v, lower_bounds=lb, upper_bounds=ub
            ),
        }
        pc.append(constr)

    return pc


def _build_linear_system(linear_constraints, n_params):
    """Collect the linear constraints into a dense weight matrix and rhs arrays.

    Args:
        linear_constraints (list): List of constraints of type "linear" with an
            "index" field and weights that are already aligned with the index.
        n_params (int): Number of parameters.

    Returns:
        weights (np.ndarray): Array of shape (n_constraints, n_params) with one row
            per constraint.
        values (np.ndarray): Values at which the weighted sums are fixed; nan where
            they are not fixed.
        lbs (np.ndarray): Lower bounds on the weighted sums; -inf where absent.
        ubs (np.ndarray): Upper bounds on the weighted sums; inf where absent.

    """
    n_rows = len(linear_constraints)
    weights = np.zeros((n_rows, n_params))
    values = np.full(n_rows, np.nan)
    lbs = np.full(n_rows, -np.inf)
    ubs = np.full(n_rows, np.inf)
    for row, constr in enumerate(linear_constraints):
        index = np.asarray(constr["index"], dtype=np.int64)
        weights[row, index] = constr["weights"]
        values[row] = constr.get("value", np.nan)
        lbs[row] = constr.get("lower_bound", -np.inf)
        ubs[row] = constr.get("upper_bound", np.inf)

    return weights, values, lbs, ubs


def _plug_equalities_into_linear_weights(weights, post_replacements):
    """Sum the weights of equality constrained parameters.

    The sum of the weights is then the new weight of the equality constrained parameter
    that is actually free. The weights of the other parameters are set to zero.

    Args:
        weights (np.ndarray): Weight matrix of shape (n_constraints, n_params).
        post_replacements (np.ndarray): For each parameter the position of the
            parameter it is equal to; -1 for parameters that are not replaced.

    Returns:
        np.ndarray: The plugged in weight matrix.

    """
    followers = np.flatnonzero(post_replacements >= 0)
    if len(followers) == 0:
        return weights

    out = weights.copy()
    representatives = post_replacements[followers]
    np.add.at(out, (slice(None), representatives), out[:, followers])
    out[:, followers] = 0

    return out


def _plug_fixes_into_linear_system(
    weights, values, lbs, ubs, is_fixed_to_value, fixed_value
):
    """Set weights of fixed parameters to 0 and adjust right hand sides accordingly.

    Args:
        weights (np.ndarray): Weight matrix of shape (n_constraints, n_params).
        values (np.ndarray): Values at which the weighted sums are fixed.
        lbs (np.ndarray): Lower bounds on the weighted sums.
        ubs (np.ndarray): Upper bounds on the weighted sums.
        is_fixed_to_value (np.ndarray): Boolean array of length n_params.
        fixed_value (np.ndarray): Array of length n_params with the fixed values.

    Returns:
        The adjusted (weights, values, lbs, ubs).

    """
    if not is_fixed_to_value.any():
        return weights, values, lbs, ubs

    fixed_contribution = weights[:, is_fixed_to_value] @ fixed_value[is_fixed_to_value]
    new_weights = weights.copy()
    new_weights[:, is_fixed_to_value] = 0

    return (
        new_weights,
        values - fixed_contribution,
        lbs - fixed_contribution,
        ubs - fixed_contribution,
    )


def _append_bound_rows(w, v, lb, ub, columns, lower_bounds, upper_bounds):
    """Express bounds of linearly constrained params as additional linear constraints.

    In general it is easier to keep bounds separately from the constraints
    but in the case of linearly constrained parameters we need to express them as
    additional linear constraints to check compatibility and to choose the correct
    reparametrization.

    Args:
        w (np.ndarray): Weight matrix of one bundle, shape (n_rows, len(columns)).
        v (np.ndarray): Values at which the weighted sums are fixed.
        lb (np.ndarray): Lower bounds on the weighted sums.
        ub (np.ndarray): Upper bounds on the weighted sums.
        columns (list): The parameter positions the bundle applies to, sorted.
        lower_bounds (np.ndarray): Lower bounds of all parameters.
        upper_bounds (np.ndarray): Upper bounds of all parameters.

    Returns:
        The extended (w, v, lb, ub).

    """
    extra_rows, extra_lb, extra_ub = [], [], []
    for pos, i in enumerate(columns):
        has_lower = np.isfinite(lower_bounds[i])
        has_upper = np.isfinite(upper_bounds[i])
        if has_lower or has_upper:
            row = np.zeros(len(columns))
            row[pos] = 1
            extra_rows.append(row)
            extra_lb.append(lower_bounds[i] if has_lower else -np.inf)
            extra_ub.append(upper_bounds[i] if has_upper else np.inf)

    if extra_rows:
        w = np.vstack([w, extra_rows])
        v = np.concatenate([v, np.full(len(extra_rows), np.nan)])
        lb = np.concatenate([lb, extra_lb])
        ub = np.concatenate([ub, extra_ub])

    return w, v, lb, ub


def _rescale_rows(w, v, lb, ub):
    """Rescale rows in w such that the first nonzero element equals one.

    This will make it easier to detect redundant rows. If a row is rescaled with a
    negative factor, its lower and upper bound switch roles.

    Args:
        w (np.ndarray): Weight matrix of one bundle.
        v (np.ndarray): Values at which the weighted sums are fixed.
        lb (np.ndarray): Lower bounds on the weighted sums.
        ub (np.ndarray): Upper bounds on the weighted sums.

    Returns:
        The rescaled (w, v, lb, ub).

    """
    first_nonzero_pos = np.argmax(w != 0, axis=1)
    first_nonzero = w[np.arange(len(w)), first_nonzero_pos]
    factor = 1 / first_nonzero

    new_w = w * factor.reshape(-1, 1)
    new_v = v * factor
    scaled_lb = lb * factor
    scaled_ub = ub * factor
    negative = factor < 0
    new_lb = np.where(negative, scaled_ub, scaled_lb)
    new_ub = np.where(negative, scaled_lb, scaled_ub)

    return new_w, new_v, new_lb, new_ub


def _drop_redundant_rows(w, v, lb, ub):
    """Drop linear constraints that are implied by other linear constraints.

    This is not yet very smart. We just merge rows with identical weights, keeping
    the strictest bounds. Rows with a fixed value keep the value and lose their
    bounds. The first occurrence order of unique rows is preserved because the
    internal parametrization depends on the row order.

    Args:
        w (np.ndarray): Weight matrix of one bundle.
        v (np.ndarray): Values at which the weighted sums are fixed.
        lb (np.ndarray): Lower bounds on the weighted sums.
        ub (np.ndarray): Upper bounds on the weighted sums.

    Returns:
        The deduplicated (w, v, lb, ub).

    Raises:
        ValueError: If identical weighted sums are fixed to different values.

    """
    # normalize -0.0 to 0.0 so that rows only differing in the sign of a zero (which
    # happens when a row is rescaled with a negative factor) are grouped together
    normalized = w + 0.0

    unique_rows, first_positions, inverse = np.unique(
        normalized, axis=0, return_index=True, return_inverse=True
    )
    order = np.argsort(first_positions)

    n_groups = len(unique_rows)
    new_w = unique_rows[order]
    new_v = np.full(n_groups, np.nan)
    new_lb = np.full(n_groups, -np.inf)
    new_ub = np.full(n_groups, np.inf)

    for out_position, group in enumerate(order):
        members = inverse == group
        distinct_values = np.unique(v[members][np.isfinite(v[members])])
        if len(distinct_values) > 1:
            raise ValueError
        elif len(distinct_values) == 1:
            # bounds are dropped for fixed weighted sums
            new_v[out_position] = distinct_values[0]
        else:
            new_lb[out_position] = lb[members].max()
            new_ub[out_position] = ub[members].min()

    return new_w, new_v, new_lb, new_ub


def _check_consolidated_weights(weights, columns, param_names):
    """Check the rank condition on the linear weights."""
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
    relevant_names = [param_names[i] for i in columns]

    if n_constraints > n_params:
        raise InvalidConstraintError(
            msg_too_many + msg_general.format(relevant_names, weights)
        )

    if np.linalg.matrix_rank(weights) < n_constraints:
        raise InvalidConstraintError(
            msg_rank + msg_general.format(relevant_names, weights)
        )


def _get_kernel_transformation_matrices(weights):
    """Construct the m matrix for the kernel transformations.

    See :ref:`linear_constraint_implementation` for details.

    Args:
        weights (np.ndarray): Weight matrix of a linear constraint bundle.

    """
    n_constraints, n_params = weights.shape

    identity = np.eye(n_params)

    i = 0
    filled_weights = weights.copy()
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
    """Check if a constraint is redundant given other constraints.

    Applicable to all but linear constraints.

    """
    assert candidate["type"] != "linear"
    if len(others) == 0:
        is_redundant = False
    else:
        same_type, _ = _split_constraints(others, candidate["type"])
        duplicates = [c for c in same_type if c["index"] == candidate["index"]]
        is_redundant = len(duplicates) > 0

    return is_redundant


def _unique_values(arr, dropna=True):
    if dropna:
        arr = arr[np.isfinite(arr)]
    return list(set(arr.tolist()))
