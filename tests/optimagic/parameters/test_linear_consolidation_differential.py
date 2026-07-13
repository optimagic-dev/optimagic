"""Differential test for the numpy rewrite of the linear constraint consolidation.

Runs the rewritten _consolidate_linear_constraints against a verbatim copy of the
original pandas implementation on randomized scenarios and asserts that the outputs
are numerically identical. The scenarios cover overlapping constraint bundles,
interactions with fixes and equality constraints, bounds on linearly constrained
parameters, negative and duplicate weights, and conflicting fixed values.

This test is deleted together with _legacy_linear_consolidation.py once the
refactored constraints pipeline is complete.

"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from optimagic.exceptions import InvalidConstraintError
from optimagic.parameters.consolidate_constraints import (
    _consolidate_linear_constraints,
)
from optimagic.utilities import get_rng

# ======================================================================================
# Verbatim copy of the pandas based linear consolidation (pre-rewrite).
# Deleted together with this differential test once the refactoring is complete.
# Do not fix or improve anything in here.
# ======================================================================================


def consolidate_linear_constraints_legacy(
    params_vec, linear_constraints, constr_info, param_names
):
    """Original pandas implementation of _consolidate_linear_constraints.

    The constraints must have their weights as pd.Series indexed by the flat
    positions of the selected parameters (the old ``_process_linear_weights``
    format).

    """
    weights, right_hand_side = _transform_linear_constraints_to_pandas_objects(
        linear_constraints, n_params=len(params_vec)
    )

    weights = _plug_equality_constraints_into_linear_weights(
        weights, constr_info["post_replacements"]
    )
    weights, right_hand_side = _plug_fixes_into_linear_weights_and_rhs(
        weights,
        right_hand_side,
        constr_info["is_fixed_to_value"],
        constr_info["fixed_values"],
    )

    involved_parameters = [set(w[w != 0].index) for _, w in weights.iterrows()]

    bundled_indices = _join_overlapping_lists(involved_parameters)

    pc = []
    for involved_parameters in bundled_indices:
        w = weights[involved_parameters][
            (weights[involved_parameters] != 0).any(axis=1)
        ].copy(deep=True)
        rhs = right_hand_side.loc[w.index].copy(deep=True)
        w, rhs = _express_bounds_as_linear_constraints(
            w, rhs, constr_info["lower_bounds"], constr_info["upper_bounds"]
        )
        w, rhs = _rescale_linear_constraints(w, rhs)
        w, rhs = _drop_redundant_linear_constraints(w, rhs)
        _check_consolidated_weights(w, param_names=param_names)
        to_internal, from_internal = _get_kernel_transformation_matrices(w)
        constr = {
            "index": list(w.columns),
            "type": "linear",
            "to_internal": to_internal,
            "from_internal": from_internal,
            "right_hand_side": rhs,
        }
        pc.append(constr)

    return pc


def _join_overlapping_lists(candidates):
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
    first = set(indices[0])
    new_first = first
    new_others = []
    for idx in indices[1:]:
        if len(first.intersection(idx)) > 0:
            new_first = new_first.union(idx)
        else:
            new_others.append(idx)

    return [new_first, *new_others]


def _transform_linear_constraints_to_pandas_objects(linear_constranits, n_params):
    all_weights, all_values, all_lbs, all_ubs = [], [], [], []
    for constr in linear_constranits:
        all_weights.append(constr["weights"])
        all_values.append(constr.get("value", np.nan))
        all_lbs.append(constr.get("lower_bound", -np.inf))
        all_ubs.append(constr.get("upper_bound", np.inf))

    weights = pd.concat(all_weights, axis=1).T.reset_index()
    weights = weights.reindex(columns=np.arange(n_params)).fillna(0)
    values = pd.Series(all_values, name="value")
    lbs = pd.Series(all_lbs, name="lower_bound")
    ubs = pd.Series(all_ubs, name="upper_bound")
    rhs = pd.concat([values, lbs, ubs], axis=1)

    return weights, rhs


def _plug_equality_constraints_into_linear_weights(weights, post_replacements):
    w = weights.T
    plugged_iloc = pd.Series(post_replacements)
    plugged_iloc = plugged_iloc.where(plugged_iloc >= 0, np.arange(len(plugged_iloc)))
    w["plugged_iloc"] = plugged_iloc

    plugged_weights = w.groupby("plugged_iloc").sum()
    plugged_weights = plugged_weights.reindex(w.index).fillna(0).T

    return plugged_weights


def _plug_fixes_into_linear_weights_and_rhs(
    weights, rhs, is_fixed_to_value, fixed_value
):
    ilocs = np.arange(len(fixed_value))
    fixed_ilocs = ilocs[is_fixed_to_value].tolist()
    new_rhs = rhs.copy()
    new_weights = weights.copy()

    if len(fixed_ilocs) > 0:
        fixed_values = fixed_value[fixed_ilocs]
        fixed_contribution = weights[fixed_ilocs] @ fixed_values
        for column in ["lower_bound", "upper_bound", "value"]:
            new_rhs[column] = new_rhs[column] - fixed_contribution
        for i in fixed_ilocs:
            new_weights[i] = 0

    return new_weights, new_rhs


def _express_bounds_as_linear_constraints(weights, rhs, lower, upper):
    additional_pc = []
    for i in weights.columns:
        new = {}
        if np.isfinite(lower[i]):
            new["lower_bound"] = lower[i]
        if np.isfinite(upper[i]):
            new["upper_bound"] = upper[i]
        if new != {}:
            new["weights"] = pd.Series([1], name="w", index=[i])
            additional_pc.append(new)

    if len(additional_pc) > 0:
        new_weights, new_rhs = _transform_linear_constraints_to_pandas_objects(
            additional_pc, len(lower)
        )
        new_weights = new_weights[weights.columns]

        extended_weights = pd.concat([weights, new_weights]).reset_index(drop=True)
        extended_rhs = pd.concat([rhs, new_rhs]).reset_index(drop=True)
    else:
        extended_weights, extended_rhs = weights, rhs

    return extended_weights, extended_rhs


def _rescale_linear_constraints(weights, rhs):
    first_nonzero = weights.replace(0, np.nan).bfill(axis=1).iloc[:, 0]
    scaling_factor = 1 / first_nonzero.to_numpy().reshape(-1, 1)
    new_weights = scaling_factor * weights
    scaled_rhs = scaling_factor * rhs
    new_rhs = scaled_rhs.copy()
    new_rhs["lower_bound"] = scaled_rhs["lower_bound"].where(
        scaling_factor.flatten() > 0, scaled_rhs["upper_bound"]
    )
    new_rhs["upper_bound"] = scaled_rhs["upper_bound"].where(
        scaling_factor.flatten() > 0, scaled_rhs["lower_bound"]
    )

    return new_weights, new_rhs


def _drop_redundant_linear_constraints(weights, rhs):
    weights["dupl_group"] = weights.groupby(list(weights.columns)).ngroup()
    rhs["dupl_group"] = weights["dupl_group"]
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

    ub = rhs.groupby("dupl_group")["upper_bound"].min()
    lb = rhs.groupby("dupl_group")["lower_bound"].max()
    fix = rhs.groupby("dupl_group")["value"].apply(_consolidate_fix)

    # remove the bounds for fixed parameters
    ub = ub.where(fix.isnull(), np.inf)
    lb = lb.where(fix.isnull(), -np.inf)

    new_rhs = pd.concat(
        [lb, ub, fix], axis=1, names=["lower_bound", "upper_bound", "value"]
    )
    new_rhs = new_rhs.reindex(new_weights.index)

    return new_weights, new_rhs


def _check_consolidated_weights(weights, param_names):
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
    relevant_names = [param_names[i] for i in weights.columns]

    if n_constraints > n_params:
        raise InvalidConstraintError(
            msg_too_many + msg_general.format(relevant_names, weights)
        )

    if np.linalg.matrix_rank(weights) < n_constraints:
        raise InvalidConstraintError(
            msg_rank + msg_general.format(relevant_names, weights)
        )


def _get_kernel_transformation_matrices(weights):
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


def _draw_scenario(rng):
    """Draw a random consolidation scenario.

    Returns the params vector, the constr_info dict expected by the consolidation
    functions and a list of scenario constraints, each a dict with index, weights
    and right hand side entries.

    """
    n_params = int(rng.integers(4, 11))

    # equality sets: disjoint sets of parameters that are equal to each other
    post_replacements = np.full(n_params, -1, dtype=int)
    available = list(range(n_params))
    equality_sets = []
    for _ in range(int(rng.integers(0, 3))):
        if len(available) < 2:
            break
        size = int(rng.integers(2, min(4, len(available) + 1)))
        members = sorted(rng.choice(available, size=size, replace=False).tolist())
        available = [i for i in available if i not in members]
        equality_sets.append(members)
        for follower in members[1:]:
            post_replacements[follower] = members[0]

    # fixes; a fix on any member of an equality set propagates to all members, as
    # guaranteed by the consolidation steps that run before the linear consolidation.
    # At least two parameters must stay free, otherwise no valid linear constraint
    # can be drawn below.
    while True:
        fixed_values = np.full(n_params, np.nan)
        for i in range(n_params):
            if rng.uniform() < 0.15:
                fixed_values[i] = rng.uniform(-2, 2)
        for members in equality_sets:
            member_values = fixed_values[members]
            if np.isfinite(member_values).any():
                fixed_values[members] = member_values[np.isfinite(member_values)][0]
        is_fixed_to_value = np.isfinite(fixed_values)
        if (~is_fixed_to_value).sum() >= 2:
            break

    # bounds
    lower_bounds = np.full(n_params, -np.inf)
    upper_bounds = np.full(n_params, np.inf)
    for i in range(n_params):
        if rng.uniform() < 0.3:
            lower_bounds[i] = rng.uniform(-3, 0)
        if rng.uniform() < 0.3:
            upper_bounds[i] = rng.uniform(1, 4)

    # linear constraints
    constraints = []
    n_constraints = int(rng.integers(1, 5))
    while len(constraints) < n_constraints:
        if constraints and rng.uniform() < 0.25:
            # a scaled (possibly conflicting) copy of an earlier constraint
            template = constraints[int(rng.integers(0, len(constraints)))]
            factor = float(rng.choice([-1.5, -1.0, 2.0]))
            new = {
                "index": template["index"],
                "weights": factor * template["weights"],
            }
            for key in ("value", "lower_bound", "upper_bound"):
                if key in template:
                    perturbation = (
                        float(rng.choice([0.0, 0.5])) if key == "value" else 0.0
                    )
                    new[key] = factor * template[key] + perturbation
            constraints.append(new)
            continue

        size = int(rng.integers(2, min(5, n_params + 1)))
        index = np.sort(rng.choice(n_params, size=size, replace=False))
        if is_fixed_to_value[index].all():
            # constraints on only fixed parameters are dropped before the linear
            # consolidation
            continue
        signs = rng.choice([-1.0, 1.0], size=size)
        weights = signs * rng.uniform(0.1, 2.0, size=size)
        new = {"index": index, "weights": weights}
        rhs_case = rng.choice(["value", "lower", "upper", "both"])
        if rhs_case == "value":
            new["value"] = float(rng.uniform(-3, 3))
        if rhs_case in ("lower", "both"):
            new["lower_bound"] = float(rng.uniform(-3, 0))
        if rhs_case in ("upper", "both"):
            new["upper_bound"] = float(rng.uniform(0, 3))
        constraints.append(new)

    params_vec = rng.uniform(-1, 1, size=n_params)
    constr_info = {
        "post_replacements": post_replacements,
        "is_fixed_to_value": is_fixed_to_value,
        "fixed_values": fixed_values,
        "lower_bounds": lower_bounds,
        "upper_bounds": upper_bounds,
    }
    return params_vec, constr_info, constraints


def _to_legacy_format(constraints):
    out = []
    for constr in constraints:
        new = {
            key: value
            for key, value in constr.items()
            if key in ("value", "lower_bound", "upper_bound")
        }
        new["weights"] = pd.Series(constr["weights"], index=list(constr["index"]))
        out.append(new)
    return out


def _to_new_format(constraints):
    out = []
    for constr in constraints:
        new = dict(constr)
        new["type"] = "linear"
        out.append(new)
    return out


def _run(func, *args, **kwargs):
    try:
        return func(*args, **kwargs), None
    except Exception as e:
        return None, type(e)


@pytest.mark.parametrize("seed", range(200))
def test_new_linear_consolidation_equals_legacy(seed):
    rng = get_rng(seed)
    params_vec, constr_info, constraints = _draw_scenario(rng)
    param_names = [str(i) for i in range(len(params_vec))]

    expected, expected_error = _run(
        consolidate_linear_constraints_legacy,
        params_vec=params_vec,
        linear_constraints=_to_legacy_format(constraints),
        constr_info=constr_info,
        param_names=param_names,
    )
    got, got_error = _run(
        _consolidate_linear_constraints,
        params_vec=params_vec,
        linear_constraints=_to_new_format(constraints),
        constr_info=constr_info,
        param_names=param_names,
    )

    if expected_error is not None:
        assert got_error is not None, (
            f"legacy implementation raised {expected_error} but the new one did not"
        )
        return

    assert got_error is None, f"new implementation raised {got_error} unexpectedly"
    assert len(got) == len(expected)

    for new_constr, old_constr in zip(got, expected, strict=True):
        assert new_constr["index"] == [int(i) for i in old_constr["index"]]
        assert_allclose(
            new_constr["to_internal"],
            np.asarray(old_constr["to_internal"], dtype=float),
            rtol=1e-12,
            atol=1e-12,
        )
        assert_allclose(
            new_constr["from_internal"],
            np.asarray(old_constr["from_internal"], dtype=float),
            rtol=1e-12,
            atol=1e-12,
        )
        old_rhs = old_constr["right_hand_side"]
        new_rhs = new_constr["right_hand_side"]
        assert_allclose(
            new_rhs.values, old_rhs["value"].to_numpy(), equal_nan=True, atol=1e-12
        )
        assert_allclose(
            new_rhs.lower_bounds, old_rhs["lower_bound"].to_numpy(), atol=1e-12
        )
        assert_allclose(
            new_rhs.upper_bounds, old_rhs["upper_bound"].to_numpy(), atol=1e-12
        )
