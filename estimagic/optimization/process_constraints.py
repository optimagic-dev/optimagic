import warnings

import numpy as np
import pandas as pd

from estimagic.optimization.check_constraints import check_constraints_are_satisfied
from estimagic.optimization.check_constraints import check_fixes_and_bounds
from estimagic.optimization.check_constraints import check_for_incompatible_overlaps
from estimagic.optimization.check_constraints import check_types
from estimagic.optimization.consolidate_constraints import consolidate_constraints
from estimagic.optimization.utilities import number_of_triangular_elements_to_dimension


def process_constraints(constraints, params):
    """Process, consolidate and check constraints.

    Args:
        constraints (list): List of dictionaries where each dictionary is a constraint.
        params (pd.DataFrame): see :ref:`params`.

    Returns:

        transforming_constraints (list): A processed version of those constraints
            that entail actual transformations and not just fixing parameters.
        processed_params (pd.DataFrame): Copy of params with additional columns:
            - _internal_lower:
                Lower bounds for the internal parameter vector. Those are derived from
                the original lower bounds and additional bounds implied by other
                constraints.
            - _internal_upper: As _internal_lower but for upper bounds.
            - _internal_free: Boolean column that is true for those parameters over
                which the optimizer will actually optimize.
            - _pre_replacements: The j_th element indicates the position of the internal
                parameter that has to be copied into the j_th position of the external
                parameter vector when reparametrizing from_internal, before any
                transformations are applied. Negative if no element has to be copied.
            - _post_replacements: As pre_replacements, but applied after the
                transformations are done.
            - _internal_fixed_value: Contains transformed versions of the fixed values
                that will become equal to the external fixed values after the
                kernel transformations are applied.

    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="indexing past lexsort depth may impact performance."
        )
        constraints = _apply_constraint_killers(constraints)
        check_types(constraints)
        constraints = _process_selectors(constraints, params)
        constraints = _replace_pairwise_equality_by_equality(constraints, params)
        constraints = _process_linear_weights(constraints, params)
        check_constraints_are_satisfied(constraints, params)
        constraints = _replace_increasing_and_decreasing_by_linear(constraints)
        constraints = _process_linear_weights(constraints, params)
        constraints, pp = consolidate_constraints(constraints, params)
        check_for_incompatible_overlaps(pp, constraints)
        check_fixes_and_bounds(pp, constraints)

        int_lower, int_upper = _create_internal_bounds(pp, constraints)
        pp["_internal_lower"] = int_lower
        pp["_internal_upper"] = int_upper

        pp["_internal_free"] = _create_internal_free(pp, constraints)

        pp["_pre_replacements"] = _create_pre_replacements(pp, constraints)
        pp["_internal_fixed_value"] = _create_internal_fixed_value(pp, constraints)

        return constraints, pp


def _create_internal_bounds(processed_params, processed_constraints):
    int_lower = processed_params["lower"].copy()
    int_upper = processed_params["upper"].copy()

    for constr in processed_constraints:
        if constr["type"] in ["covariance", "sdcorr"]:
            # Note that the diagonal positions are the same for covariance and sdcorr
            # because the internal params contain the cholesky factor of the implied
            # covariance matrix in both cases.
            dim = number_of_triangular_elements_to_dimension(len(constr["index"]))
            diag_positions = [0] + np.cumsum(range(2, dim + 1)).tolist()
            diag_indices = np.array(constr["index"])[diag_positions].tolist()
            bd = constr.get("bounds_distance", 0.0)
            int_lower.iloc[diag_indices] = np.maximum(int_lower.iloc[diag_indices], bd)
        elif constr["type"] == "probability":
            int_lower.iloc[constr["index"]] = 0
        elif constr["type"] == "linear":
            int_lower.iloc[constr["index"]] = -np.inf
            int_upper.iloc[constr["index"]] = np.inf
            int_lower.update(constr["right_hand_side"]["lower_bound"])
            int_upper.update(constr["right_hand_side"]["upper_bound"])
        else:
            raise TypeError("Invalid constraint type {}".format(constr["type"]))

    return int_lower, int_upper


def _create_internal_free(processed_params, processed_constraints):
    """Boolean Series that is true for parameters over which the optimizer optimizes."""
    int_fixed = processed_params["_is_fixed_to_value"]
    int_fixed = int_fixed | processed_params["_is_fixed_to_other"]

    for constr in processed_constraints:
        if constr["type"] == "probability":
            int_fixed.iloc[constr["index"][-1]] = True
        elif constr["type"] == "linear":
            int_fixed.iloc[constr["index"]] = False
            int_fixed.update(constr["right_hand_side"]["value"].notnull())
            # dtype gets messed up by update
            int_fixed = int_fixed.astype(bool)

    int_free = ~int_fixed
    return int_free


def _create_pre_replacements(processed_params, processed_constraints):
    """Series with internal position of parameters.

    The j_th element indicates the position of the internal parameter that has to be
    copied into the j_th position of the external parameter vector when reparametrizing
    from_internal, before any transformations are applied. Negative if no element has
    to be copied.

    This will be used to copy the free internal parameters into a parameter vector
    that has the same length as all params.

    """
    free = processed_params.query("_internal_free").copy()
    free["_internal_iloc"] = np.arange(len(free))
    free = free["_internal_iloc"]
    pre_replacements = pd.concat([processed_params, free], axis=1)["_internal_iloc"]
    pre_replacements = pre_replacements.fillna(-1).astype(int)
    return pre_replacements


def _create_internal_fixed_value(processed_params, processed_constraints):
    int_fix = processed_params["_fixed_value"].copy()
    for constr in processed_constraints:
        if constr["type"] == "probability":
            int_fix.iloc[constr["index"][-1]] = 1
        elif constr["type"] in ["covariance", "sdcorr"]:
            int_fix.iloc[constr["index"][0]] = np.sqrt(int_fix.iloc[constr["index"][0]])
        elif constr["type"] == "linear":
            int_fix.iloc[constr["index"]] = np.nan
            int_fix.update(constr["right_hand_side"]["value"])
    return int_fix


def _apply_constraint_killers(constraints):
    """Filter out constraints that have a killer."""
    to_kill, real_constraints = [], []
    for constr in constraints:
        if "kill" in constr and len(constr) == 1:
            to_kill.append(constr["kill"])
        else:
            real_constraints.append(constr)

    to_kill = set(to_kill)

    survivors = []
    for constr in real_constraints:
        if "id" not in constr or constr["id"] not in to_kill:
            survivors.append(constr)

    present_ids = [constr["id"] for c in real_constraints if "id" in constr]

    if not to_kill.issubset(present_ids):
        invalid = to_kill.difference(present_ids)
        raise KeyError(f"You try to kill constraint with non-exsting id: {invalid}")

    return survivors


def _process_selectors(constraints, params):
    """Convert the query and loc field of the constraints into position based indices.

    Args:
        constraints (list): List of dictionaries where each dictionary is a constraint.
        params (pd.DataFrame): see :ref:`params`.

    Returns:
        processed (list): The resulting constraint dictionaries contain a new entry
            called 'index' that consists of the positions of the selected parameters.
            If the selected parameters are consecutive entries, the value corresponding
            to 'index' is a list of positions.

    """
    processed = []

    for constr in constraints:
        new_constr = constr.copy()

        if constr["type"] != "pairwise_equality":
            locs = [constr["loc"]] if "loc" in constr else []
            queries = [constr["query"]] if "query" in constr else []
        else:
            locs = new_constr.pop("locs", [])
            queries = new_constr.pop("queries", [])

        positions = pd.Series(data=np.arange(len(params)), index=params.index)

        indices = []
        for loc in locs:
            index = positions.loc[loc].astype(int).tolist()
            index = [index] if not isinstance(index, list) else index
            assert len(set(index)) == len(index), "Duplicates in loc are not allowed."
            indices.append(index)
        for query in queries:
            loc = params.query(query).index
            index = positions.loc[loc].astype(int).tolist()
            index = [index] if not isinstance(index, list) else index
            indices.append(index)

        if constr["type"] == "pairwise_equality":
            assert (
                len(indices) >= 2
            ), "Select at least 2 sets of parameters for pairwise equality constraint!"
            length = len(indices[0])
            for index in indices:
                assert len(index) == length, (
                    "All sets of parameters in pairwise_equality constraints must have "
                    "the same length."
                )
            new_constr["indices"] = indices
        else:
            assert (
                len(indices) == 1
            ), "Either loc or query can be in constraint but not both."
            new_constr["index"] = indices[0]
        processed.append(new_constr)
    return processed


def _replace_pairwise_equality_by_equality(constraints, params):
    """Rewrite pairwise equality constraints to equality constraints.

    Args:
        constraints (list): List of dictionaries where each dictionary is a constraint.
            It is assumed that the selectors in the constraints were already processed.
        params (DataFrame): see :ref:`params` for details.

    Returns:
        constraints (list): equality constraints

    """
    pairwise_constraints = [c for c in constraints if c["type"] == "pairwise_equality"]
    final_constraints = [c for c in constraints if c["type"] != "pairwise_equality"]
    for constr in pairwise_constraints:
        equality_constraints = []
        for elements in zip(*constr["indices"]):
            equality_constraints.append({"index": list(elements), "type": "equality"})
        final_constraints += equality_constraints
    return final_constraints


def _process_linear_weights(constraints, params):
    processed = []
    for constr in constraints:
        if constr["type"] == "linear":
            raw_weights = constr["weights"]
            params_subset = params.iloc[constr["index"]]
            msg = f"Weights must be same length as selected parameters: {params_subset}"
            if isinstance(raw_weights, pd.Series):
                weights = raw_weights.loc[params_subset.index].to_numpy()
            elif isinstance(raw_weights, (np.ndarray, list, tuple)):
                if len(raw_weights) != len(params_subset):
                    raise ValueError(msg)
                weights = np.asarray(raw_weights)
            elif isinstance(raw_weights, (float, int)):
                weights = np.full(len(params_subset), float(weights))
            else:
                raise TypeError(
                    "Invalid type for linear weights {}.".format(type(raw_weights))
                )
            new_constr = constr.copy()
            weights_sr = pd.Series(weights, index=params_subset.index)
            new_constr["weights"] = weights_sr
            processed.append(new_constr)
        else:
            processed.append(constr)

    return processed


def _replace_increasing_and_decreasing_by_linear(constraints):
    increasing_ilocs, other_constraints = [], []

    for constr in constraints:
        if constr["type"] == "increasing":
            increasing_ilocs.append(constr["index"])
        elif constr["type"] == "decreasing":
            increasing_ilocs.append(constr["index"][::-1])
        else:
            other_constraints.append(constr)

    linear_constraints = []
    for iloc in increasing_ilocs:
        for smaller, larger in zip(iloc, iloc[1:]):
            lincon = {
                "index": [smaller, larger],
                "type": "linear",
                "weights": np.array([-1, 1]),
                "lower_bound": 0,
            }
            linear_constraints.append(lincon)

    return linear_constraints + other_constraints
