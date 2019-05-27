import warnings

import numpy as np
import pandas as pd

from estimagic.optimization.utilities import cov_params_to_matrix
from estimagic.optimization.utilities import sdcorr_params_to_matrix


def process_constraints(constraints, params):
    """Process, consolidate and check constraints.

    Note: Do not change the order of function calls.

    Args:
        constraints (list): see :ref:`constraints`.
        params (pd.DataFrame): see :ref:`params_df`.

    Returns:
        processed (list): the processed constraints.

    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="indexing past lexsort depth may impact performance."
        )

        processed_constraints = []
        params = params.copy()
        fixed = pd.DataFrame(index=params.index)
        fixed["bool"] = False
        fixed["value"] = np.nan

        constraints = _process_selectors(constraints, params)
        constraints = _replace_pairwise_equality_by_equality(constraints, params)

        equality_constraints = [c for c in constraints if c["type"] == "equality"]
        other_constraints = [c for c in constraints if c["type"] != "equality"]

        for constr in other_constraints:
            if constr["type"] == "fixed":
                fixed.loc[constr["index"], "bool"] = True
                fixed.loc[constr["index"], "value"] = constr["value"]

        processed_constraints += _consolidate_equality_constraints(
            equality_constraints, params
        )

        for constr in other_constraints:
            if constr["type"] == "covariance":
                processed_constraints.append(
                    _process_cov_constraint(constr, params, fixed)
                )
            elif constr["type"] == "sdcorr":
                processed_constraints.append(
                    _process_sdcorr_constraint(constr, params, fixed)
                )
            elif constr["type"] in ["sum", "probability", "increasing", "fixed"]:
                processed_constraints.append(constr)
            else:
                raise ValueError("Invalid constraint type {}".format(constr["type"]))

        _check_compatibility_of_constraints(processed_constraints, params, fixed)

    return processed_constraints


def _process_selectors(constraints, params):
    """Process and harmonize the query and loc field of the constraints.

    Args:
        constraints (list): see :ref:`constraints`.
        params (pd.DataFrame): see :ref:`params_df`.

    Returns:
        processed (list): The resulting constraint dictionaries contain a new entry
            called 'index' that consists of the full index of all selected parameters.

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

        indices = []
        for loc in locs:
            selected = pd.DataFrame(index=params.index, data=False, columns=["select"])
            selected.loc[loc] = True
            index = selected.query("select").index
            assert not index.duplicated().any(), "Duplicates in loc are not allowed."
            indices.append(index)
        for query in queries:
            indices.append(params.query(query).index)

        if constr["type"] == "pairwise_equality":
            assert (
                len(indices) >= 2
            ), "Select at least two 2 of parameters for pairwise equality constraint!"
            length = len(indices[0])
            for index in indices:
                assert (
                    len(index) == length
                ), "Invalid selector in pairwise equality constraint."
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
        constraints (list): list of constraints.
            It is assumed that the selectors in the constraints were already processed.
        params (DataFrame): see :ref:`params_df` for details.

    Returns:
        constraints (list): equality constraints

    """
    pairwise_constraints = [c for c in constraints if c["type"] == "pairwise_equality"]
    final_constraints = [c for c in constraints if c["type"] != "pairwise_equality"]
    for constr in pairwise_constraints:
        equality_constraints = []
        for elements in zip(*constr["indices"]):
            equality_constraints.append({"loc": list(elements), "type": "equality"})
        final_constraints += _process_selectors(equality_constraints, params)
    return final_constraints


def _process_cov_constraint(constraint, params, fixed):
    """Process covariance constraints.

    Args:
        constraint (dict)
        params (pd.DataFrame): see :ref:`params_df`.

    Returns:
        new_constr (dict): copy of *constraint* with a new entry called 'case',
            which can take the values 'all_fixed', 'uncorrelated' and 'all_free'.

    """
    new_constr = constraint.copy()
    params_subset = params.loc[constraint["index"]]
    fixed_subset = fixed.loc[constraint["index"]]
    value_mat = cov_params_to_matrix(params_subset["value"].to_numpy())
    fixed_mat = cov_params_to_matrix(fixed_subset["bool"].to_numpy()).astype(bool)
    new_constr["case"] = _determine_cov_case(value_mat, fixed_mat, params_subset)
    return new_constr


def _process_sdcorr_constraint(constraint, params, fixed):
    """Process sdcorr constraints.

    Args:
        constraint (dict)
        params (pd.DataFrame): see :ref:`params_df`.


    Returns:
        new_constr (dict): copy of *constraint* with a new entry called 'case',
            which can take the values 'all_fixed', 'uncorrelated' and 'all_free'.

    """
    new_constr = constraint.copy()
    params_subset = params.loc[constraint["index"]]
    value_mat = sdcorr_params_to_matrix(params_subset["value"].to_numpy())
    dim = len(value_mat)
    fixed_vec = fixed.loc[constraint["index"], "bool"].to_numpy().astype(int)
    fixed_diag = np.diag(fixed_vec[:dim])
    fixed_lower = np.zeros((dim, dim), dtype=int)
    fixed_lower[np.tril_indices(dim, k=-1)] = fixed_vec[dim:]
    fixed_mat = (fixed_lower + fixed_diag + fixed_lower.T).astype(bool)
    new_constr["case"] = _determine_cov_case(value_mat, fixed_mat, params_subset)
    return new_constr


def _determine_cov_case(value_mat, fixed_mat, params_subset):
    """How constrained a covariance matrix is.

    Args:
        value_mat (np.array): start parameters for the implied covariance matrix
        fixed_mat (np.array): which elements of the implied covariance matrix are fixed.
        params_subset (DataFrame): relevant subset of a :ref:`params_df`.

    Returns:
        case (str): takes the values 'all_fixed', 'uncorrelated', 'all_free'

    """
    dim = len(value_mat)
    off_diagonal_zero = bool((value_mat[np.tril_indices(dim, k=-1)] == 0).all())

    off_diagonal_fixed = bool(fixed_mat[np.tril_indices(dim, k=-1)].all())
    all_fixed = bool(fixed_mat.all())

    if all_fixed is True:
        case = "all_fixed"
    elif off_diagonal_fixed and off_diagonal_zero:
        case = "uncorrelated"
    else:
        assert (
            not fixed_mat.any()
        ), "Fixed parameters are not allowed for covariance or sdcorr constraint."
        case = "all_free"

    return case


def _consolidate_equality_constraints(constraints, params):
    """Consolidate equality constraints as far as possible.

    Since equality is a transitive conditions we can consolidate any two equality
    constraints have at least one parameter in common into one condition. Besides being
    faster, this also ensures that the result remains unchanged if equality conditions
    are split into several different constraints or their order specified in a different
    order.

    The index in the consolidated equality constraints is sorted in the same order
    as the index of params.

    Args:
        constraint (dict)
        params (pd.DataFrame): see :ref:`params_df`.

    Returns:
        consolidated (list): The consolidated equality constraints.

    """
    candidates = [constr["index"] for constr in constraints]
    # drop constraints that just restrict one parameter to be equal to itself
    candidates = [c for c in candidates if len(c) >= 2]

    merged = []

    while len(candidates) > 0:
        new_candidates = _unite_first_with_all_intersecting_elements(candidates)
        if len(candidates) == len(new_candidates):
            merged.append(candidates[0])
            candidates = candidates[1:]
        else:
            candidates = new_candidates

    ordered = []
    for m in merged:
        helper = params.copy()
        helper["selected"] = False
        helper.loc[m, "selected"] = True
        ordered.append(helper.query("selected").index)

    consolidated = [{"index": index, "type": "equality"} for index in ordered]
    return consolidated


def _unite_first_with_all_intersecting_elements(indices):
    """Helper function to consolidate equality constraints.

    Args:
        indices (list): A list of pandas Index objects.

    """
    first = indices[0]
    new_first = first
    new_others = []
    for idx in indices[1:]:
        if len(first.intersection(idx)) > 0:
            new_first = new_first.union(idx)
        else:
            new_others.append(idx)
    return [new_first] + new_others


def _check_compatibility_of_constraints(constraints, params, fixed):
    """Additional compatibility checks for constraints.

    Checks that require fine grained case distinctions are already done in the functions
    that reparametrize to_internal.

    Args:
        constraint (dict)
        params (pd.DataFrame): see :ref:`params_df`.

    """
    _check_no_overlapping_transforming_constraints(constraints, params)
    _check_no_invalid_equality_constraints(constraints, params)
    _check_fixes(params, fixed)


def _check_no_overlapping_transforming_constraints(constraints, params):
    counter = pd.Series(index=params.index, data=0, name="constraint_type")

    transforming_types = ["covariance", "sdcorr", "sum", "probability", "increasing"]

    for constr in constraints:
        if constr["type"] in transforming_types:
            counter.loc[constr["index"]] += 1

    invalid = counter >= 2

    if invalid.any() > 0:
        raise ValueError("Overlapping constraints for {}".format(params.loc[invalid]))


def _check_no_invalid_equality_constraints(constraints, params):
    """Check that equality constraints are compatible with other constraints.

    In general, we don't allow for equality constraints on parameters that have
    constraints that require reparametrizations. The only exception is when a set of
    parameters is pairwise equal to another set of parameters that has the same
    constraint.

    In the long run we could allow some more equality constraints for sum and
    probability constraints bit this is relatively complex and probably rarely
    needed.

    """
    helper = pd.DataFrame(index=params.index)
    helper["eq_id"] = -1
    helper["constraint_type"] = "None"

    transforming_types = ["covariance", "sdcorr", "probability", "increasing"]
    sums = []
    for constr in constraints:
        if constr["type"] == "sum":
            sums.append("sum_" + str(constr["value"]))
    transforming_types += sums

    extended_constraints = []
    for constr in constraints:
        if constr["type"] == "sum":
            new_constr = constr.copy()
            new_constr["type"] = "sum_" + str(constr["value"])
            extended_constraints.append(new_constr)
        else:
            extended_constraints.append(constr)

    equality_constraints = [c for c in constraints if c["type"] == "equality"]

    for i, constr in enumerate(equality_constraints):
        if constr["type"] == "equality":
            helper.loc[constr["index"], "eq_id"] = i

    for constr in constraints:
        if constr["type"] in transforming_types:
            helper.loc[constr["index"], "constraint_type"] = constr["type"]

    for constr in equality_constraints:
        other_constraint_types = helper.loc[constr["index"], "constraint_type"].unique()

        if len(other_constraint_types) > 1:
            raise ValueError("Incompatible equality constraint.")
        other_type = other_constraint_types[0]
        if other_type != "None":
            other_constraints = [c for c in constraints if c["type"] == other_type]

            relevant_others = []
            for ind_tup in constr["index"]:
                for other_constraint in other_constraints:
                    if ind_tup in other_constraint["index"]:
                        relevant_others.append(other_constraint)

            first_eq_ids = helper.loc[relevant_others[0]["index"], "eq_id"]
            if len(first_eq_ids.unique()) != len(first_eq_ids):
                raise ValueError("Incompatible equality constraint.")

            for rel in relevant_others:
                eq_ids = helper.loc[rel["index"], "eq_id"]

                if not (eq_ids.to_numpy() == first_eq_ids.to_numpy()).all():
                    raise ValueError("Incompatible equality constraint.")


def _check_fixes(params, fixed):
    fixed = fixed.query("bool")
    for p in fixed.index:
        if not pd.isnull(params.loc[p, "value"]):
            fvalue = fixed.loc[p, "value"]
            value = params.loc[p, "value"]
            if fvalue != value:
                warnings.warn(
                    "Parameter {} is fixed to {} but value column is {}".format(
                        p, fvalue, value
                    )
                )
