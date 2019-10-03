import warnings

import numpy as np
import pandas as pd

from estimagic.optimization.check_constraints import check_compatibility_of_constraints
from estimagic.optimization.utilities import cov_params_to_matrix
from estimagic.optimization.utilities import sdcorr_params_to_matrix


def process_constraints(constraints, params):
    """Process, consolidate and check constraints.

    Note: Do not change the order of function calls.

    Args:
        constraints (list): List of dictionaries where each dictionary is a constraint.
        params (pd.DataFrame): see :ref:`params`.

    Returns:
        processed (list): the processed constraints.

    """

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="indexing past lexsort depth may impact performance."
        )
        # it is important to process selectors first
        constraints = _process_selectors(constraints, params)
        fixed = apply_fixes_to_external_params(
            params, [c for c in constraints if c["type"] == "fixed"]
        )
        constraints = _replace_pairwise_equality_by_equality(constraints, params)
        constraints = _consolidate_equality_constraints(constraints, params)

        processed_constraints = []
        for constr in constraints:
            if constr["type"] == "covariance":
                processed_constraints.append(
                    _process_cov_constraint(constr, params, fixed)
                )
            elif constr["type"] == "sdcorr":
                processed_constraints.append(
                    _process_sdcorr_constraint(constr, params, fixed)
                )
            elif constr["type"] in [
                "fixed",
                "sum",
                "probability",
                "increasing",
                "equality",
            ]:
                processed_constraints.append(constr)
            else:
                raise ValueError("Invalid constraint type {}".format(constr["type"]))

        check_compatibility_of_constraints(processed_constraints, params, fixed)

    return processed_constraints


def _process_selectors(constraints, params):
    """Process and harmonize the query and loc field of the constraints.

    Args:
        constraints (list): List of dictionaries where each dictionary is a constraint.
        params (pd.DataFrame): see :ref:`params`.

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
            equality_constraints.append({"loc": list(elements), "type": "equality"})
        final_constraints += _process_selectors(equality_constraints, params)
    return final_constraints


def _process_cov_constraint(constraint, params, fixed):
    """Process covariance constraints.

    Args:
        constraint (dict)
        params (pd.DataFrame): see :ref:`params`.

    Returns:
        new_constr (dict): copy of *constraint* with a new entry called 'case',
            which can take the values 'all_fixed', 'uncorrelated' and 'free'.

    """
    new_constr = constraint.copy()
    params_subset = params.loc[constraint["index"]]
    fixed_subset = fixed.loc[constraint["index"]]
    value_mat = cov_params_to_matrix(params_subset["value"].to_numpy())
    fixed_mat = cov_params_to_matrix(fixed_subset["_fixed"].to_numpy()).astype(bool)
    new_constr["case"] = _determine_cov_case(value_mat, fixed_mat, params_subset)
    new_constr["bounds_distance"] = constraint.pop("bounds_distance", 0.0)

    return new_constr


def _process_sdcorr_constraint(constraint, params, fixed):
    """Process sdcorr constraints.

    Args:
        constraint (dict)
        params (pd.DataFrame): see :ref:`params`.


    Returns:
        new_constr (dict): copy of *constraint* with a new entry called 'case',
            which can take the values 'all_fixed', 'uncorrelated' and 'free'.

    """
    new_constr = constraint.copy()
    params_subset = params.loc[constraint["index"]]
    value_mat = sdcorr_params_to_matrix(params_subset["value"].to_numpy())
    dim = len(value_mat)
    fixed_vec = fixed.loc[constraint["index"], "_fixed"].to_numpy().astype(int)
    fixed_diag = np.diag(fixed_vec[:dim])
    fixed_lower = np.zeros((dim, dim), dtype=int)
    fixed_lower[np.tril_indices(dim, k=-1)] = fixed_vec[dim:]
    fixed_mat = (fixed_lower + fixed_diag + fixed_lower.T).astype(bool)
    new_constr["case"] = _determine_cov_case(value_mat, fixed_mat, params_subset)
    new_constr["bounds_distance"] = constraint.pop("bounds_distance", 0.0)
    return new_constr


def _determine_cov_case(value_mat, fixed_mat, params_subset):
    """How constrained a covariance matrix is.

    Args:
        value_mat (np.array): start parameters for the implied covariance matrix
        fixed_mat (np.array): which elements of the implied covariance matrix are fixed.
        params_subset (DataFrame): relevant subset of a :ref:`params`.

    Returns:
        case (str): takes the values 'all_fixed', 'uncorrelated', 'free'

    """
    dim = len(value_mat)
    off_diagonal_zero = bool((value_mat[np.tril_indices(dim, k=-1)] == 0).all())

    off_diagonal_fixed = bool(fixed_mat[np.tril_indices(dim, k=-1)].all())
    all_fixed = bool(fixed_mat.all())

    if all_fixed:
        case = "all_fixed"
    elif off_diagonal_fixed and off_diagonal_zero:
        case = "uncorrelated"
    else:
        fixed_copy = fixed_mat.copy()
        fixed_copy[0, 0] = False

        assert not fixed_copy.any(), (
            "Only the first diagonal element can be fixed for parameters with "
            "covariance or sdcorr constraint."
        )
        case = "free"

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
        params (pd.DataFrame): see :ref:`params`.

    Returns:
        consolidated (list): The consolidated equality constraints.

    """
    equality_constraints = [c for c in constraints if c["type"] == "equality"]
    other_constraints = [c for c in constraints if c["type"] != "equality"]

    candidates = [constr["index"] for constr in equality_constraints]
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
    return consolidated + other_constraints


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


def _sort_key(x):
    if x["type"] == "fixed":
        return 0
    elif x["type"] == "equality":
        return 1
    else:
        return 2


def apply_fixes_to_external_params(params, fixes):
    params = params.copy()
    params["_fixed"] = False
    for fix in fixes:
        assert fix["type"] == "fixed", "Invalid constraint of type {} in fixes.".format(
            fix["type"]
        )
        params.loc[fix["index"], "_fixed"] = True
        if "value" in fix:
            params.loc[fix["index"], "value"] = fix["value"]
    return params
