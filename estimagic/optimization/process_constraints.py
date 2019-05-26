import warnings

import numpy as np

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
        params["__fixed__"] = False

        constraints = _process_selectors(constraints, params)
        constraints = _replace_pairwise_equality_by_equality(constraints, params)

        equality_constraints = [c for c in constraints if c["type"] == "equality"]
        other_constraints = [c for c in constraints if c["type"] != "equality"]

        for constr in other_constraints:
            if constr["type"] == "fixed":
                params.loc[constr["index"], "__fixed__"] = True

        processed_constraints += _consolidate_equality_constraints(
            equality_constraints, params
        )

        for constr in other_constraints:
            if constr["type"] == "covariance":
                processed_constraints.append(_process_cov_constraint(constr, params))
            elif constr["type"] == "sdcorr":
                processed_constraints.append(_process_sdcorr_constraint(constr, params))
            elif constr["type"] in ["sum", "probability", "increasing", "fixed"]:
                processed_constraints.append(constr)
            else:
                raise ValueError("Invalid constraint type {}".format(constr["type"]))

        _check_compatibility_of_constraints(processed_constraints, params)

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
        if constr["type"] != "pairwise_equality":
            suffixes = [""]
        else:
            suffixes = [1, 2]

        new_constr = constr.copy()

        for suf in suffixes:
            assert (
                f"query{suf}" in constr or f"loc{suf}" in constr
            ), f"Either query{suf} or loc{suf} has to be in a constraint dictionary."
            assert not (
                f"query{suf}" in constr and f"loc{suf}" in constr
            ), f"query{suf} and loc{suf} cannot both be in a constraint dictionary."

            par_copy = params.copy()

            if f"query{suf}" in constr:
                query = new_constr.pop(f"query{suf}")
                index = par_copy.query(query).index
            else:
                loc = new_constr.pop(f"loc{suf}")
                par_copy["selected"] = False
                par_copy.loc[loc, "selected"] = True
                index = par_copy.query("selected").index

            new_constr[f"index{suf}"] = index

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
        index1, index2 = constr["index1"], constr["index2"]
        assert len(index1) == len(
            index2
        ), "index1 and index2 must have the same length."
        equality_constraints = []
        for i1, i2 in zip(index1, index2):
            equality_constraints.append({"loc": [i1, i2], "type": "equality"})
        final_constraints += _process_selectors(equality_constraints, params)
    return final_constraints


def _process_cov_constraint(constraint, params):
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
    value_mat = cov_params_to_matrix(params_subset["value"].to_numpy())
    fixed_mat = cov_params_to_matrix(params_subset["__fixed__"].to_numpy()).astype(bool)
    new_constr["case"] = _determine_cov_case(value_mat, fixed_mat, params_subset)
    return new_constr


def _process_sdcorr_constraint(constraint, params):
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
    fixed_vec = params_subset["__fixed__"].to_numpy().astype(int)
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
    all_fixed = bool(params_subset["__fixed__"].all())

    if all_fixed is True:
        case = "all_fixed"
    elif off_diagonal_fixed and off_diagonal_zero:
        case = "uncorrelated"
    else:
        assert not params_subset[
            "__fixed__"
        ].any(), "Fixed parameters are not allowed for covariance or sdcorr containt."
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


def _check_compatibility_of_constraints(constraints, params):
    """Additional compatibility checks for constraints.

    Checks that require fine grained case distinctions are already done in the functions
    that reparametrize to_internal.

    Args:
        constraint (dict)
        params (pd.DataFrame): see :ref:`params_df`.

    """
    params = params.copy()
    constr_types = [
        "covariance",
        "sdcorr",
        "sum",
        "probability",
        "increasing",
        "equality",
    ]

    for typ in constr_types:
        params["has_" + typ] = False

    for constr in constraints:
        params.loc[constr["index"], "has_" + constr["type"]] = True

    params["has_lower"] = params["lower"] != -np.inf
    params["has_upper"] = params["upper"] != np.inf

    invalid_cov = (
        "has_covariance & (has_equality | has_sum | has_increasing | has_probability | "
        "has_sdcorr)"
    )

    invalid_sdcorr = (
        "has_sdcorr & (has_equality | has_sum | has_increasing | has_probability | "
        "has_covariance)"
    )

    assert (
        len(params.query(invalid_cov)) == 0
    ), "covariance constraints are not compatible with other constraints"

    assert (
        len(params.query(invalid_sdcorr)) == 0
    ), "sdcorr constraints are not compatible with other constraints"
