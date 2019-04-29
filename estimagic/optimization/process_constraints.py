import numpy as np
import warnings
from estimagic.optimization.utilities import cov_params_to_matrix


def process_constraints(constraints, params):
    """Process, consolidate and check constraints.

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

        processed = []

        constraints = _process_selectors(constraints, params)
        equality_constraints = []
        for constr in constraints:
            if constr["type"] == "equality":
                equality_constraints.append(constr)
            elif constr["type"] == "covariance":
                processed.append(_process_cov_constraint(constr, params))
            else:
                processed.append(constr)

        processed += _consolidate_equality_constraints(equality_constraints, params)
        _check_compatibility_of_constraints(constraints, params)

    return processed


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
        assert (
            "query" in constr or "loc" in constr
        ), "Either query or loc has to be in a constraint dictionary."
        assert not (
            "query" in constr and "loc" in constr
        ), "query and loc cannot both be specified in a constraint dictionary."

        par_copy = params.copy()
        new_constr = constr.copy()

        if "query" in constr:
            query = new_constr.pop("query")
            index = par_copy.query(query).index
        else:
            loc = new_constr.pop("loc")
            par_copy["selected"] = False
            par_copy.loc[loc, "selected"] = True
            index = par_copy.query("selected").index

        new_constr["index"] = index
        processed.append(new_constr)

    return processed


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
    cov = cov_params_to_matrix(params_subset["value"].to_numpy())
    dim = len(cov)
    off_diagonal_zero = bool((cov[np.tril_indices(dim, k=-1)] == 0).all())

    fixed_helper = cov_params_to_matrix(params_subset["fixed"].to_numpy()).astype(bool)
    off_diagonal_fixed = bool(fixed_helper[np.tril_indices(dim, k=-1)].all())
    all_fixed = bool(params_subset["fixed"].all())

    if all_fixed is True:
        case = "all_fixed"
    elif off_diagonal_fixed and off_diagonal_zero:
        case = "uncorrelated"
    else:
        case = "all_free"
    new_constr["case"] = case
    return new_constr


def _consolidate_equality_constraints(constraints, params):
    """Consolidate equality constraints as far as possible.

    Since equality is a transitive conditions we can consolidate any two equality
    constraints have at least one parameter in common into one condition. Besides being
    faster, this also ensures that the result remains unchanged if equality conditions
    are split into several different constraints or their order specified in a differnt
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
    constr_types = ["covariance", "sum", "probability", "increasing", "equality"]

    for typ in constr_types:
        params["has_" + typ] = False

    for constr in constraints:
        params.loc[constr["index"], "has_" + constr["type"]] = True

    params["has_lower"] = params["lower"] != -np.inf
    params["has_upper"] = params["upper"] != np.inf

    invalid_cov = (
        "has_covariance & (has_equality | has_sum | has_increasing | has_probability)"
    )

    assert (
        len(params.query(invalid_cov)) == 0
    ), "covariance constraints are not compatible with other constraints"
