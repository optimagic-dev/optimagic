import numpy as np


def process_constraints(constraints, params):
    """Process, consolidate and check constraints."""
    processed = []
    equality_constraints = []
    for constr in constraints:
        if constr['type'] == 'equality':
            equality_constraints.append(constr)
        else:
            processed.append(constr)

    processed += _consolidate_equality_constraints(equality_constraints, params)
    _check_compatibility_of_constraints(constraints, params)

    return processed


def _consolidate_equality_constraints(constraints, params):
    """Consolidate equality constraints as far as possible.

    Since equality is a transitive conditions we can consolidate any two equality
    constraints have at least one parameter in common into one condition. Besides being
    faster, this also ensures that the result remains unchanged if equality conditions
    are split into several different constraints or their order specified in a differnt
    order.

    """
    candidates = [
        params.loc[constr['selector']].index for constr in constraints]

    consolidated = []

    while len(candidates) > 0:
        new_candidates = _unite_first_with_all_intersecting_elements(candidates)
        if len(candidates) == len(new_candidates):
            consolidated.append(candidates[0])
            candidates = candidates[1:]
        else:
            candidates = new_candidates

    res = [{'selector': cons, 'type': 'equality'} for cons in consolidated]
    return res


def _unite_first_with_all_intersecting_elements(indices):
    """Helper function to consolidate equality constraints."""
    first = indices[0]
    new_first = first
    new_others = []
    for idx in indices[1:]:
        print(type(first))
        if len(first.intersection(idx)) > 0:
            new_first = new_first.union(idx)
        else:
            new_others.append(idx)
    return [new_first] + new_others


def _check_compatibility_of_constraints(constraints, params):
    """Additional compatibility checks for constraints.

    Checks that require fine grained case distinctions are already done in the functions
    that reparametrize to_internal.

    """
    params = params.copy()
    constr_types = ['covariance', 'sum', 'probability', 'increasing', 'equality']

    for typ in constr_types:
        params['has_' + typ] = False

    for constr in constraints:
        params.loc[constr['selector'], 'has_' + constr['type']] = True

    params['has_lower'] = params['lower'] != - np.inf
    params['has_upper'] = params['upper'] != np.inf

    invalid_cov = (
        'has_covariance & (has_equality | has_sum | has_increasing | has_probability)')

    assert len(params.query(invalid_cov)) == 0, (
        'covariance constraints are not compatible with other constraints')
