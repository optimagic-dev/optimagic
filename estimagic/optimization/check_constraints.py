import warnings

import pandas as pd


def check_compatibility_of_constraints(constraints, params, fixed):
    """Compatibility checks for constraints.

    Args:
        constraint (dict)
        params (pd.DataFrame): see :ref:`params_df`.
        fixed (pd.DataFrame): Same index as params. The column '_fixed' indicates
            if a parameter is fixed. The column 'value' indicates the value to which
            it is fixed.

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
    constraints which require reparametrizations. The only exception is when a set of
    parameters is pairwise equal to another set of parameters that shares the same
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
    """Check fixes.

    Warn the user if he fixes a parameter to a value even though that parameter has
    a different non-nan value in params.

    """
    fixed = fixed.query("_fixed")
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
