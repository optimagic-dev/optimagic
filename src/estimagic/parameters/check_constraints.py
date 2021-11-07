"""Check compatibility of pc with each other and with bounds and fixes.

See the module docstring of process_constraints for naming conventions.
"""
import warnings

import numpy as np
from estimagic.utilities import cov_params_to_matrix
from estimagic.utilities import sdcorr_params_to_matrix


def check_constraints_are_satisfied(pc, params):
    """Check that params satisfies all constraints.

    This should be called before the more specialized constraints are rewritten to
    linear constraints in order to get better error messages!

    We let the checks pass if all "values" are np.nan. This way `process_constraints`
    can be used on empty params DataFrames which is useful to construct templates for
    start parameters that can be filled out by the user.

    Args:
        pc (list): List of constraints with processed selectors.
        params (pd.DataFrame): See :ref:`params`

    Raises:
        ValueError if constraints are not satisfied.

    """

    if params["value"].notnull().any():
        for constr in pc:
            typ = constr["type"]
            subset = params.iloc[constr["index"]]["value"]
            msg = f"{{}}:\n{subset.to_frame()}"
            if typ == "covariance":
                cov = cov_params_to_matrix(subset)
                e, v = np.linalg.eigh(cov)
                if not np.all(e > -1e-8):
                    raise ValueError(msg.format("Invalid covariance parameters."))
            elif typ == "sdcorr":
                cov = sdcorr_params_to_matrix(subset)
                dim = len(cov)
                if (subset.iloc[:dim] < 0).any():
                    raise ValueError(msg.format("Invalid standard deviations."))
                if ((subset.iloc[dim:] < -1) | (subset.iloc[dim:] > 1)).any():
                    raise ValueError(msg.format("Invalid correlations."))
                e, v = np.linalg.eigh(cov)
                if not np.all(e > -1e-8):
                    raise ValueError(msg.format("Invalid sdcorr parameters."))
            elif typ == "probability":
                if not np.isclose(subset.sum(), 1, rtol=0.01):
                    raise ValueError(msg.format("Probabilities do not sum to 1"))
                if np.any(subset < 0):
                    raise ValueError(msg.format("Negative Probability."))
                if np.any(subset > 1):
                    raise ValueError(msg.format("Probability larger than 1."))
            elif typ == "increasing":
                if np.any(np.diff(subset) < 0):
                    raise ValueError(msg.format("Increasing constraint violated."))
            elif typ == "decreasing":
                if np.any(np.diff(subset) > 0):
                    raise ValueError(msg.format("Decreasing constraint violated"))
            elif typ == "linear":
                # using sr.dot is important in case weights are a series in wrong order
                wsum = subset.dot(constr["weights"])
                if "lower_bound" in constr and wsum < constr["lower_bound"]:
                    raise ValueError(
                        msg.format("Lower bound of linear constraint is violated")
                    )
                elif "upper_bound" in constr and wsum > constr["upper_bound"]:
                    raise ValueError(
                        msg.format("Upper bound of linear constraint violated")
                    )
                elif "value" in constr and not np.isclose(wsum, constr["value"]):
                    raise ValueError(
                        msg.format("Equality condition of linear constraint violated")
                    )
            elif typ == "equality":
                if len(subset.unique()) != 1:
                    raise ValueError(msg.format("Equality constraint violated."))


def check_types(constraints):
    """Check that no invalid constraint types are requested.

    Args:
        constraints (list): List of constraints.

    Raises:
        TypeError if invalid constraint types are encountered



    """
    valid_types = {
        "covariance",
        "sdcorr",
        "linear",
        "probability",
        "increasing",
        "decreasing",
        "equality",
        "pairwise_equality",
        "fixed",
    }
    for constr in constraints:
        if constr["type"] not in valid_types:
            raise TypeError("Invalid constraint_type: {}".format(constr["type"]))


def check_for_incompatible_overlaps(params, pc):
    """Check that there are no overlaps between constraints that transform parameters.

    Since the constraints are already consolidated such that only those that transform
    a parameter are left and all equality constraints are already plugged in, this
    boils down to checking that no parameter appears more than once.

    Args:
        params (pd.DataFrame): see :ref:`params`. Only used for error messages.
        pc (list): List with consolidated constraint dictionaries.

    """
    all_ilocs = []
    for constr in pc:
        all_ilocs += constr["index"]

    msg = (
        "Transforming constraints such as 'covariance', 'sdcorr', 'probability' "
        "and 'linear' cannot overlap. This includes overlaps induced by equality "
        "constraints. This was violated for the following parameters:\n{}"
    )

    if len(set(all_ilocs)) < len(all_ilocs):
        unique, counts = np.unique(all_ilocs, return_counts=True)
        invalid_indices = unique[counts >= 2]
        invalid_names = params.iloc[invalid_indices].index

        raise ValueError(msg.format(invalid_names))


def check_fixes_and_bounds(pp, pc):
    """Check fixes.

    Warn the user if he fixes a parameter to a value even though that parameter has
    a different non-nan value in params

    check that fixes are compatible with other constraints.

    Args:
        pp (pd.DataFrame): see :ref:`params`.
        pc (list): Processed and consolidated constraints.
    """
    # warn about fixes to a different value that what is in the "value" column
    problematic_fixes = pp.query(
        "value != _fixed_value & _fixed_value.notnull() & value.notnull()",
        engine="python",
    )

    warn_msg = (
        "The following parameters were fixed to a different value than their start "
        "value. The start values are overwritten with the fixed values. "
        "You can ignore this message if you did this on purpose. :\n\n {}."
    )

    if len(problematic_fixes) > 0:
        warnings.warn(warn_msg.format(problematic_fixes[["value", "_fixed_value"]]))

    # Check fixes and bounds are compatible with other constraints
    prob_msg = (
        "{} constraints are incompatible with fixes or bounds. "
        "This is violated for:\n{}"
    )

    cov_msg = (
        "{} constraints are incompatible with fixes or bounds except for the first "
        "parameter. This is violated for:\n{}"
    )

    for constr in pc:
        if constr["type"] in ["covariance", "sdcorr"]:
            subset = pp.iloc[constr["index"][1:]]
            if subset["_is_fixed_to_value"].any():
                problematic = subset[subset["_is_fixed_to_value"]].index
                raise ValueError(cov_msg.format(constr["type"], problematic))
            if np.isfinite(subset[["lower_bound", "upper_bound"]]).any(axis=None):
                problematic = (
                    subset.replace([-np.inf, np.inf], np.nan).dropna(how="all").index
                )
                raise ValueError(cov_msg.format(constr["type"], problematic))
        elif constr["type"] == "probability":
            subset = pp.iloc[constr["index"]]
            if subset["_is_fixed_to_value"].any():
                problematic = subset[subset["_is_fixed_to_value"]].index
                raise ValueError(prob_msg.format(constr["type"], problematic))
            if np.isfinite(subset[["lower_bound", "upper_bound"]]).any(axis=None):
                problematic = (
                    subset.replace([-np.inf, np.inf], np.nan).dropna(how="all").index
                )
                raise ValueError(prob_msg.format(constr["type"], problematic))

    invalid = pp.query("lower_bound >= upper_bound")[["lower_bound", "upper_bound"]]
    msg = (
        "lower_bound must be strictly smaller than upper_bound. "
        + f"This is violated for:\n{invalid}"
    )
    if len(invalid) > 0:
        raise ValueError(msg)
