"""Process the user provided pc for use during the optimization.

The main purpose of this module is to convert the user provided constraints into
inputs for fast reparametrization functions. In the process, the constraints are
checked and consolidated. Consolidation means that redundant constraints are dropped
and other constraints are collected in meaningful bundles.

To improve readability, the actual code for checking and consolidation are in separate
modules.

Calls to functions doing checking are scattered across the module.
This is in order to perform each check as soon as it becomes possible, which allows
errors to be raised at a point where constraints still look similar to
what users wrote. However, some checks can only be done
after consolidation.

The challenge in making this module readable is that after each function is applied
the list of constraints and the params dataframe will be slightly different, but it
is not possible to reflect all of the changes in meaningful names because thy would
get way too long. We chose the following conventions:

As soon as a list of constraints or a params DataFrame is different from what the
user provided they are called pc (for processed constraints) and pp (for processed
params) respectively. If all constraints of a certain type, say linear, are collected
this collection is called pc_linear.

If only few columns of processed params are used in a function, it is better to
pass them as Series, to make the flow of information more explicit.

"""
import warnings

import numpy as np
import pandas as pd
from estimagic.parameters.check_constraints import check_constraints_are_satisfied
from estimagic.parameters.check_constraints import check_fixes_and_bounds
from estimagic.parameters.check_constraints import check_for_incompatible_overlaps
from estimagic.parameters.check_constraints import check_types
from estimagic.parameters.consolidate_constraints import consolidate_constraints
from estimagic.parameters.kernel_transformations import scale_to_internal
from estimagic.parameters.parameter_preprocessing import add_default_bounds_to_params
from estimagic.utilities import number_of_triangular_elements_to_dimension


def process_constraints(
    constraints,
    parvec,
    scaling_factor=None,
    scaling_offset=None,
):
    """Process, consolidate and check constraints.

    Args:
        constraints (list): List of dictionaries where each dictionary is a constraint.
        parvec (np.ndarray): 1d numpy array with flattened params.
        scaling_factor (np.ndarray or None): If None, no scaling factor is used.
        scaling_offset (np.ndarray or None): If None, no scaling offset is used.

    Returns:

        transformations (list): A processed version of those constraints
            that entail actual transformations and not just fixing parameters.
        constr_info (dict): Dict of 1d numpy arrays of length n_params (or None) with
            information that is needed for the reparametrizations.
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
            - _is_fixed_to_value: True for parameters that are fixed to a value
            - _is_fixed_to_other: True for parameters that are fixed to another
              parameter

    """
    parvec = add_default_bounds_to_params(parvec)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="indexing past lexsort depth may impact performance."
        )
        parvec = parvec.copy()
        check_types(constraints)
        # selectors have to be processed before anything else happens to the params
        transformations = _process_selectors(constraints, parvec)

        transformations = _replace_pairwise_equality_by_equality(transformations)
        transformations = _process_linear_weights(transformations, parvec)
        check_constraints_are_satisfied(transformations, parvec)
        transformations = _replace_increasing_and_decreasing_by_linear(transformations)
        transformations = _process_linear_weights(transformations, parvec)
        transformations, constr_info = consolidate_constraints(transformations, parvec)
        check_for_incompatible_overlaps(constr_info, transformations)
        check_fixes_and_bounds(constr_info, transformations)

        int_lower, int_upper = _create_unscaled_internal_bounds(
            constr_info.lower_bound, constr_info.upper_bound, transformations
        )
        constr_info["_internal_lower"] = int_lower
        constr_info["_internal_upper"] = int_upper
        # ==============================================================================
        constr_info = {
            name: constr_info[name].to_numpy() for name in constr_info.columns
        }  # xxxx
        # ==============================================================================
        constr_info["_internal_free"] = _create_internal_free(
            constr_info["_is_fixed_to_value"],
            constr_info["_is_fixed_to_other"],
            transformations,
        )

        for col in ["_internal_lower", "_internal_upper"]:
            constr_info[col] = _scale_bound_to_internal(
                constr_info[col],
                constr_info["_internal_free"],
                scaling_factor=scaling_factor,
                scaling_offset=scaling_offset,
            )
        constr_info["_pre_replacements"] = _create_pre_replacements(
            constr_info["_internal_free"]
        )

        constr_info["_internal_fixed_value"] = _create_internal_fixed_value(
            constr_info["_fixed_value"], transformations
        )

        return transformations, constr_info


def _process_selectors(constraints, params):
    """Convert the query and loc field of the constraint into position based indices.

    Args:
        constraints (list): List of dictionaries where each dictionary is a constraint.
        params (pd.DataFrame): see :ref:`params`.

    Returns:
        list: The resulting constraint dictionaries contain a new entry
            called 'index' that consists of the positions of the selected parameters.
            If the selected parameters are consecutive entries, the value corresponding
            to 'index' is a list of positions.

    """
    out = []

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
                    "All sets of parameters in pairwise_equality pc must have "
                    "the same length."
                )
        else:
            assert (
                len(indices) == 1
            ), "Either loc or query can be in constraint but not both."

        n_selected = len(indices[0])
        if n_selected >= 1:
            if constr["type"] == "pairwise_equality":
                new_constr["indices"] = indices
            else:
                new_constr["index"] = indices[0]
            out.append(new_constr)

    return out


def _replace_pairwise_equality_by_equality(constraints):
    """Rewrite pairwise equality constraints to equality constraints.

    Args:
        pc (list): List of dictionaries where each dictionary is a constraint.
            It is assumed that the selectors in constraints were already processed.

    Returns:
        list: List of processed constraints.

    """
    pairwise_constraints = [c for c in constraints if c["type"] == "pairwise_equality"]
    constraints = [c for c in constraints if c["type"] != "pairwise_equality"]
    for constr in pairwise_constraints:
        equality_constraints = []
        for elements in zip(*constr["indices"]):
            equality_constraints.append({"index": list(elements), "type": "equality"})
        constraints += equality_constraints

    return constraints


def _process_linear_weights(constraints, params):
    """Harmonize the weights of linear constraints.

    Args:
        pc (list): Constraints where the selector have already been processed.
        params (pd.DataFrame): see :ref:`params`.

    Returns:
        processed (list): Constraints where all weights are Series.

    """
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
                weights = np.full(len(params_subset), float(raw_weights))
            else:
                raise TypeError(f"Invalid type for linear weights {type(raw_weights)}.")

            new_constr = constr.copy()
            weights_sr = pd.Series(weights, index=params_subset.index)
            new_constr["weights"] = weights_sr
            processed.append(new_constr)
        else:
            processed.append(constr)

    return processed


def _replace_increasing_and_decreasing_by_linear(constraints):
    """Write increasing and decreasing constraints as linear constraints.

    Args:
        pc (list): Constraints where the selectors have already been processed.

    Returns:
        processed (list)

    """
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
            linear_constr = {
                "index": [smaller, larger],
                "type": "linear",
                "weights": np.array([-1, 1]),
                "lower_bound": 0,
            }
            linear_constraints.append(linear_constr)

    processed = linear_constraints + other_constraints
    return processed


def _create_unscaled_internal_bounds(lower, upper, constraints):
    """Create columns with bounds for the internal parameter vector.

    The columns have the length of the external params and will be reduced later.

    Args:
        lower (np.ndarray): Processed and consolidated external lower bounds.
        upper (np.ndarray): Processed and consolidated external upper bounds.
        pc (pd.DataFrame): Processed and consolidated constraints.

    Returns:
        int_lower (np.ndarray): Lower bound of internal parameters.
        int_upper (np.ndarray): Upper bound of internal parameters.

    """
    int_lower, int_upper = lower.copy(), upper.copy()

    for constr in constraints:
        if constr["type"] in ["covariance", "sdcorr"]:
            # Note that the diagonal positions are the same for covariance and sdcorr
            # because the internal params contains the Cholesky factor of the implied
            # covariance matrix in both cases.
            dim = number_of_triangular_elements_to_dimension(len(constr["index"]))
            diag_positions = [0] + np.cumsum(range(2, dim + 1)).tolist()
            diag_indices = np.array(constr["index"])[diag_positions].tolist()
            bd = constr.get("bounds_distance", 0)
            bd = np.sqrt(bd) if constr["type"] == "covariance" else bd
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


def _create_internal_free(is_fixed_to_value, is_fixed_to_other, constraints):
    """Boolean Series that is true for parameters over which the optimizer optimizes.

    Args:
        is_fixed_to_value (np.ndarray): The _is_fixed_to_value column of pp.
        is_fixed_to_other (np.ndarray): The _is_fixed_to_other column of pp.

    Returns:
        int_free (np.ndarray)
    """
    int_fixed = is_fixed_to_value | is_fixed_to_other

    for constr in constraints:
        if constr["type"] == "probability":
            int_fixed[constr["index"][-1]] = True
        elif constr["type"] == "linear":
            int_fixed[constr["index"]] = False
            relevant_index = constr["index"][-len(constr["right_hand_side"]) :]
            int_fixed[relevant_index] = np.isfinite(constr["right_hand_side"]["value"])

    int_free = ~int_fixed

    return int_free


def _create_pre_replacements(internal_free):
    """Series with internal position of parameters.

    The j_th element indicates the position of the internal parameter that has to be
    copied into the j_th position of the external parameter vector when reparametrizing
    from_internal, before any transformations are applied. Negative if no element has
    to be copied.

    This will be used to copy the free internal parameters into a parameter vector
    that has the same length as all params.

    Args:
        internal_free (np.ndarray): The _internal_free column of the processed params.

    """
    pre_replacements = np.full(len(internal_free), -1)
    pre_replacements[internal_free] = np.arange(internal_free.sum())

    return pre_replacements


def _create_internal_fixed_value(fixed_value, constraints):
    """Pandas Series containing the values to which internal parameters are fixe.

    This contains additional fixes used to enforce other constraints and (potentially
    transformed) user specified fixed values.

    Args:
        fixed_value (np.ndarray): The (external) _fixed_value column of pp.
        constraints (list): Processed and consolidated params.

    """
    int_fix = fixed_value.copy()
    for constr in constraints:
        if constr["type"] == "probability":
            int_fix[constr["index"][-1]] = 1
        elif constr["type"] in ["covariance", "sdcorr"]:
            int_fix[constr["index"][0]] = np.sqrt(int_fix[constr["index"][0]])
        elif constr["type"] == "linear":
            int_fix[constr["index"]] = np.nan
            relevant_index = constr["index"][-len(constr["right_hand_side"]) :]
            int_fix[relevant_index] = constr["right_hand_side"]["value"].to_numpy()

    return int_fix


def _scale_bound_to_internal(bounds, internal_free, scaling_factor, scaling_offset):
    _bounds = bounds.copy()
    free_bounds = bounds[internal_free]

    scaled = scale_to_internal(free_bounds, scaling_factor, scaling_offset)

    _bounds[internal_free] = scaled
    return _bounds
