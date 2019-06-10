import numpy as np
import pandas as pd

from estimagic.optimization.process_constraints import apply_fixes_to_external_params
from estimagic.optimization.process_constraints import process_constraints
from estimagic.optimization.reparametrize import _equality_to_internal


def make_start_params_helpers(params_index, constraints):
    """Helper DataFrames to generate start params.

    Construct a default params DataFrame and split it into free and parameters and
    parameters that are fixed explicitly or implicitly through equality constraints.

    The free parameters can be exposed to a user to generate custom start parameters
    in a complex model. The fixed part can then be used to transform the user provided
    start parameters into a full params_df.

    Args:
        params_index (DataFrame): The index of a non-internal parameter DataFrame.
            See :ref:`params`.
        constraints (list): A list of constraints

    Returns:
        free (DataFrame): free parameters
        fixed (DataFrame): parameters that are fixed because of explicit fixes
            or equality constraints.

    """
    params = pd.DataFrame(index=params_index)
    params["value"] = np.nan
    params["lower"] = -np.inf
    params["upper"] = np.inf

    constraints = process_constraints(constraints, params)

    fixes = [c for c in constraints if c["type"] == "fixed"]
    params = apply_fixes_to_external_params(params, fixes)

    equality_constraints = [c for c in constraints if c["type"] == "equality"]
    for constr in equality_constraints:
        params.update(_equality_to_internal(params.loc[constr["index"]]))

    # It is a known bug that df.update changes some dtypes: https://tinyurl.com/y66hqxg2
    params["_fixed"] = params["_fixed"].astype(bool)
    free = params.query("~_fixed").drop(columns="_fixed")
    fixed = params.query("_fixed").drop(columns="_fixed")
    return free, fixed


def get_start_params_from_free_params(free, constraints, params_index):
    """Construct a full params df from free parameters, constraints and the param_index.

    Args:
        free (DataFrame): free parameters
        constraints (list): list of constraints
        params_index (DataFrame): The index of a non-internal parameter DataFrame.
            See :ref:`params`.

    Returns:
        params (DataFrame): see :ref:`params`.

    """
    _, fixed = make_start_params_helpers(params_index, constraints)
    fake_params = pd.DataFrame(index=params_index, columns=["value", "lower", "upper"])
    processed_constraints = process_constraints(constraints, fake_params)
    equality_constraints = [c for c in processed_constraints if c["type"] == "equality"]
    params = pd.concat([free, fixed], axis=0).loc[params_index]
    for constr in equality_constraints:
        params_subset = params.loc[constr["index"]]
        values = list(params_subset["value"].value_counts(dropna=True).index)
        assert len(values) <= 1, "Too many values."
        params.loc[constr["index"], "value"] = values[0]
    return params
