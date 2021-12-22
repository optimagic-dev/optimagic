"""Functions to calculate and apply scaling factors and offsets.

All scaling operations are applied to internal parameter vectors, i.e.
after the reparametrizations for constraints have been applied.

The order of applying the operations is the following:

to internal: (external - scaling_offset) / scaling_factor
from internal: internal * scaling_factor + scaling_offset

"""
import numpy as np
from estimagic.differentiation.derivatives import first_derivative
from estimagic.parameters.parameter_conversion import get_internal_bounds
from estimagic.parameters.parameter_conversion import get_reparametrize_functions


def calculate_scaling_factor_and_offset(
    params,
    constraints,
    criterion,
    method="start_values",
    clipping_value=0.1,
    magnitude=1,
    numdiff_options=None,
    processed_params=None,
    processed_constraints=None,
):
    numdiff_options = {} if numdiff_options is None else numdiff_options
    to_internal, from_internal = get_reparametrize_functions(
        params=params,
        constraints=constraints,
        processed_params=processed_params,
        processed_constraints=processed_constraints,
    )

    x = to_internal(params["value"].to_numpy())

    if method in ("bounds", "gradient"):
        lower_bounds, upper_bounds = get_internal_bounds(
            params, constraints, processed_params=processed_params
        )

    if method == "start_values":
        raw_factor = np.clip(np.abs(x), clipping_value, np.inf)
        scaling_offset = None
    elif method == "bounds":
        raw_factor = upper_bounds - lower_bounds
        scaling_offset = lower_bounds
    elif method == "gradient":
        default_numdiff_options = {
            "scaling_factor": 100,
            "lower_bounds": lower_bounds,
            "upper_bounds": upper_bounds,
            "error_handling": "raise",
        }

        numdiff_options = {**default_numdiff_options, **numdiff_options}

        def func(x):
            p = params.copy(deep=True)
            p["value"] = from_internal(x)
            crit = criterion(p)
            if isinstance(crit, dict):
                crit = crit["value"]
            return crit

        gradient = first_derivative(func, x, **numdiff_options)["derivative"]

        raw_factor = np.clip(np.abs(gradient), clipping_value, np.inf)
        scaling_offset = None

    scaling_factor = raw_factor / magnitude

    return scaling_factor, scaling_offset
