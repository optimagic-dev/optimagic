from functools import partial

import numpy as np
from estimagic.differentiation.derivatives import first_derivative


CONVERGENCE_ABSOLUTE_CONSTRAINT_TOLERANCE = 1e-5
"""float: Allowed tolerance of the equality and inequality constraints for values to be
considered 'feasible'.

"""


def process_nonlinear_constraints(
    nonlinear_constraints,
    params,
    converter,
    numdiff_options,
    lower_bounds,
    upper_bounds,
):
    """Process and prepare nonlinear constraints for internal use.

    Args:
        nonlinear_constraints (list[dict]): List of dictionaries, each representing a
            nonlinear constraint.
        params (pandas): A pytree containing the parameters with respect to which the
            criterion is optimized. Examples are a numpy array, a pandas Series,
            a DataFrame with "value" column, a float and any kind of (nested) dictionary
            or list containing these elements. See :ref:`params` for examples.
        converter (Converter): NamedTuple with methods to convert between internal and
            external parameters, derivatives and function outputs.
        numdiff_options (dict): Keyword arguments for the calculation of numerical
            derivatives. See :ref:`first_derivative` for details. Note that the default
            method is changed to "forward" for speed reasons.
        lower_bounds (pytree): A pytree with the same structure as params with lower
            bounds for the parameters. Can be ``-np.inf`` for parameters with no lower
            bound. These are only used for the internal finite difference derivative.
        upper_bounds (pytree): As lower_bounds. Can be ``np.inf`` for parameters with
            no upper bound.

    Returns:
        list[dict]: List of processed constraints.

    """
    for c in nonlinear_constraints:
        _check_validity_nonlinear_constraint(c)

    _processor = partial(
        _process_nonlinear_constraint,
        params=params,
        converter=converter,
        numdiff_options=numdiff_options,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )

    processed = [_processor(c) for c in nonlinear_constraints]
    return processed


def _process_nonlinear_constraint(
    c, params, converter, numdiff_options, lower_bounds, upper_bounds
):
    """Process a single nonlinear constraint."""

    constraint_fun = c["fun"]
    selector = _process_selector(c)

    ################################################################################
    # Retrieve number of constraints
    ################################################################################
    if "n_constr" in c:
        _n_constr = c["n_constr"]
    else:
        constraint_value = constraint_fun(selector(params))
        _n_constr = len(np.atleast_1d(constraint_value))

    ################################################################################
    # Consolidate and transform jacobian
    ################################################################################
    if "jac" in c:
        if not callable(c["jac"]):
            msg = "Jacobian of constraints needs to be callable."
            raise ValueError(msg)
        jacobian = c["jac"]
    else:
        # use finite-differences if no closed-form jacobian is defined
        def jacobian(p):
            return first_derivative(
                constraint_fun,
                p,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                **numdiff_options
            )["derivative"]

    def _jacobian_from_internal(x):
        params = converter.params_from_internal(x)
        select = selector(params)
        return np.atleast_1d(jacobian(select)).reshape(_n_constr, -1)

    ################################################################################
    # Transform constraint function and derive bounds
    ################################################################################
    _type = "eq" if "value" in c else "ineq"

    if _type == "eq":

        ############################################################################
        # Equality constraints
        #
        # We define the internal constraint function to be satisfied if it is equal
        # to zero, by subtracting the fixed value.

        _value = np.atleast_1d(np.array(c["value"], dtype=float))

        def constraint_from_internal(x):
            params = converter.params_from_internal(x)
            select = selector(params)
            out = np.atleast_1d(constraint_fun(select)) - _value
            return out

        jacobian_from_internal = _jacobian_from_internal
        n_constr = _n_constr

    else:

        ############################################################################
        # Inequality constraints
        #
        # We define the internal constraint function to be satisfied if it is
        # greater or equal to zero (positivity constraint). If the bounds already
        # satify this condition we do not change anything, otherwise we subtract the
        # bounds and stack the constraint on top of itself with a sign switch.

        def _constraint_from_internal(x):
            params = converter.params_from_internal(x)
            select = selector(params)
            return np.atleast_1d(constraint_fun(select))

        lower_bounds = c.get("lower_bounds", 0)
        upper_bounds = c.get("upper_bounds", np.inf)

        is_pos_constr = _is_positivity_constraint(lower_bounds, upper_bounds)

        transform_constraint_fun = _get_positivity_transform(
            is_pos_constr, lower_bounds, upper_bounds, case="fun"
        )
        transform_jacobian = _get_positivity_transform(
            is_pos_constr, lower_bounds, upper_bounds, case="jac"
        )

        def constraint_from_internal(x):
            return transform_constraint_fun(_constraint_from_internal(x))

        def jacobian_from_internal(x):
            return transform_jacobian(_jacobian_from_internal(x))

        n_constr = _n_constr if is_pos_constr else 2 * _n_constr

    internal_constr = {
        "n_constr": n_constr,
        "type": _type,
        "fun": constraint_from_internal,
        "jac": jacobian_from_internal,
        "tol": c.get("tol", CONVERGENCE_ABSOLUTE_CONSTRAINT_TOLERANCE),
    }

    return internal_constr


def equality_as_inequality_constraints(nonlinear_constraints):
    """Return constraints where equality constraints are converted to inequality."""
    constraints = []
    for c in nonlinear_constraints:
        if c["type"] == "eq":

            def _fun(x):
                value = c["fun"]
                return np.concatenate((value, -value), axis=0)

            def _jac(x):
                value = c["jac"]
                return np.concatenate((value, -value), axis=0)

            _c = {
                "fun": _fun,
                "jac": _jac,
                "n_constr": 2 * c["n_constr"],
                "type": "ineq",
            }
        else:
            _c = c
        constraints.append(_c)
    return constraints


def _get_positivity_transform(is_pos_constr, lower_bounds, upper_bounds, case):
    if is_pos_constr:
        _transform = _identity
    elif case == "fun":

        def _transform(value):
            return np.concatenate((value - lower_bounds, upper_bounds - value), axis=0)

    elif case == "jac":

        def _transform(value):
            return np.concatenate((value, -value), axis=0)

    return _transform


def _is_positivity_constraint(lower_bounds, upper_bounds):
    """Returns True if bounds define a positivity constraint, and False otherwise.

    A positivity constraint for a function g(x) is defined via g(x) >= 0. With lower and
    upper bounds this is expressed via 0 = lb <= g(x) <= ub = infinity.

    """
    lb_is_zero = not np.count_nonzero(lower_bounds)
    ub_is_inf = np.all(upper_bounds == np.inf)
    return lb_is_zero and ub_is_inf


def _process_selector(c):
    if "selector" in c:
        if not callable(c["selector"]):
            raise ValueError("'selector' entry in constraints needs to be callable.")
        selector = c["selector"]
    elif "loc" in c:

        def selector(params):
            return params.loc[c["loc"]]

    elif "query" in c:

        def selector(params):
            return params.query(c["query"])

    else:
        selector = _identity
    return selector


def _identity(x):
    return x


def _check_validity_nonlinear_constraint(c):
    if "fun" not in c:
        raise ValueError(
            "Constraint needs to have entry 'fun', representing the constraint "
            "function."
        )
    if not callable(c["fun"]):
        raise ValueError("Entry 'fun' in nonlinear constraints has be callable.")

    if "jac" in c and not callable(c["jac"]):
        raise ValueError("Entry 'jac' in nonlinear constraints has be callable.")

    is_equality_constraint = "value" in c

    if is_equality_constraint:
        if "lower_bounds" in c or "upper_bounds" in c:
            raise ValueError(
                "Only one of 'value' or ('lower_bounds', 'upper_bounds') can be "
                "passed to a nonlinear constraint."
            )

    if not is_equality_constraint:
        if "lower_bounds" not in c and "upper_bounds" not in c:
            raise ValueError(
                "For inequality constraint at least one of ('lower_bounds', "
                "'upper_bounds') has to be passed to the nonlinear constraint."
            )
