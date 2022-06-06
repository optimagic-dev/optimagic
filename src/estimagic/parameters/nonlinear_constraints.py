import numpy as np
from estimagic.differentiation.derivatives import first_derivative


def process_nonlinear_constraints(
    nonlinear_constraints, params, converter, numdiff_options
):

    processed = []

    for c in nonlinear_constraints:

        _check_validity_nonlinear_constraint(c)

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
                # parameter bounds etc.
                return first_derivative(constraint_fun, p, **numdiff_options)[
                    "derivative"
                ]

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

            def constraint_from_internal(x):
                params = converter.params_from_internal(x)
                select = selector(params)
                return constraint_fun(select) - c["value"]

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

            transform_constraint_fun = get_positivity_transform(
                is_pos_constr, lower_bounds, upper_bounds, case="fun"
            )
            transform_jacobian = get_positivity_transform(
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
        }

        processed.append(internal_constr)
    return processed


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


def get_positivity_transform(is_pos_constr, lower_bounds, upper_bounds, case):
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
