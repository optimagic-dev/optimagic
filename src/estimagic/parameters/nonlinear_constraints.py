import numpy as np
from estimagic.differentiation.derivatives import first_derivative


def process_nonlinear_constraints(nonlinear_constraints, params, converter):

    processed = []

    for c in nonlinear_constraints:
        _check_nonlinear_constraint(c)

        new_constr = {}
        _constraint_fun = c["fun"]

        ################################################################################
        # process selector
        ################################################################################
        _selector = _process_selector(c)

        ################################################################################
        # retrieve number of constraints
        ################################################################################
        if "n_constr" in c:
            new_constr["n_constr"] = c["n_constr"]
        else:
            constraint_value = _constraint_fun(_selector(params))
            new_constr["n_constr"] = len(np.atleast_1d(constraint_value))

        ################################################################################
        # consolidate and transform jacobian
        ################################################################################

        if "jac" in c:
            if not callable(c["jac"]):
                msg = "Jacobian of constraints needs to be callable."
                raise ValueError(msg)
            _jacobian = c["jac"]
        else:
            # use finite-differences if no closed-form jacobian is defined
            def _jacobian(p):
                return first_derivative(_constraint_fun, p)["derivative"]

        def _jacobian_from_internal(x):
            params = converter.params_from_internal(x)
            select = _selector(params)
            return _jacobian(select).reshape(new_constr["n_constr"], -1)

        new_constr["jac"] = _jacobian_from_internal

        ################################################################################
        # transform constraint function and derive bounds
        ################################################################################

        _type = "eq" if "value" in c else "ineq"
        new_constr["type"] = _type

        if _type == "eq":

            # define constraint to be equal to zero
            def _constraint_from_internal(x):
                params = converter.params_from_internal(x)
                select = _selector(params)
                return _constraint_fun(select) - c["value"]

        else:

            def _constraint_from_internal(x):
                params = converter.params_from_internal(x)
                select = _selector(params)
                return _constraint_fun(select)

            new_constr["lower_bounds"] = c.get(
                "lower_bounds", np.tile(-np.inf, new_constr["n_constr"])
            )
            new_constr["upper_bounds"] = c.get(
                "upper_bounds", np.tile(np.inf, new_constr["n_constr"])
            )

        new_constr["fun"] = _constraint_from_internal

        processed.append(new_constr)
    return processed


def transform_bounds_to_positivity_constraint(nonlinear_constraints):
    _transformed = []
    for c in nonlinear_constraints:
        if c["type"] == "ineq" and is_positivity_constraint(c):

            new_constr = c.copy()
            del new_constr["lower_bounds"]
            del new_constr["upper_bounds"]
            _transformed.append(new_constr)

        elif not is_positivity_constraint(c):

            def _long_constraint_fun(x):
                value = np.atleast_1d(c["fun"](x))
                return np.concatenate(
                    (value - c["lower_bounds"], -value + c["upper_bounds"])
                )

            def _long_jacobian(x):
                value = c["jac"](x)
                return np.concatenate((value, -value), axis=0)

            new_constr = c.copy()
            del new_constr["lower_bounds"]
            del new_constr["upper_bounds"]

            new_constr["fun"] = _long_constraint_fun
            new_constr["jac"] = _long_jacobian

            _transformed.append(new_constr)
        else:
            _transformed.append(c)
    return _transformed


def is_positivity_constraint(constr):
    lb_is_zero = not np.count_nonzero(constr["lower_bounds"])
    return lb_is_zero and np.all(constr["upper_bounds"] == np.inf)


def _process_selector(c):
    if "selector" in c:
        _selector = c["selector"]
    elif "loc" in c:

        def _selector(params):
            return params.loc[c["loc"]]

    elif "query" in c:

        def _selector(params):
            return params.query(c["query"])

    else:
        _selector = _identity_selector
    return _selector


def _identity_selector(params):
    return params


def _check_nonlinear_constraint(c):
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
