import numpy as np
from estimagic.differentiation.derivatives import first_derivative


def process_nonlinear_constraints(nonlinear_constraints, params, converter):
    _check_nonlinear_constraints(nonlinear_constraints)

    processed = []

    for c in nonlinear_constraints:

        _constraint_fun = c["fun"]
        _c = {}

        ################################################################################
        # process selector
        ################################################################################

        if "selector" in c:
            _selector = c["selector"]
        else:
            _selector = _identity_selector

        ################################################################################
        # retrieve number of constraints
        ################################################################################

        if "n_constr" in c:
            _c["n_constr"] = c["n_constr"]
        else:
            constraint_value = _constraint_fun(_selector(params))
            _c["n_constr"] = len(np.atleast_1d(constraint_value))

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
            return _jacobian(select).reshape(_c["n_constr"], -1)

        ################################################################################
        # consolidate and transform jacobian
        ################################################################################

        _type = "eq" if "value" in c else "ineq"
        _c["type"] = _type
        _c["jac"] = _jacobian_from_internal

        ################################################################################
        # transform constraint function and derive bounds
        ################################################################################

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

            _c["lower_bound"] = c.get("lower_bound", np.tile(-np.inf, _c["n_constr"]))
            _c["upper_bound"] = c.get("upper_bound", np.tile(np.inf, _c["n_constr"]))

        _c["fun"] = _constraint_from_internal

        processed.append(_c)
    return processed


def transform_bounds_to_positivity_constraint(nonlinear_constraints):
    _transformed = []
    for c in nonlinear_constraints:
        if c["type"] == "ineq" and is_positivity_constraint(c):

            _c = c.copy()
            del _c["lower_bound"]
            del _c["upper_bound"]
            _transformed.append(_c)

        elif not is_positivity_constraint(c):

            def _long_constraint_fun(x):
                value = np.atleast_1d(c["fun"](x))
                return np.concatenate(
                    (value - c["lower_bound"], -value + c["upper_bound"])
                )

            def _long_jacobian(x):
                value = c["jac"](x)
                return np.concatenate((value, -value), axis=0)

            _c = c.copy()
            del _c["lower_bound"]
            del _c["upper_bound"]

            _c["fun"] = _long_constraint_fun
            _c["jac"] = _long_jacobian

            _transformed.append(_c)
        else:
            _transformed.append(c)
    return _transformed


def is_positivity_constraint(constr):
    lb_is_zero = not np.count_nonzero(constr["lower_bound"])
    if lb_is_zero and np.all(constr["upper_bound"] == np.inf):
        out = True
    else:
        out = False
    return out


def _check_nonlinear_constraints(nonlinear_constraints):
    for c in nonlinear_constraints:
        assert c["type"] == "nonlinear"
        assert callable(c["fun"])

        is_equality_constraint = "value" in c
        is_inequality_constraint = not is_equality_constraint

        if is_equality_constraint:
            assert "lower_bound" not in c and "upper_bound" not in c

        if is_inequality_constraint:
            assert "lower_bound" in c or "upper_bound" in c


def _identity_selector(params):
    return params
