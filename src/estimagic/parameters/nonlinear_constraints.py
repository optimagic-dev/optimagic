from functools import partial

import numpy as np
import pandas as pd
from estimagic.differentiation.derivatives import first_derivative
from estimagic.exceptions import InvalidFunctionError


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
    skip_checks,
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
        skip_checks (bool): Whether checks on the inputs are skipped. This makes the
            optimization faster, especially for very fast constraint functions. Default
            False.

    Returns:
        list[dict]: List of processed constraints.

    """
    for c in nonlinear_constraints:
        _check_validity_nonlinear_constraint(c, params=params, skip_checks=skip_checks)

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

    constraint_fun = c["func"]
    selector = _process_selector(c)

    # ==================================================================================
    # Retrieve number of constraints
    # ==================================================================================
    if "n_constr" in c:
        _n_constr = c["n_constr"]
    else:
        constraint_value = constraint_fun(selector(params))
        _n_constr = len(np.atleast_1d(constraint_value))

    # ==================================================================================
    # Consolidate and transform jacobian
    # ==================================================================================
    if "derivative" in c:
        if not callable(c["derivative"]):
            msg = "Jacobian of constraints needs to be callable."
            raise ValueError(msg)
        jacobian = c["derivative"]
    else:
        # use finite-differences if no closed-form jacobian is defined
        def jacobian(p):
            return first_derivative(
                constraint_fun,
                p,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                **numdiff_options,
            )["derivative"]

    def _jacobian_from_internal(x):
        params = converter.params_from_internal(x)
        select = selector(params)
        return np.atleast_1d(jacobian(select)).reshape(_n_constr, -1)

    # ==================================================================================
    # Transform constraint function and derive bounds
    # ==================================================================================
    _type = "eq" if "value" in c else "ineq"

    if _type == "eq":

        # ==============================================================================
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

        # ==============================================================================
        # Inequality constraints
        #
        # We define the internal constraint function to be satisfied if it is
        # greater than or equal to zero (positivity constraint). If the bounds already
        # satify this condition we do not change anything, otherwise we need to perform
        # a transformation.

        def _constraint_from_internal(x):
            params = converter.params_from_internal(x)
            select = selector(params)
            return np.atleast_1d(constraint_fun(select))

        lower_bounds = c.get("lower_bounds", 0)
        upper_bounds = c.get("upper_bounds", np.inf)

        transformation = _get_transformation(lower_bounds, upper_bounds)

        constraint_from_internal = _compose_funcs(
            _constraint_from_internal, transformation["func"]
        )

        jacobian_from_internal = _compose_funcs(
            _jacobian_from_internal, transformation["derivative"]
        )

        n_constr = 2 * _n_constr if transformation.name == "stack" else _n_constr

    internal_constr = {
        "n_constr": n_constr,
        "type": _type,
        "fun": constraint_from_internal,  # internal name for 'func'
        "jac": jacobian_from_internal,  # internal name for 'derivative'
        "tol": c.get("tol", CONVERGENCE_ABSOLUTE_CONSTRAINT_TOLERANCE),
    }

    return internal_constr


def equality_as_inequality_constraints(nonlinear_constraints):
    """Return constraints where equality constraints are converted to inequality."""
    constraints = [_equality_to_inequality(c) for c in nonlinear_constraints]
    return constraints


def _equality_to_inequality(c):
    """Transform a single constraint.

    An equality constaint g(x) = 0 can be transformed to two inequality constraints
    using (g(x), -g(x)) >= 0. Hence, the number of constraints doubles, and the
    constraint functions itself as well as the derivative need to be updated.

    """
    if c["type"] == "eq":

        def transform(x, func):
            return np.concatenate((func(x), -func(x)), axis=0)

        out = {
            "fun": partial(transform, func=c["fun"]),
            "jac": partial(transform, func=c["jac"]),
            "n_constr": 2 * c["n_constr"],
            "tol": c["tol"],
            "type": "ineq",
        }
    else:
        out = c
    return out


def _compose_funcs(f, g):
    return lambda x: f(g(x))


def _get_transformation(lower_bounds, upper_bounds):
    """Get transformation given bounds.

    The internal inequality constraint is defined as h(x) >= 0. However, the user can
    specify: a <= g(x) <= b. To get the internal represenation we need to transform the
    constraint.

    """
    transformation_type = _get_transformation_type(lower_bounds, upper_bounds)

    if transformation_type == "identity":
        transformer = {"func": _identity, "derivative": _identity}
    elif transformation_type == "subtract_lb":
        transformer = {
            "func": lambda v: v - lower_bounds,
            "derivative": _identity,
        }
    elif transformation_type == "stack":
        transformer = {
            "func": lambda v: np.concatenate(
                (v - lower_bounds, upper_bounds - v), axis=0
            ),
            "derivative": lambda v: np.concatenate((v, -v), axis=0),
        }
    return transformer


def _get_transformation_type(lower_bounds, upper_bounds):
    lb_is_zero = not np.count_nonzero(lower_bounds)
    ub_is_inf = np.all(np.isposinf(upper_bounds))

    if lb_is_zero and ub_is_inf:
        # the external constraint is already in the correct format
        _transformation_type = "identity"
    elif ub_is_inf:
        # the external constraint can be transformed by subtraction
        _transformation_type = "subtract_lb"
    else:
        # the external constraint can only be transformed by duplication (stacking)
        _transformation_type = "stack"
    return _transformation_type


def _process_selector(c):
    if "selector" in c:
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


def _check_validity_nonlinear_constraint(c, params, skip_checks):
    # ==================================================================================
    # check functions
    # ==================================================================================

    if "func" not in c:
        raise ValueError(
            "Constraint needs to have entry 'fun', representing the constraint "
            "function."
        )
    if not callable(c["func"]):
        raise ValueError("Entry 'fun' in nonlinear constraints has be callable.")

    if "derivative" in c and not callable(c["derivative"]):
        raise ValueError("Entry 'jac' in nonlinear constraints has be callable.")

    # ==================================================================================
    # check bounds
    # ==================================================================================

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

    if "lower_bounds" in c and "upper_bounds" in c:
        if not np.all(np.array(c["lower_bounds"]) <= np.array(c["upper_bounds"])):
            raise ValueError(
                "If lower bounds need to less than or equal to upper bounds."
            )

    # ==================================================================================
    # check selector
    # ==================================================================================

    if "selector" in c:
        if not callable(c["selector"]):
            raise ValueError(
                f"'selector' entry needs to be callable in constraint {c}."
            )
        else:
            try:
                c["selector"](params)
            except Exception:
                raise InvalidFunctionError(
                    "Error when calling 'selector' function on params in constraint "
                    f" {c}"
                )

    elif "loc" in c:
        if not isinstance(params, (pd.Series, pd.DataFrame)):
            raise ValueError(
                "params needs to be pd.Series or pd.DataFrame to use 'loc' selector in "
                f"in consrtaint {c}."
            )
        try:
            params.loc[c["loc"]]
        except (KeyError, IndexError):
            raise ValueError("'loc' string is invalid.")

    elif "query" in c:
        if not isinstance(params, pd.DataFrame):
            raise ValueError(
                "params needs to be pd.DataFrame to use 'query' selector in "
                f"constraints {c}."
            )
        try:
            params.query(c["query"])
        except Exception:
            raise ValueError(f"'query' string is invalid in constraint {c}.")

    # ==================================================================================
    # check that constraints can be evaluated
    # ==================================================================================

    if not skip_checks:

        selector = _process_selector(c)

        try:
            c["func"](selector(params))
        except Exception:
            raise InvalidFunctionError(
                f"Error when evaluating function of constraint {c}."
            )
