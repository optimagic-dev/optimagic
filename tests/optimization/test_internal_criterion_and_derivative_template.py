import functools
import itertools

import numpy as np
import pandas as pd
import pytest
from estimagic.decorators import AlgoInfo
from estimagic.differentiation.derivatives import first_derivative
from estimagic.examples.criterion_functions import sos_criterion_and_gradient
from estimagic.examples.criterion_functions import sos_dict_criterion
from estimagic.examples.criterion_functions import sos_dict_criterion_with_pd_objects
from estimagic.examples.criterion_functions import sos_gradient
from estimagic.examples.criterion_functions import sos_pandas_gradient
from estimagic.examples.criterion_functions import sos_scalar_criterion
from estimagic.optimization.internal_criterion_template import _penalty_contributions
from estimagic.optimization.internal_criterion_template import (
    _penalty_contributions_derivative,
)
from estimagic.optimization.internal_criterion_template import (
    _penalty_root_contributions,
)
from estimagic.optimization.internal_criterion_template import (
    _penalty_root_contributions_derivative,
)
from estimagic.optimization.internal_criterion_template import _penalty_value
from estimagic.optimization.internal_criterion_template import _penalty_value_derivative
from estimagic.optimization.internal_criterion_template import (
    internal_criterion_and_derivative_template,
)
from estimagic.optimization.optimize import _fill_error_penalty_with_defaults
from numpy.testing import assert_array_almost_equal as aaae


def no_second_call(func):
    counter = np.array([0])

    if func is None:
        wrapper_func = None
    else:

        def wrapper_func(params, counter=counter, *args, **kwargs):
            res = func(params, *args, **kwargs)
            if counter[0] >= 1:
                raise AssertionError("This was called twice.")
            counter += 1
            return res

    return wrapper_func


def reparametrize_from_internal(x):
    res = pd.DataFrame()
    res["value"] = x
    return res


def convert_derivative(external_derivative, internal_values):
    return external_derivative


@pytest.fixture()
def base_inputs():
    x = np.arange(5).astype(float)
    params = pd.DataFrame(data=np.arange(5).reshape(-1, 1), columns=["value"])
    inputs = {
        "x": x,
        "params": params,
        "reparametrize_from_internal": reparametrize_from_internal,
        "convert_derivative": convert_derivative,
        "algo_info": AlgoInfo(
            name="my_algorithm",
            primary_criterion_entry="value",
            parallelizes=False,
            needs_scaling=False,
            disable_cache=False,
            is_available=True,
        ),
        "numdiff_options": {},
        "logging": False,
        "db_kwargs": {"database": False, "fast_logging": False, "path": "logging.db"},
        "error_handling": "raise",
        "error_penalty": None,
        "first_criterion_evaluation": {"internal_params": x, "external_params": params},
        "cache": {},
        "cache_size": 10,
        "fixed_log_data": {"stage": "optimization", "substage": 0},
    }
    return inputs


directions = ["maximize", "minimize"]
crits = [sos_dict_criterion, sos_dict_criterion_with_pd_objects, sos_scalar_criterion]
derivs = [sos_gradient, sos_pandas_gradient, None]
crits_and_derivs = [sos_criterion_and_gradient, None]

test_cases = list(itertools.product(directions, crits, derivs, crits_and_derivs))


@pytest.mark.parametrize("direction, crit, deriv, crit_and_deriv", test_cases)
def test_criterion_and_derivative_template(
    base_inputs, direction, crit, deriv, crit_and_deriv
):
    inputs = base_inputs.copy()
    inputs["first_criterion_evaluation"]["output"] = crit(inputs["params"])
    crit = crit if (deriv, crit_and_deriv) == (None, None) else no_second_call(crit)

    inputs["criterion"] = crit
    inputs["derivative"] = no_second_call(deriv)
    inputs["criterion_and_derivative"] = no_second_call(crit_and_deriv)
    inputs["direction"] = direction

    calc_criterion, calc_derivative = internal_criterion_and_derivative_template(
        task="criterion_and_derivative", **inputs
    )

    calc_criterion2 = internal_criterion_and_derivative_template(
        task="criterion", **inputs
    )

    calc_derivative2 = internal_criterion_and_derivative_template(
        task="derivative", **inputs
    )

    if direction == "minimize":
        for c in calc_criterion, calc_criterion2:
            assert c == 30

        for d in calc_derivative, calc_derivative2:
            aaae(d, 2 * np.arange(5))
    else:
        for c in calc_criterion, calc_criterion2:
            assert c == -30

        for d in calc_derivative, calc_derivative2:
            aaae(d, -2 * np.arange(5))


@pytest.mark.parametrize("direction", directions)
def test_internal_criterion_with_penalty(base_inputs, direction):
    inputs = base_inputs.copy()
    scaling = 1 if direction == "minimize" else -1
    inputs["first_criterion_evaluation"]["output"] = scaling * 30

    def raising_crit_and_deriv(x):
        raise ValueError()

    inputs["x"] = inputs["x"] + 10
    inputs["criterion"] = sos_scalar_criterion
    inputs["derivative"] = sos_gradient
    inputs["criterion_and_derivative"] = raising_crit_and_deriv
    inputs["direction"] = direction
    inputs["error_handling"] = "continue"
    inputs["error_penalty"] = _fill_error_penalty_with_defaults(
        error_penalty={},
        first_eval=inputs["first_criterion_evaluation"],
        direction=direction,
    )

    with pytest.warns(None):
        calc_criterion, calc_derivative = internal_criterion_and_derivative_template(
            task="criterion_and_derivative", **inputs
        )

    calc_criterion2 = internal_criterion_and_derivative_template(
        task="criterion", **inputs
    )

    calc_derivative2 = internal_criterion_and_derivative_template(
        task="derivative", **inputs
    )

    norm = np.linalg.norm(np.ones(5) * 10)
    slope = 0.1 if direction == "minimize" else -0.1
    constant = 160 if direction == "minimize" else -160

    x = inputs["x"]
    x0 = np.arange(5)

    expected_crit = constant + slope * norm
    expected_grad = slope * 10 / np.linalg.norm(x - x0)

    if direction == "minimize":
        for c in calc_criterion, calc_criterion2:
            assert c == expected_crit

        for d in calc_derivative, calc_derivative2:
            aaae(d, expected_grad)

    else:
        for c in calc_criterion, calc_criterion2:
            assert c == -expected_crit

        for d in calc_derivative, calc_derivative2:
            aaae(d, -expected_grad)


@pytest.mark.parametrize("seed", range(10))
def test_penalty_aggregations(seed):
    np.random.seed(seed)
    x = np.random.uniform(size=5)
    x0 = np.random.uniform(size=5)
    slope = 0.3
    constant = 3
    dim_out = 10

    scalar = _penalty_value(x, constant, slope, x0)
    contribs = _penalty_contributions(x, constant, slope, x0, dim_out)
    root_contribs = _penalty_root_contributions(x, constant, slope, x0, dim_out)

    assert np.isclose(scalar, contribs.sum())
    assert np.isclose(scalar, (root_contribs**2).sum())


pairs = [
    (_penalty_value, _penalty_value_derivative),
    (_penalty_contributions, _penalty_contributions_derivative),
    (_penalty_root_contributions, _penalty_root_contributions_derivative),
]


@pytest.mark.parametrize("func, deriv", pairs)
def test_penalty_derivatives(func, deriv):
    np.random.seed(1234)
    x = np.random.uniform(size=5)
    x0 = np.random.uniform(size=5)
    slope = 0.3
    constant = 3
    dim_out = 8

    calculated = deriv(x, constant, slope, x0, dim_out)

    partialed = functools.partial(
        func, constant=constant, slope=slope, x0=x0, dim_out=dim_out
    )
    expected = first_derivative(partialed, x)

    aaae(calculated, expected["derivative"])
