from functools import partial
from itertools import product

import numpy as np
import pandas as pd
import pytest
from estimagic.differentiation.derivatives import first_derivative
from estimagic.parameters.parameter_conversion import get_derivative_conversion_function
from estimagic.parameters.parameter_conversion import get_reparametrize_functions
from estimagic.parameters.process_constraints import process_constraints
from estimagic.parameters.reparametrize import _multiply_from_left
from estimagic.parameters.reparametrize import _multiply_from_right
from estimagic.parameters.reparametrize import post_replace
from estimagic.parameters.reparametrize import post_replace_jacobian
from estimagic.parameters.reparametrize import pre_replace
from estimagic.parameters.reparametrize import pre_replace_jacobian
from estimagic.parameters.reparametrize import reparametrize_from_internal
from estimagic.parameters.reparametrize import reparametrize_to_internal
from numpy.testing import assert_array_almost_equal as aaae


to_test = list(
    product(
        [
            "basic_probability",
            "uncorrelated_covariance",
            "basic_covariance",
            "basic_fixed",
            "basic_increasing",
            "basic_equality",
            "query_equality",
            "basic_sdcorr",
            "normalized_covariance",
        ],
        [0, 1, 2],
    )
)


def reduce_params(params, constraints):
    all_locs = []
    for constr in constraints:
        if "query" in constr:
            all_locs = ["i", "j1", "j2"]
        elif isinstance(constr["loc"], tuple):
            all_locs.append(constr["loc"][0])
        elif isinstance(constr["loc"], list):
            all_locs.append(constr["loc"][0][0])
        else:
            all_locs.append(constr["loc"])
    all_locs = sorted(set(all_locs))
    return params.loc[all_locs].copy()


@pytest.mark.parametrize("case, number", to_test)
def test_reparametrize_to_internal(example_params, all_constraints, case, number):
    constraints = all_constraints[case]
    params = reduce_params(example_params, constraints)
    params["value"] = params[f"value{number}"]

    keep = params[f"internal_value{number}"].notnull()
    expected_internal_values = params[f"internal_value{number}"][keep]
    expected_internal_lower = params["internal_lower"]
    expected_internal_upper = params["internal_upper"]

    to_internal, _ = get_reparametrize_functions(
        params=params,
        constraints=constraints,
        scaling_factor=None,
        scaling_offset=None,
    )

    _, pp = process_constraints(constraints, params)

    calculated_internal_values_np = to_internal(pp["value"])
    calculated_internal_values_pd = to_internal(pp)

    calculated_internal_lower = pp["_internal_lower"]
    calculated_internal_upper = pp["_internal_upper"]

    aaae(calculated_internal_values_np, calculated_internal_values_pd)
    aaae(calculated_internal_values_np, expected_internal_values)
    aaae(calculated_internal_lower, expected_internal_lower)
    aaae(calculated_internal_upper, expected_internal_upper)


@pytest.mark.parametrize("case, number", to_test)
def test_reparametrize_from_internal(example_params, all_constraints, case, number):
    constraints = all_constraints[case]
    params = reduce_params(example_params, constraints)
    params["value"] = params[f"value{number}"]

    keep = params[f"internal_value{number}"].notnull()

    _, from_internal = get_reparametrize_functions(
        params=params,
        constraints=constraints,
        scaling_factor=None,
        scaling_offset=None,
    )

    internal_p = params[f"internal_value{number}"][keep].to_numpy()
    calculated_external_value = from_internal(internal_p)
    expected_external_value = params["value"].to_numpy()

    aaae(calculated_external_value, expected_external_value)


def test_scaling_cancels_itself():
    params = pd.DataFrame()
    params["value"] = np.arange(10) + 10
    params["lower_bound"] = np.arange(10)
    params["upper_bound"] = 25

    to_internal, from_internal = get_reparametrize_functions(
        params=params,
        constraints=[],
        scaling_factor=np.arange(10) + 1,
        scaling_offset=np.ones(10),
    )

    internal = to_internal(params["value"].to_numpy())
    external = from_internal(internal)

    aaae(external, params["value"].to_numpy())


@pytest.mark.parametrize("case, number", to_test)
def test_reparametrize_from_internal_jacobian(
    example_params, all_constraints, case, number
):
    constraints = all_constraints[case]
    params = reduce_params(example_params, constraints)
    params["value"] = params[f"value{number}"]

    _, pp = process_constraints(constraints, params)

    n_free = int(pp["_internal_free"].sum())
    scaling_factor = np.ones(n_free) * 2  # np.arange(n_free) + 1
    scaling_offset = np.arange(n_free) - 1

    params_to_internal, params_from_internal = get_reparametrize_functions(
        params=params,
        constraints=constraints,
        scaling_factor=scaling_factor,
        scaling_offset=scaling_offset,
    )

    internal_p = params_to_internal(params["value"].to_numpy())

    numerical_jacobian = first_derivative(params_from_internal, internal_p)

    derivative_to_internal = get_derivative_conversion_function(
        params=params,
        constraints=constraints,
        scaling_factor=scaling_factor,
        scaling_offset=scaling_offset,
    )

    # calling convert_external_derivative with identity matrix as external derivative
    # is just a trick to get out the jacobian of reparametrize_from_internal
    jacobian = derivative_to_internal(
        external_derivative=np.eye(len(params)),
        internal_values=internal_p,
    )

    aaae(jacobian, numerical_jacobian["derivative"])


@pytest.mark.parametrize("case, number", to_test)
def test_pre_replace_jacobian(example_params, all_constraints, case, number):
    constraints = all_constraints[case]
    params = reduce_params(example_params, constraints)
    params["value"] = params[f"value{number}"]

    keep = params[f"internal_value{number}"].notnull()

    pc, pp = process_constraints(constraints, params)

    internal_p = params[f"internal_value{number}"][keep].to_numpy()
    fixed_val = pp["_internal_fixed_value"]
    pre_repl = pp["_pre_replacements"]

    func = partial(
        pre_replace, **{"fixed_values": fixed_val, "pre_replacements": pre_repl}
    )
    numerical_deriv = first_derivative(func, internal_p)["derivative"]
    numerical_deriv[np.isnan(numerical_deriv)] = 0

    deriv = pre_replace_jacobian(pre_repl, len(internal_p))

    aaae(deriv, numerical_deriv)


@pytest.mark.parametrize("case, number", to_test)
def test_post_replace_jacobian(example_params, all_constraints, case, number):
    constraints = all_constraints[case]
    params = reduce_params(example_params, constraints)
    params["value"] = params[f"value{number}"]

    keep = params[f"internal_value{number}"].notnull()

    pc, pp = process_constraints(constraints, params)

    internal_p = params[f"internal_value{number}"][keep].to_numpy()
    fixed_val = pp["_internal_fixed_value"]
    pre_repl = pp["_pre_replacements"]
    post_repl = pp["_post_replacements"]

    external = pre_replace(internal_p, fixed_val, pre_repl)
    external[np.isnan(external)] = 0  # if not set to zero the numerical differentiation
    # fails due to potential np.nan.

    func = partial(post_replace, **{"post_replacements": post_repl})
    numerical_deriv = first_derivative(func, external)

    deriv = post_replace_jacobian(post_repl)

    aaae(deriv, numerical_deriv["derivative"])


def test_linear_constraint():
    params = pd.DataFrame(
        index=pd.MultiIndex.from_product([["a", "b", "c"], [0, 1, 2]]),
        data=[[2], [1], [0], [1], [3], [4], [1], [1], [1.0]],
        columns=["value"],
    )
    params["lower_bound"] = [-1] + [-np.inf] * 8
    params["upper_bound"] = [1] + [np.inf] * 8

    constraints = [
        {"loc": "a", "type": "linear", "weights": [1, -2, 0], "value": 0},
        {"loc": "b", "type": "linear", "weights": 1 / 3, "upper_bound": 3},
        {
            "loc": "c",
            "type": "linear",
            "weights": 1,
            "lower_bound": 0,
            "upper_bound": 5,
        },
        {"loc": params.index, "type": "linear", "weights": 1, "value": 14},
        {"loc": "c", "type": "equality"},
    ]

    internal, external = back_and_forth_transformation_and_assert(params, constraints)
    assert len(internal) == 5


def test_covariance_is_inherited_from_pairwise_equality(example_params):
    params = example_params.loc[["f", "l"]].copy()
    params["value"] = params["value0"]
    constraints = [
        {"loc": "l", "type": "covariance"},
        {"locs": ["l", "f"], "type": "pairwise_equality"},
    ]

    internal, external = back_and_forth_transformation_and_assert(params, constraints)
    assert len(internal) == 10


def back_and_forth_transformation_and_assert(params, constraints):
    pc, pp = process_constraints(constraints, params)

    internal = reparametrize_to_internal(pp["value"], pp["_internal_free"], pc)

    external = reparametrize_from_internal(
        internal=internal,
        fixed_values=pp["_internal_fixed_value"],
        pre_replacements=pp["_pre_replacements"],
        params=params,
        return_numpy=True,
        transformations=pc,
        post_replacements=pp["_post_replacements"],
    )

    aaae(external, params["value"].to_numpy())
    return internal, external


@pytest.mark.parametrize("seed", range(5))
def test_multiply_from_left_and_right(seed):
    np.random.seed(seed)
    mat_list = [np.random.uniform(size=(10, 10)) for i in range(5)]
    a, b, c, d, e = mat_list

    expected = a @ b @ c @ d @ e

    calc_from_left = _multiply_from_left(mat_list)
    calc_from_right = _multiply_from_right(mat_list)

    aaae(calc_from_left, expected)
    aaae(calc_from_right, expected)
