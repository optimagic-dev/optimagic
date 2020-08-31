import numpy as np
import pandas as pd
import pytest

from estimagic.optimization.broadcast_arguments import broadcast_arguments


def check_single_argument_types(argument):
    fail_msg = "Type of {} is not correct: is {} but should be {}"
    key_to_expected_tuple = {
        "params": pd.DataFrame,
        "algorithm": str,
        "criterion_kwargs": dict,
        "constraints": list,
        "numdiff_options": dict,
        "algo_options": dict,
        "error_penalty": dict,
    }

    for key, exp_type in key_to_expected_tuple.items():
        if key in argument:
            actual_type = type(argument[key])
            assert actual_type == exp_type, fail_msg.format(key, actual_type, exp_type)


@pytest.fixture()
def single_non_standard_inputs():
    kwargs = {
        "criterion": np.mean,
        "params": pd.DataFrame(
            np.arange(12).reshape(4, 3), columns=["lower_bound", "value", "upper_bound"]
        ),
        "algorithm": "scipy_L-BFGS-B",
        "criterion_kwargs": {"keepdims": True, "dtype": float},
        "constraints": [
            {"loc": 1, "type": "fixed", "value": 4},
            {"loc": [2, 3], "type": "increasing"},
        ],
        "numdiff_options": {"n_cores": 1},
        "algo_options": {"maxfun": 5000, "maxiter": 1000},
        "error_penalty": {"slope": 0.1, "constant": 100},
    }
    return kwargs


@pytest.fixture()
def multiple_non_standard_inputs():
    kwargs = {
        "criterion": np.mean,
        "params": [
            pd.DataFrame(
                np.arange(12).reshape(4, 3),
                columns=["lower_bound", "value", "upper_bound"],
            ),
            pd.DataFrame(
                np.arange(12).reshape(4, 3),
                columns=["lower_bound", "value", "upper_bound"],
            ),
        ],
        "algorithm": "scipy_L-BFGS-B",
        "criterion_kwargs": [
            {"keepdims": True, "dtype": float},
            {"keepdims": False, "dtype": float},
        ],
        "constraints": [
            {"loc": 1, "type": "fixed", "value": 4},
            {"loc": [2, 3], "type": "increasing"},
        ],
        "numdiff_options": {"n_cores": 1},
        "algo_options": {"maxfun": 5000, "maxiter": 1000},
        "error_penalty": {"slope": 0.1, "constant": 100},
    }
    return kwargs


def test_processing_single_optim_with_all_standard_inputs():
    criterion = np.mean
    params = pd.DataFrame(np.ones(12).reshape(4, 3))
    algorithm = "scipy_L-BFGS-B"

    res = broadcast_arguments(criterion=criterion, params=params, algorithm=algorithm)

    check_single_argument_types(res[0])


def test_processing_single_optim_with_non_standard_inputs(single_non_standard_inputs):
    kwargs = single_non_standard_inputs
    res = broadcast_arguments(**kwargs)

    check_single_argument_types(res[0])
    assert res[0]["constraints"] == single_non_standard_inputs["constraints"]


def test_processing_multiple_optim_with_all_standard_inputs():
    criterion = np.mean
    params = [
        pd.DataFrame(np.ones(12).reshape(4, 3)),
        pd.DataFrame(np.zeros(16).reshape(4, 4)),
    ]

    algorithms = ["scipy_L-BFGS-B", "pygmo_xnes"]

    res = broadcast_arguments(criterion=criterion, params=params, algorithm=algorithms)

    assert len(res) == 2
    check_single_argument_types(res[0])
    check_single_argument_types(res[1])


def test_processing_multi_optim_with_non_standard_inputs_same_constraints_for_all(
    multiple_non_standard_inputs,
):
    kwargs = multiple_non_standard_inputs
    res = broadcast_arguments(**kwargs)

    check_single_argument_types(res[0])
    assert res[0]["constraints"] == kwargs["constraints"]
    assert res[1]["constraints"] == kwargs["constraints"]


def test_processing_multi_optim_with_non_standard_inputs_different_constraints(
    multiple_non_standard_inputs,
):
    kwargs = multiple_non_standard_inputs
    differing_constraints = [
        [{"loc": 1, "type": "fixed", "value": 4}],
        [{"loc": [2, 3], "type": "increasing"}],
    ]
    kwargs["constraints"] = differing_constraints

    res = broadcast_arguments(**kwargs)

    check_single_argument_types(res[0])
    assert res[0]["constraints"] == differing_constraints[0]
    assert res[1]["constraints"] == differing_constraints[1]
