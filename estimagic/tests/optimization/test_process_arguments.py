import numpy as np
import pandas as pd

from estimagic.optimization.process_arguments import process_optimization_arguments


def check_single_argument_types(argument):
    fail_msg = "Type of {} is not correct: is {} but should be {}"
    key_to_expected_tuple = {
        "params": pd.DataFrame,
        "algorithm": str,
        "criterion_kwargs": dict,
        "constraints": list,
        "general_options": dict,
        "algo_options": dict,
        "dashboard": bool,
        "db_options": dict,
    }

    for key, exp_type in key_to_expected_tuple.items():
        actual_type = type(argument[key])
        assert actual_type == exp_type, fail_msg.format(key, actual_type, exp_type)


def test_processing_single_optim_with_all_standard_inputs():
    criterion = np.mean
    params = pd.DataFrame(np.ones(12).reshape(4, 3))
    algorithm = "scipy_L-BFGS-B"

    res = process_optimization_arguments(criterion, params, algorithm)

    check_single_argument_types(res[0])
