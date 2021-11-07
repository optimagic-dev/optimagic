"""Test optimizations with logging in a temporary database.

- Only minimize
- Only dict criterion
- scipy_lbfgsb and scipy_ls_dogbox
- with and without derivatives

"""
import itertools

import numpy as np
import pandas as pd
import pytest
from estimagic.examples.criterion_functions import sos_dict_criterion
from estimagic.examples.criterion_functions import sos_dict_derivative
from estimagic.exceptions import TableExistsError
from estimagic.optimization.optimize import minimize
from numpy.testing import assert_array_almost_equal as aaae


algorithms = ["scipy_lbfgsb", "scipy_ls_dogbox"]
derivatives = [None, sos_dict_derivative]
test_cases = list(itertools.product(algorithms, derivatives))


@pytest.mark.parametrize("algorithm, derivative", test_cases)
def test_optimization_with_valid_logging(algorithm, derivative):
    res = minimize(
        sos_dict_criterion,
        pd.Series([1, 2, 3], name="value").to_frame(),
        algorithm=algorithm,
        derivative=derivative,
        logging="logging.db",
    )
    aaae(res["solution_params"]["value"].to_numpy(), np.zeros(3))


def test_optimization_with_existing_exsting_database():
    minimize(
        sos_dict_criterion,
        pd.Series([1, 2, 3], name="value").to_frame(),
        algorithm="scipy_lbfgsb",
        logging="logging.db",
        log_options={"if_database_exists": "raise"},
    )

    with pytest.raises(FileExistsError):
        minimize(
            sos_dict_criterion,
            pd.Series([1, 2, 3], name="value").to_frame(),
            algorithm="scipy_lbfgsb",
            logging="logging.db",
            log_options={"if_database_exists": "raise"},
        )


def test_optimization_with_existing_exsting_table():
    minimize(
        sos_dict_criterion,
        pd.Series([1, 2, 3], name="value").to_frame(),
        algorithm="scipy_lbfgsb",
        logging="logging.db",
        log_options={"if_database_exists": "raise"},
    )

    with pytest.raises(TableExistsError):
        minimize(
            sos_dict_criterion,
            pd.Series([1, 2, 3], name="value").to_frame(),
            algorithm="scipy_lbfgsb",
            logging="logging.db",
            log_options={"if_table_exists": "raise"},
        )
