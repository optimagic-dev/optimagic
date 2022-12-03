from itertools import product

import numpy as np
import pandas as pd
import pytest
from estimagic.decorators import switch_sign
from estimagic.examples.criterion_functions import sos_dict_criterion
from estimagic.examples.criterion_functions import sos_scalar_criterion
from estimagic.logging.database_utilities import load_database
from estimagic.logging.database_utilities import read_new_rows
from estimagic.logging.read_log import read_steps_table
from estimagic.optimization.optimize import maximize
from estimagic.optimization.optimize import minimize
from estimagic.optimization.optimize_result import OptimizeResult
from numpy.testing import assert_array_almost_equal as aaae

criteria = [sos_scalar_criterion, sos_dict_criterion]


@pytest.fixture
def params():
    params = pd.DataFrame()
    params["value"] = np.arange(4)
    params["soft_lower_bound"] = [-5] * 4
    params["soft_upper_bound"] = [10] * 4
    return params


test_cases = product(criteria, ["maximize", "minimize"])


@pytest.mark.parametrize("criterion, direction", test_cases)
def test_multistart_minimize_with_sum_of_squares_at_defaults(
    criterion, direction, params
):
    if direction == "minimize":
        res = minimize(
            criterion=criterion,
            params=params,
            algorithm="scipy_lbfgsb",
            multistart=True,
        )
    else:
        res = maximize(
            criterion=switch_sign(sos_dict_criterion),
            params=params,
            algorithm="scipy_lbfgsb",
            multistart=True,
        )

    assert hasattr(res, "multistart_info")
    ms_info = res.multistart_info
    assert len(ms_info["exploration_sample"]) == 40
    assert len(ms_info["exploration_results"]) == 40
    assert all(isinstance(entry, float) for entry in ms_info["exploration_results"])
    assert all(isinstance(entry, OptimizeResult) for entry in ms_info["local_optima"])
    assert all(isinstance(entry, pd.DataFrame) for entry in ms_info["start_parameters"])
    assert np.allclose(res.criterion, 0)
    aaae(res.params["value"], np.zeros(4))


def test_multistart_with_existing_sample(params):
    sample = pd.DataFrame(
        np.arange(20).reshape(5, 4) / 10,
        columns=params.index,
    )
    options = {"sample": sample}

    res = minimize(
        criterion=sos_dict_criterion,
        params=params,
        algorithm="scipy_lbfgsb",
        multistart=True,
        multistart_options=options,
    )

    calc_sample = _params_list_to_aray(res.multistart_info["exploration_sample"])
    aaae(calc_sample, options["sample"])


def test_convergence_via_max_discoveries_works(params):
    options = {
        "convergence_relative_params_tolerance": np.inf,
        "convergence_max_discoveries": 2,
    }

    res = maximize(
        criterion=switch_sign(sos_dict_criterion),
        params=params,
        algorithm="scipy_lbfgsb",
        multistart=True,
        multistart_options=options,
    )

    assert len(res.multistart_info["local_optima"]) == 2


def test_steps_are_logged_as_skipped_if_convergence(params):
    options = {
        "convergence_relative_params_tolerance": np.inf,
        "convergence_max_discoveries": 2,
    }

    minimize(
        criterion=sos_dict_criterion,
        params=params,
        algorithm="scipy_lbfgsb",
        multistart=True,
        multistart_options=options,
        logging="logging.db",
    )

    steps_table = read_steps_table("logging.db")
    expected_status = ["complete", "complete", "complete", "skipped", "skipped"]
    assert steps_table["status"].tolist() == expected_status


def test_all_steps_occur_in_optimization_iterations_if_no_convergence(params):
    options = {"convergence_max_discoveries": np.inf}

    minimize(
        criterion=sos_dict_criterion,
        params=params,
        algorithm="scipy_lbfgsb",
        multistart=True,
        multistart_options=options,
        logging="logging.db",
    )

    database = load_database(path="logging.db")
    iterations, _ = read_new_rows(
        database=database,
        table_name="optimization_iterations",
        last_retrieved=0,
        return_type="dict_of_lists",
    )

    present_steps = set(iterations["step"])

    assert present_steps == {1, 2, 3, 4, 5}


def test_with_non_transforming_constraints(params):
    res = minimize(
        criterion=sos_dict_criterion,
        params=params,
        constraints=[{"loc": [0, 1], "type": "fixed", "value": [0, 1]}],
        algorithm="scipy_lbfgsb",
        multistart=True,
    )

    aaae(res.params["value"].to_numpy(), np.array([0, 1, 0, 0]))


def test_error_is_raised_with_transforming_constraints(params):
    with pytest.raises(NotImplementedError):
        minimize(
            criterion=sos_dict_criterion,
            params=params,
            constraints=[{"loc": [0, 1], "type": "probability"}],
            algorithm="scipy_lbfgsb",
            multistart=True,
        )


def _params_list_to_aray(params_list):
    data = [params["value"].tolist() for params in params_list]
    return np.array(data)


def test_multistart_with_numpy_params():
    res = minimize(
        criterion=lambda params: params @ params,
        params=np.arange(5),
        algorithm="scipy_lbfgsb",
        soft_lower_bounds=np.full(5, -10),
        soft_upper_bounds=np.full(5, 10),
        multistart=True,
    )

    res.params


def test_with_invalid_bounds():

    with pytest.raises(ValueError):
        minimize(
            criterion=lambda x: x @ x,
            params=np.arange(5),
            algorithm="scipy_neldermead",
            multistart=True,
        )


def test_with_scaling():
    def _crit(params):
        x = params - np.arange(len(params))
        return x @ x

    res = minimize(
        criterion=_crit,
        params=np.full(5, 10),
        soft_lower_bounds=np.full(5, -1),
        soft_upper_bounds=np.full(5, 11),
        algorithm="scipy_lbfgsb",
        multistart=True,
    )

    aaae(res.params, np.arange(5))


def test_with_ackley():
    def ackley(x):
        out = (
            -20 * np.exp(-0.2 * np.sqrt(np.mean(x**2)))
            - np.exp(np.mean(np.cos(2 * np.pi * x)))
            + 20
            + np.exp(1)
        )
        return out

    dim = 5

    kwargs = {
        "criterion": ackley,
        "params": np.full(dim, -10),
        "lower_bounds": np.full(dim, -32),
        "upper_bounds": np.full(dim, 32),
        "algo_options": {"stopping.max_criterion_evaluations": 1000},
    }

    minimize(
        **kwargs,
        algorithm="scipy_lbfgsb",
        multistart=True,
        multistart_options={
            "n_samples": 200,
            "share_optimizations": 0.1,
            "convergence_max_discoveries": 10,
        },
    )


def test_multistart_with_least_squares_optimizers():
    est = minimize(
        criterion=sos_dict_criterion,
        params=np.array([-1, 1.0]),
        lower_bounds=np.full(2, -10.0),
        upper_bounds=np.full(2, 10.0),
        algorithm="scipy_ls_trf",
        multistart=True,
        multistart_options={"n_samples": 3, "share_optimizations": 1.0},
    )

    aaae(est.params, np.zeros(2))
