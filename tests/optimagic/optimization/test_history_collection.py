import sys
from dataclasses import dataclass

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae
from numpy.testing import assert_array_equal as aae
from optimagic import mark
from optimagic.algorithms import AVAILABLE_ALGORITHMS
from optimagic.decorators import mark_minimizer
from optimagic.logging.read_log import OptimizeLogReader
from optimagic.optimization.algorithm import Algorithm, InternalOptimizeResult
from optimagic.optimization.optimize import minimize
from optimagic.parameters.bounds import Bounds
from optimagic.typing import AggregationLevel

OPTIMIZERS = []
BOUNDED = []
for name, algo in AVAILABLE_ALGORITHMS.items():
    info = algo.__algo_info__
    if not info.disable_history:
        if info.supports_parallelism:
            OPTIMIZERS.append(name)
        if info.supports_bounds:
            BOUNDED.append(name)


@pytest.mark.xfail(reason="Iteration logging is not yet implemented.")
@pytest.mark.skipif(sys.platform == "win32", reason="Slow on windows.")
@pytest.mark.parametrize("algorithm", OPTIMIZERS)
def test_history_collection_with_parallelization(algorithm, tmp_path):
    lb = np.zeros(5) if algorithm in BOUNDED else None
    ub = np.full(5, 10) if algorithm in BOUNDED else None

    logging = tmp_path / "log.db"

    collected_hist = minimize(
        fun=mark.least_squares(lambda x: x),
        params=np.arange(5),
        algorithm=algorithm,
        bounds=Bounds(lower=lb, upper=ub),
        algo_options={"n_cores": 2, "stopping_maxiter": 3},
        logging=logging,
        log_options={"if_database_exists": "replace", "fast_logging": True},
    ).history

    reader = OptimizeLogReader(logging)

    log_hist = reader.read_history()

    # We cannot expect the order to be the same
    aaae(sorted(collected_hist["criterion"]), sorted(log_hist["criterion"]))


@mark_minimizer(name="dummy")
def _dummy_optimizer(criterion, x, n_cores, batch_size, batch_evaluator):
    assert batch_size in [1, 2, 4]

    xs = np.arange(15).repeat(len(x)).reshape(15, len(x))

    for iteration in range(3):
        start_index = iteration * 5
        # do four evaluations in a batch evaluator
        batch_evaluator(
            func=criterion,
            arguments=list(xs[start_index : start_index + 4]),
            n_cores=n_cores,
        )

        # do one evaluation without the batch evaluator
        criterion(xs[start_index + 4])

    out = {
        "solution_x": xs[-1],
        "solution_criterion": 5,
        "n_fun_evals": 15,
        "n_iterations": 3,
        "success": True,
    }

    return out


@mark.minimizer(
    name="dummy",
    solver_type=AggregationLevel.SCALAR,
    is_available=True,
    is_global=False,
    needs_jac=False,
    needs_hess=False,
    supports_parallelism=True,
    supports_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class DummyOptimizer(Algorithm):
    n_cores: int = 1
    batch_size: int = 1

    def _solve_internal_problem(self, problem, x0):
        assert self.batch_size in [1, 2, 4]

        xs = np.arange(15).repeat(len(x0)).reshape(15, len(x0))

        start_index = 0

        for iteration in range(3):
            start_index = iteration * 5
            # do four evaluations in a batch evaluator
            problem.batch_fun(
                list(xs[start_index : start_index + 4]),
                n_cores=self.n_cores,
                batch_size=self.batch_size,
            )

            # do one evaluation without the batch evaluator
            problem.fun(xs[start_index + 4])

        out = InternalOptimizeResult(
            x=xs[-1],
            fun=5,
            success=True,
            n_fun_evals=15,
            n_iterations=3,
        )
        return out


def _get_fake_history(batch_size):
    if batch_size == 1:
        batches = list(range(15))
    elif batch_size == 2:
        batches = [0, 0, 1, 1, 2, 3, 3, 4, 4, 5, 6, 6, 7, 7, 8]
    elif batch_size == 4:
        batches = [0, 0, 0, 0, 1, 2, 2, 2, 2, 3, 4, 4, 4, 4, 5]
    else:
        raise ValueError("batch_size must be 1, 2 or 4.")

    out = {
        "params": list(np.arange(15).repeat(5).reshape(15, 5)),
        "criterion": [5] * 15,
        "batches": batches,
    }

    return out


def _fake_criterion(x):
    return 5


CASES = [(1, 1), (1, 2), (2, 2), (1, 4), (2, 4)]


@pytest.mark.skipif(sys.platform == "win32", reason="Slow on windows.")
@pytest.mark.parametrize("n_cores, batch_size", CASES)
def test_history_collection_with_dummy_optimizer(n_cores, batch_size):
    options = {
        "batch_size": batch_size,
        "n_cores": n_cores,
    }

    res = minimize(
        fun=_fake_criterion,
        params=np.arange(5),
        algorithm=DummyOptimizer,
        algo_options=options,
    )

    got_history = res.history

    expected_history = _get_fake_history(batch_size)

    aae(got_history.batches, expected_history["batches"])
    assert got_history.fun == expected_history["criterion"][: len(got_history.fun)]
    aaae(got_history.params, expected_history["params"][: len(got_history.params)])
