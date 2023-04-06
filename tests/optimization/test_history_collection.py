import sys

import numpy as np
import pytest
from tranquilo.logging.read_log import OptimizeLogReader
from tranquilo.optimization import AVAILABLE_ALGORITHMS
from tranquilo.optimization.optimize import minimize
from numpy.testing import assert_array_almost_equal as aaae

OPTIMIZERS = []
BOUNDED = []
for name, algo in AVAILABLE_ALGORITHMS.items():
    args = algo._algorithm_info.arguments
    if not algo._algorithm_info.disable_history:
        if "n_cores" in args:
            OPTIMIZERS.append(name)
        if "lower_bounds" in args:
            BOUNDED.append(name)


@pytest.mark.skipif(sys.platform != "linux", reason="Slow on other platforms.")
@pytest.mark.parametrize("algorithm", OPTIMIZERS)
def test_history_collection_with_parallelization(algorithm, tmp_path):
    lb = np.zeros(5) if algorithm in BOUNDED else None
    ub = np.full(5, 10) if algorithm in BOUNDED else None

    logging = tmp_path / "log.db"

    collected_hist = minimize(
        criterion=lambda x: {"root_contributions": x, "value": x @ x},
        params=np.arange(5),
        algorithm=algorithm,
        lower_bounds=lb,
        upper_bounds=ub,
        algo_options={"n_cores": 2, "stopping.max_iterations": 3},
        logging=logging,
        log_options={"if_database_exists": "replace", "fast_logging": True},
    ).history

    reader = OptimizeLogReader(logging)

    log_hist = reader.read_history()

    # We cannot expect the order to be the same
    aaae(sorted(collected_hist["criterion"]), sorted(log_hist["criterion"]))
