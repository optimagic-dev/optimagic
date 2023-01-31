import itertools

import numpy as np
import pytest
from estimagic.optimization.tranquilo.tranquilo_history import History
from estimagic.optimization.tranquilo.wrap_criterion import get_wrapped_criterion
from numpy.testing import assert_array_almost_equal as aaae

TEST_CASES = list(itertools.product(["scalar", "least_squares", "likelihood"], [1, 2]))


@pytest.mark.parametrize("functype, n_evals", TEST_CASES)
def test_wrapped_criterion(functype, n_evals):
    # set up criterion (all should have same results)
    func_dict = {
        "least_squares": lambda x: x,
        "likelihood": lambda x: x**2,
        "scalar": lambda x: x @ x,
    }

    criterion = func_dict[functype]

    # set up history
    history = History(functype=functype)
    for params in [np.zeros(3), np.ones(3)]:
        history.add_entries(params, criterion(params))

    assert history.get_n_fun() == 2

    # set up wrapped params
    wrapped_criterion = get_wrapped_criterion(
        criterion=criterion, batch_evaluator="joblib", n_cores=1, history=history
    )

    # set up params and expected results
    if n_evals == 1:
        params = np.arange(3)
        expected_fvecs = criterion(params)
        expected_fvals = params @ params
        expected_indices = 2
    else:
        params = np.arange(3 * n_evals).reshape(n_evals, 3)
        expected_fvecs = np.array([criterion(x) for x in params]).reshape(2, -1)
        expected_fvals = np.array([x @ x for x in params])
        expected_indices = np.arange(2, 2 + n_evals)

    # use wrapped_criterion
    got_fvecs, got_fvals, got_indices = wrapped_criterion(params)

    aaae(got_fvecs, expected_fvecs)
    aaae(got_fvals, expected_fvals)
    aaae(got_indices, expected_indices)
