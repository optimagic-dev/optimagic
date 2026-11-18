"""Fast tests for the optimizers from DFO-LS and Py-BOBYQA.

The detailed tests of all options are in ``test_nag_wrappers.py`` and only run when
selected with ``-m nag_wrappers``.

"""

from typing import Any

import numpy as np
import pytest

import optimagic as om
from optimagic import mark
from optimagic.config import IS_DFOLS_INSTALLED, IS_PYBOBYQA_INSTALLED
from optimagic.exceptions import InvalidAlgoOptionError
from optimagic.optimization.optimize import minimize
from optimagic.optimizers.nag_optimizers import (
    DfolsGrowingOptions,
    DfolsRegressionOptions,
    DfolsRestartOptions,
    NagDFOLS,
    NagInitOptions,
    NagNoiseOptions,
    NagPyBOBYQA,
    NagSlowProgressOptions,
    NagTrustregionOptions,
    PybobyqaRestartOptions,
)
from optimagic.parameters.bounds import Bounds
from tests.estimagic.test_bootstrap import aaae

pytestmark = pytest.mark.nag

# Typed as Any because the tests use deprecated options, which is valid at runtime
# but not for type checkers.
DFOLS: Any = NagDFOLS
PYBOBYQA: Any = NagPyBOBYQA

needs_dfols = pytest.mark.skipif(not IS_DFOLS_INSTALLED, reason="DFO-LS not installed.")
needs_pybobyqa = pytest.mark.skipif(
    not IS_PYBOBYQA_INSTALLED, reason="Py-BOBYQA not installed."
)


@mark.least_squares
def sos(x):
    return x


def test_nag_dfols_starting_at_optimum():
    # From issue: https://github.com/optimagic-dev/optimagic/issues/538
    params = np.zeros(2, dtype=float)
    res = minimize(
        fun=sos,
        params=params,
        algorithm="nag_dfols",
        bounds=Bounds(-1 * np.ones_like(params), np.ones_like(params)),
    )
    aaae(res.params, params)


@needs_dfols
def test_nag_dfols_with_option_groups_and_evaluation_count():
    algo = DFOLS(
        n_evals_per_point=2,
        restart_options=DfolsRestartOptions(
            use_restarts=True, max_unsuccessful_restarts=1
        ),
        growing_options=DfolsGrowingOptions(n_initial_directions=1),
    )
    res: Any = minimize(fun=sos, params=np.arange(3.0) + 1, algorithm=algo)

    aaae(res.params, np.zeros(3))
    assert res.n_iterations > 0
    assert res.n_fun_evals == len(res.history.fun)
    assert res.n_fun_evals == 2 * res.algorithm_output["n_evaluated_points"]


@needs_pybobyqa
def test_nag_pybobyqa_with_option_groups_and_evaluation_count():
    algo = PYBOBYQA(
        n_evals_per_point=lambda radius, min_radius, n_iterations, n_restarts: 2,
        convergence_target_value=1e-10,
        restart_options=PybobyqaRestartOptions(
            use_restarts=True, max_unsuccessful_restarts=1
        ),
    )
    res: Any = minimize(fun=lambda x: x @ x, params=np.arange(3.0) + 1, algorithm=algo)

    aaae(res.params, np.zeros(3), decimal=4)
    assert res.fun <= 1e-10
    assert res.n_iterations > 0
    assert res.n_fun_evals == len(res.history.fun)
    assert res.n_fun_evals == 2 * res.algorithm_output["n_evaluated_points"]


@needs_dfols
def test_nag_dfols_deprecated_options():
    old = {
        "clip_criterion_if_overflowing": False,
        "convergence_noise_corrected_criterion_tolerance": 0.0,
        "initial_directions": "random",
        "noise_additive_level": 0.1,
        "trustregion_method_to_replace_extra_points": "momentum",
        "trustregion_threshold_successful": 0.2,
        "convergence_slow_progress": {"comparison_period": 6},
        "trustregion_reset_options": {
            "reset_type": "hard",
            "max_interpolation_points": 8,
            "n_extra_interpolation_points_per_soft_reset": 2,
            "n_extra_interpolation_points_per_hard_reset": 3,
        },
        "trustregion_fast_start_options": {"min_inital_points": 2, "method": "auto"},
    }
    new = {
        "check_fun_for_overflow": False,
        "noisy": True,
        "init_options": NagInitOptions(random_initial_directions=True),
        "noise_options": NagNoiseOptions(
            additive_noise_level=0.1, quit_on_noise_level=False
        ),
        "regression_options": DfolsRegressionOptions(momentum_extra_steps=True),
        "trustregion_options": NagTrustregionOptions(shrinking_threshold=0.2),
        "slow_progress_options": NagSlowProgressOptions(history_for_slow=6),
        "restart_options": DfolsRestartOptions(
            use_soft_restarts=False,
            max_n_interpolation_points=8,
            increase_n_interpolation_points=True,
            increase_n_interpolation_points_amount=2,
            hard_increase_n_initial_directions_amount=1,
        ),
        "growing_options": DfolsGrowingOptions(n_initial_directions=2),
    }
    with pytest.warns(FutureWarning, match="removed in optimagic 0.7.0"):
        algo = DFOLS(**old)

    assert algo == DFOLS(**new)


@needs_pybobyqa
def test_nag_pybobyqa_deprecated_options():
    old = {
        "stopping_max_criterion_evaluations": 50,
        "clip_criterion_if_overflowing": False,
        "noise_multiplicative_level": 0.1,
        "trustregion_reset_options": {"use_resets": True, "reset_type": "soft"},
    }
    new = {
        "stopping_maxfun": 50,
        "noisy": True,
        "noise_options": NagNoiseOptions(multiplicative_noise_level=0.1),
        "restart_options": PybobyqaRestartOptions(
            use_restarts=True, use_soft_restarts=True
        ),
    }
    with pytest.warns(FutureWarning, match="removed in optimagic 0.7.0"):
        algo = PYBOBYQA(**old)

    assert algo == PYBOBYQA(**new)


@pytest.mark.parametrize(
    "old",
    [
        {"initial_directions": "diagonal"},
        {"trustregion_method_to_replace_extra_points": "other"},
        {"convergence_slow_progress": {"not_an_entry": 1}},
        {"trustregion_reset_options": {"reset_type": "medium"}},
        {"trustregion_fast_start_options": {"method": "other"}},
        {
            "trustregion_reset_options": {
                "n_extra_interpolation_points_per_soft_reset": 2,
                "n_extra_interpolation_points_per_hard_reset": 1,
            }
        },
    ],
)
def test_nag_dfols_invalid_deprecated_options(old):
    with pytest.raises(InvalidAlgoOptionError):
        DFOLS(**old)


def test_nag_dfols_deprecated_dict_options_can_be_none():
    with pytest.warns(FutureWarning, match="has no effect anymore"):
        algo = DFOLS(convergence_slow_progress=None, trustregion_reset_options=None)
    assert algo == DFOLS()


def test_nag_option_groups_are_rejected_as_dictionaries():
    with pytest.raises(InvalidAlgoOptionError, match="instance of DfolsRestartOptions"):
        DFOLS(restart_options={"use_restarts": True})
    with pytest.raises(
        InvalidAlgoOptionError, match="instance of PybobyqaRestartOptions"
    ):
        PYBOBYQA().with_option(restart_options={"use_restarts": True})


@needs_dfols
def test_nag_dfols_raises_if_solver_rejects_options():
    with pytest.raises(InvalidAlgoOptionError, match="npt must be >= n"):
        minimize(
            fun=sos,
            params=np.arange(3.0),
            algorithm=om.algos.nag_dfols(n_interpolation_points=1),
        )
