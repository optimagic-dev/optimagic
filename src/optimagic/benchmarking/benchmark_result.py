"""Representation of the result of running one optimizer on one benchmark problem."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from optimagic.optimization.optimize_result import OptimizeResult


@dataclass(frozen=True, eq=False)
class BenchmarkResult:
    """The result of running one optimizer on one benchmark problem.

    If the optimization raised an error, the histories only contain the start
    parameters, ``optimize_result`` is None and ``traceback`` contains the error.

    """

    params_history: NDArray[np.float64]
    """The parameters at which the function was evaluated, in the order of the
    evaluations. Shape (n_evaluations, n_params)."""

    fun_history: NDArray[np.float64]
    """The function values at ``params_history`` without noise. For least-squares
    problems these are sums of squared residuals, for likelihood problems sums of
    contributions. Shape (n_evaluations,)."""

    time_history: NDArray[np.float64]
    """The time in seconds between the start of the optimization and the start of each
    function evaluation. Shape (n_evaluations,)."""

    batches_history: NDArray[np.int64]
    """The batch of each function evaluation. Evaluations in the same batch were done
    in parallel. Shape (n_evaluations,)."""

    optimize_result: OptimizeResult | None
    """The result returned by ``minimize`` or None if the optimization raised an
    error."""

    traceback: str | None = None
    """The traceback of the error that was raised during the optimization or None if
    there was no error."""
