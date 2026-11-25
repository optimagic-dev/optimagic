from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from optimagic.logging.logger import LogReader
from optimagic.optimization.history import History
from optimagic.shared.compat import pd_df_map
from optimagic.typing import PyTree
from optimagic.utilities import to_pickle


@dataclass
class OptimizeResult:
    """Optimization result object."""

    params: Any
    """The optimal parameters."""

    fun: float
    """The optimal criterion value."""

    start_fun: float
    """The criterion value at the start parameters."""

    start_params: Any
    """The start parameters."""

    algorithm: str
    """The algorithm used for the optimization."""

    direction: str
    """Maximize or minimize."""

    n_free: int
    """Number of free parameters."""

    message: str | None = None
    """Message returned by the underlying algorithm."""

    success: bool | None = None
    """Whether the optimization was successful."""

    n_fun_evals: int | None = None
    """Number of criterion evaluations."""

    n_jac_evals: int | None = None
    """Number of derivative evaluations."""

    n_hess_evals: int | None = None

    n_iterations: int | None = None
    """Number of iterations until termination."""

    status: int | None = None
    jac: PyTree | None = None
    hess: PyTree | None = None
    hess_inv: PyTree | None = None
    max_constraint_violation: float | None = None

    history: History | None = None
    """Optimization history."""

    convergence_report: dict[str, Any] | None = None
    """The convergence report."""

    multistart_info: Optional["MultistartInfo"] = None
    """Multistart information."""

    algorithm_output: dict[str, Any] | None = None
    """Additional algorithm specific information."""

    logger: LogReader | None = None

    # ==================================================================================
    # Scipy aliases
    # ==================================================================================

    @property
    def x(self) -> PyTree:
        return self.params

    @property
    def x0(self) -> PyTree:
        return self.start_params

    @property
    def nfev(self) -> int | None:
        return self.n_fun_evals

    @property
    def nit(self) -> int | None:
        return self.n_iterations

    @property
    def njev(self) -> int | None:
        return self.n_jac_evals

    @property
    def nhev(self) -> int | None:
        return self.n_hess_evals

    # Enable attribute access using dictionary-style notation for scipy compatibility
    def __getitem__(self, key):
        return getattr(self, key)

    def __repr__(self) -> str:
        first_line = (
            f"{self.direction.title()} with {self.n_free} free parameters terminated"
        )

        if self.success is not None:
            snippet = "successfully" if self.success else "unsuccessfully"
            first_line += f" {snippet}"

        counters = [
            ("criterion evaluations", self.n_fun_evals),
            ("derivative evaluations", self.n_jac_evals),
            ("iterations", self.n_iterations),
        ]

        counters = [(n, v) for n, v in counters if v is not None]

        if counters:
            name, val = counters[0]
            counter_msg = f"after {val} {name}"
            if len(counters) >= 2:
                for name, val in counters[1:-1]:
                    counter_msg += f", {val} {name}"

                name, val = counters[-1]
                counter_msg += f" and {val} {name}"
            first_line += f" {counter_msg}"

        first_line += "."

        if self.message:
            message = f"The {self.algorithm} algorithm reported: {self.message}"
        else:
            message = None

        if self.start_fun is not None and self.fun is not None:
            improvement = (
                f"The value of criterion improved from {self.start_fun} to {self.fun}."
            )
        else:
            improvement = None

        if self.convergence_report is not None:
            convergence = _format_convergence_report(
                self.convergence_report, self.algorithm
            )
        else:
            convergence = None

        sections = [first_line, improvement, message, convergence]
        sections = [sec for sec in sections if sec is not None]

        msg = "\n\n".join(sections)

        return msg

    def to_pickle(self, path):
        """Save the OptimizeResult object to pickle.

        Args:
            path (str, pathlib.Path): A str or pathlib.path ending in .pkl or .pickle.

        """
        to_pickle(self, path=path)


@dataclass(frozen=True)
class MultistartInfo:
    """Information about the multistart optimization.

    Attributes:
        start_parameters: List of start parameters for each optimization.
        local_optima: List of optimization results.
        exploration_sample: List of parameters used for exploration.
        exploration_results: List of function values corresponding to exploration.
        n_optimizations: Number of local optimizations that were run.

    """

    start_parameters: list[PyTree]
    local_optima: list[OptimizeResult]
    exploration_sample: list[PyTree]
    exploration_results: list[float]

    @property
    def n_optimizations(self) -> int:
        return len(self.local_optima)


def _format_convergence_report(report, algorithm):
    report = pd.DataFrame(report)
    columns = ["one_step", "five_steps"]

    table = pd_df_map(report[columns], _format_float).astype(str)

    for col in "one_step", "five_steps":
        table[col] = table[col] + _create_stars(report[col])

    table = table.to_string(justify="center")

    introduction = (
        f"Independent of the convergence criteria used by {algorithm}, "
        "the strength of convergence can be assessed by the following criteria:"
    )

    explanation = (
        "(***: change <= 1e-10, **: change <= 1e-8, *: change <= 1e-5. "
        "Change refers to a change between accepted steps. The first column only "
        "considers the last step. The second column considers the last five steps.)"
    )

    out = "\n\n".join([introduction, table, explanation])

    return out


def _create_stars(sr):
    stars = pd.cut(
        sr,
        bins=[-np.inf, 1e-10, 1e-8, 1e-5, np.inf],
        labels=["***", "** ", "*  ", "   "],
    ).astype("str")

    return stars


def _format_float(number):
    """Round to four significant digits."""
    return f"{number:.4g}"
