from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Dict
from typing import Union

import numpy as np
import pandas as pd
from estimagic.utilities import to_pickle


@dataclass
class OptimizeResult:
    params: Any
    criterion: float
    start_criterion: float
    start_params: Any
    algorithm: str
    direction: str
    n_free: int

    message: Union[str, None] = None
    success: Union[bool, None] = None
    n_criterion_evaluations: Union[int, None] = None
    n_derivative_evaluations: Union[int, None] = None
    n_iterations: Union[int, None] = None

    history: Union[Dict, None] = None

    convergence_report: Union[Dict, None] = None

    multistart_info: Union[Dict, None] = None
    algorithm_output: Dict = field(default_factory=dict)

    def __repr__(self):
        first_line = (
            f"{self.direction.title()} with {self.n_free} free parameters terminated"
        )

        if self.success is not None:
            snippet = "successfully" if self.success else "unsuccessfully"
            first_line += f" {snippet}"

        counters = [
            ("criterion evaluations", self.n_criterion_evaluations),
            ("derivative evaluations", self.n_derivative_evaluations),
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

        if self.start_criterion is not None and self.criterion is not None:
            improvement = (
                f"The value of criterion improved from {self.start_criterion} to "
                f"{self.criterion}."
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


def _format_convergence_report(report, algorithm):
    report = pd.DataFrame.from_dict(report)
    columns = ["one_step", "five_steps"]

    table = report[columns].applymap(_format_float).astype(str)

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
    return "{0:.4g}".format(number)
