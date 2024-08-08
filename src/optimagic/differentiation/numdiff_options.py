from dataclasses import dataclass
from typing import Callable, Literal

from optimagic.config import DEFAULT_N_CORES


@dataclass(frozen=True)
class NumdiffOptions:
    method: Literal["central", "forward", "backward"] = "central"
    step_size: float | None = None
    scaling_factor: float = 1
    min_steps: float | None = None
    n_cores: int = DEFAULT_N_CORES
    batch_evaluator: Literal["joblib", "pathos"] | Callable = "joblib"  # type: ignore
