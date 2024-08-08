from dataclasses import dataclass
from enum import Enum


from optimagic.typing import PyTree


@dataclass(frozen=True)
class CriterionEvaluationResult:
    params: PyTree
    timestamp: float
    value: float
    valid: bool
    criterion_eval: PyTree | None = None
    internal_derivative: PyTree | None = None
    step: int | None = None
    exceptions: str | None = None
    hash: str | None = None


@dataclass(frozen=True)
class CriterionEvaluationWithId(CriterionEvaluationResult):
    rowid: int | None = None

    def __post_init__(self) -> None:
        if self.rowid is None:
            raise ValueError()


class StepStatus(str, Enum):
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETE = "complete"
    SKIPPED = "skipped"


class StepType(str, Enum):
    OPTIMIZATION = "optimization"
    EXPLORATION = "exploration"


@dataclass
class StepResult:
    name: str
    type: StepType
    status: StepStatus
    n_iterations: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.type, str):
            self.type = StepType(self.type)
        if isinstance(self.status, str):
            self.status = StepStatus(self.status)


@dataclass
class StepResultWithId(StepResult):
    rowid: int | None = None

    def __post_init__(self) -> None:
        if self.rowid is None:
            raise ValueError()
        super().__post_init__()


class ExistenceStrategy(str, Enum):
    RAISE = "raise"
    EXTEND = "extend"
    REPLACE = "replace"


@dataclass
class ProblemInitialization:
    direction: str
    params: PyTree


@dataclass
class ProblemInitializationWithId(ProblemInitialization):
    rowid: int | None = None

    def __post_init__(self) -> None:
        if self.rowid is None:
            raise ValueError()
