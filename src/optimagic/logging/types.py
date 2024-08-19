from dataclasses import dataclass
from enum import Enum
from typing import Literal

from optimagic.typing import (
    DictLikeAccess,
    Direction,
    DirectionLiteral,
    PyTree,
)


class StepStatus(str, Enum):
    """Status of a step in a process.

    Attributes:
        SCHEDULED: Indicates that the step is scheduled but not yet started.
        RUNNING: Indicates that the step is currently in progress.
        COMPLETE: Indicates that the step has completed successfully.
        SKIPPED: Indicates that the step was skipped.

    """

    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETE = "complete"
    SKIPPED = "skipped"


StepStatusLiteral = Literal["scheduled", "running", "complete", "skipped"]


class StepType(str, Enum):
    """Type of step in a process.

    Attributes:
        OPTIMIZATION: Represents an optimization step.
        EXPLORATION: Represents an exploration step.

    """

    OPTIMIZATION = "optimization"
    EXPLORATION = "exploration"


StepTypeLiteral = Literal["optimization", "exploration"]


class ExistenceStrategy(str, Enum):
    """Strategies to handle the existence of a database or table.

    Attributes:
        RAISE: Raise an error if the database resp. table exists.
        EXTEND: Extend the existing database or table.
        REPLACE: Replace the existing database or table.

    """

    RAISE = "raise"
    EXTEND = "extend"
    REPLACE = "replace"


ExistenceStrategyLiteral = Literal["raise", "extend", "replace"]


@dataclass(frozen=True)
class IterationState(DictLikeAccess):
    """Result of a criterion evaluation.

    Attributes:
        params: The parameters used in the evaluation.
        timestamp: The time at which the evaluation was performed.
        value: The result value of the evaluation.
        valid: Indicates if the evaluation is valid.
        criterion_eval: Optional, additional evaluation information.
        internal_derivative: Optional, derivative information used internally.
        step: Optional, step number associated with the evaluation.
        exceptions: Optional, exceptions encountered during evaluation.
        hash: Optional, hash of the evaluation for identification purposes.

    """

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
class CriterionEvaluationWithId(IterationState):
    """Criterion evaluation result with an ID.

    Attributes:
        rowid: The unique ID associated with the evaluation result.

    Raises:
        ValueError: If `rowid` is None.

    """

    rowid: int | None = None

    def __post_init__(self) -> None:
        if self.rowid is None:
            raise ValueError("rowid must not be None")


@dataclass(frozen=True)
class StepResult(DictLikeAccess):
    """Result of a process step.

    Attributes:
        name: The name of the step.
        type: The type of the step, either as `StepType` or string.
        status: The status of the step, either as `StepStatus` or string.
        n_iterations: Optional, the number of iterations performed in the step.

    """

    name: str
    type: StepType | StepTypeLiteral
    status: StepStatus | StepStatusLiteral
    n_iterations: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.type, str):
            object.__setattr__(self, "type", StepType(self.type))
        if isinstance(self.status, str):
            object.__setattr__(self, "status", StepStatus(self.status))


@dataclass(frozen=True)
class StepResultWithId(StepResult):
    """Step result with an ID.

    Attributes:
        rowid: The unique ID associated with the step result.

    Raises:
        ValueError: If `rowid` is None.

    """

    rowid: int | None = None

    def __post_init__(self) -> None:
        if self.rowid is None:
            raise ValueError("rowid must not be None")
        super().__post_init__()


@dataclass(frozen=True)
class ProblemInitialization(DictLikeAccess):
    """Start characteristics of an optimization problem.

    Attributes:
        direction: The direction of optimization,
            either as `Direction` or string literal.
        params: The parameters for the initialization.

    """

    direction: Direction | DirectionLiteral
    params: PyTree


@dataclass(frozen=True)
class ProblemInitializationWithId(ProblemInitialization):
    """Problem initialization with an ID.

    Attributes:
        rowid: The unique ID associated with the problem initialization.

    Raises:
        ValueError: If `rowid` is None.

    """

    rowid: int | None = None

    def __post_init__(self) -> None:
        if self.rowid is None:
            raise ValueError("rowid must not be None")
