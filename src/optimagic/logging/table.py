from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any

from numpy.typing import NDArray
import numpy as np

from optimagic.logging.create_tables import (
    make_optimization_iteration_table,
    make_steps_table,
)
from optimagic.logging.load_database import DataBase
from optimagic.logging.read_from_database import (
    read_last_rows,
    read_specific_row,
    read_table,
    read_new_rows,
)
from optimagic.logging.write_to_database import update_row, append_row


@dataclass(frozen=True)
class CriterionEvaluationResult:
    params: NDArray[np.float32]
    timestamp: float
    value: float
    criterion_eval: NDArray[np.float32] | None = None
    internal_derivative: NDArray[np.float32] | None = None
    step: int | None = None
    exceptions: str | None = None
    hash: str | None = None

    @property
    def is_valid(self) -> bool:
        return self.exceptions is None

    def as_dict(self) -> dict[str, Any]:
        result_dict = asdict(self)
        result_dict["valid"] = self.is_valid
        return result_dict


@dataclass(frozen=True)
class CriterionEvaluationWithId(CriterionEvaluationResult):
    row_id: int | None = None

    def __post_init__(self) -> None:
        if self.row_id is None:
            raise ValueError()


class IterationTable:
    table_name: str = "optimization_iterations"

    def __init__(self, database: DataBase, if_exist: str = "extend"):
        make_optimization_iteration_table(database, if_exists=if_exist)
        self.database = database

    def append(self, result: CriterionEvaluationResult) -> None:
        append_row(result.as_dict(), self.table_name, self.database)

    def update(
        self, key: int, result: CriterionEvaluationResult | dict[str, Any]
    ) -> None:
        if isinstance(result, CriterionEvaluationResult):
            update_info = result.as_dict()
        else:
            update_info = result
        update_row(update_info, key, self.table_name, self.database)

    def select_last_rows(
        self,
        n_rows: int,
        stride: int = 1,
        step: int | None = None,
    ) -> list[CriterionEvaluationWithId]:
        rows = read_last_rows(
            self.database,
            self.table_name,
            n_rows,
            return_type="list_of_dicts",
            stride=stride,
            step=step,
        )

        return [CriterionEvaluationWithId(**row) for row in rows]

    def select_new_rows(
        self,
        last_retrieved: int,
        limit: int | None = None,
        stride: int = 1,
        step: int | None = None,
    ) -> list[CriterionEvaluationWithId]:
        res, _ = read_new_rows(
            self.database,
            self.table_name,
            last_retrieved,
            "list_of_dicts",
            limit=limit,
            stride=stride,
            step=step,
        )
        return [CriterionEvaluationWithId(**row) for row in res]

    def select(self, key: int | None = None) -> list[CriterionEvaluationResult]:
        if key is not None:
            rows = read_specific_row(
                self.database, self.table_name, key, "list_of_dicts"
            )
        else:
            rows = read_table(self.database, self.table_name, "list_of_dicts")

        return [CriterionEvaluationWithId(**row) for row in rows]


class StepStatus(str, Enum):
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETE = "complete"
    SKIPPED = "skipped"


class StepType(str, Enum):
    OPTIMIZATION = "optimization"


@dataclass(frozen=True)
class StepResult:
    name: str
    type: StepType
    status: StepStatus
    n_iterations: int | None = None


@dataclass(frozen=True)
class StepResultWithId(StepResult):
    row_id: int | None = None

    def __post_init__(self) -> None:
        if self.row_id is None:
            raise ValueError()


class StepTable:
    table_name = "steps"

    def __init__(self, database: DataBase, if_exists: str = "extend"):
        make_steps_table(database, if_exists=if_exists)
        self.database = database

    def append(self, result: StepResult) -> None:
        append_row(asdict(result), self.table_name, self.database)

    def update(self, key: int, result: StepResult | dict[str, Any]) -> None:
        if isinstance(result, StepResult):
            update_info = asdict(result)
        else:
            update_info = result
        update_row(update_info, key, self.table_name, self.database)

    def select_new_rows(
        self, last_retrieved: int, limit: int | None = None, stride: int = 1
    ) -> list[StepResultWithId]:
        rows, _ = read_new_rows(
            self.database,
            self.table_name,
            last_retrieved,
            "list_of_dicts",
            limit=limit,
            stride=stride,
        )
        return [StepResultWithId(**row) for row in rows]

    def select_last_rows(
        self,
        n_rows: int,
        stride: int = 1,
    ) -> list[StepResultWithId]:
        rows = read_last_rows(
            self.database,
            self.table_name,
            n_rows,
            return_type="list_of_dicts",
            stride=stride,
        )
        return [StepResultWithId(**row) for row in rows]

    def select(self, key: int | None = None) -> list[StepResultWithId]:
        if key is not None:
            rows = read_specific_row(
                self.database, self.table_name, key, "list_of_dicts"
            )
        else:
            rows = read_table(self.database, self.table_name, "list_of_dicts")

        return [StepResultWithId(**row) for row in rows]
