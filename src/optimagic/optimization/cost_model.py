from dataclasses import dataclass
from enum import Enum


# Bad name and does not need to be an enum; just representative right now.
class TimeType(Enum):
    FUNC_TIME = "FUNC_TIME"
    WALL_TIME = "WALL_TIME"


@dataclass(frozen=True)
class CostModel:
    fun: float | TimeType | str | None = None
    jac: float | TimeType | str | None = None
    fun_and_jac: float | TimeType | str | None = None
    label: str = ""

    def aggregate_batch_time(times: list[float]) -> float:
        pass


class PerfectParallelizationCostModel(CostModel):
    def aggregate_batch_time(times: list[float]) -> float:
        return max(times)


# ======================================================================================
# Cost models for serial case
# ======================================================================================

FUNCTION_TIME = CostModel(
    fun=TimeType.FUNC_TIME,
    jac=TimeType.FUNC_TIME,
    fun_and_jac=TimeType.FUNC_TIME,
    label="Function time (seconds)",
)

WALL_TIME = CostModel(
    fun=TimeType.WALL_TIME,
    jac=TimeType.WALL_TIME,
    fun_and_jac=TimeType.WALL_TIME,
    label="Wall time (seconds)",
)

N_EVALUATIONS = CostModel(
    fun=1, jac=0, fun_and_jac=1, label="Number of criterion evaluations"
)

# ======================================================================================
# Cost models for parallel case
# ======================================================================================

N_BATCHES = PerfectParallelizationCostModel(
    fun=1, jac=0, fun_and_jac=1, label="Number of batches"
)
