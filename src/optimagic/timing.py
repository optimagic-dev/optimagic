from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class CostModel:
    fun: float | None
    jac: float | None
    fun_and_jac: float | None
    label: str
    aggregate_batch_time: Callable[[list[float]], float]


evaluation_time = CostModel(
    fun=None,
    jac=None,
    fun_and_jac=None,
    label="Function time (seconds)",
    aggregate_batch_time=sum,
)

fun_evaluations = CostModel(
    fun=1,
    jac=0,
    fun_and_jac=1,
    label="Number of criterion evaluations",
    aggregate_batch_time=sum,
)

fun_batches = CostModel(
    fun=1, jac=0, fun_and_jac=1, label="Number of batches", aggregate_batch_time=max
)

wall_time = "wall_time"


TIMING_REGISTRY = {
    "evaluation_time": evaluation_time,
    "fun_evaluations": fun_evaluations,
    "fun_batches": fun_batches,
    "wall_time": wall_time,
}
