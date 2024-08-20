from typing import Any, cast

from optimagic.logging.logger import LogStore
from optimagic.logging.types import StepResult, StepStatus


def log_scheduled_steps_and_get_ids(
    steps: list[dict[str, Any]], logger: LogStore | None
) -> list[int]:
    """Add scheduled steps to the steps table of the database and get their ids.

    The ids are only determined once the steps are written to the database and the
    ids of all previously existing steps are known.

    Args:
        steps (list): List of dicts with entries for the steps table.
        logging (bool): Whether to actually write to the database.

    Returns:
        list: List of integers with the step ids.

    """
    default_row = {"status": StepStatus.SCHEDULED.value}
    if logger:
        for row in steps:
            data = StepResult(**{**default_row, **row})
            logger.step_store.insert(data)

        last_steps = logger.step_store.select_last_rows(len(steps))
        step_ids = cast(list[int], [row.rowid for row in last_steps])
    else:
        step_ids = list(range(len(steps)))

    return step_ids
