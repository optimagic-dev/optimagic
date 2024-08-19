from typing import Any

from optimagic.logging.logger import LogStore
from optimagic.logging.types import StepResult, StepStatus


def log_scheduled_steps_and_get_ids(
    steps: list[dict[str, Any]], logging: LogStore | None
) -> list[int | None]:
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
    if logging:
        for row in steps:
            data = StepResult(**{**default_row, **row})
            logging.step_store.insert(data)

        last_steps = logging.step_store.select_last_rows(len(steps))
        step_ids = [row.rowid for row in last_steps]
    else:
        step_ids = list(range(len(steps)))

    return step_ids
