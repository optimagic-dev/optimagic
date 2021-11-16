from estimagic.logging.database_utilities import append_row
from estimagic.logging.database_utilities import read_last_rows
from estimagic.logging.database_utilities import update_row


def log_scheduled_steps_and_get_ids(steps, logging, db_kwargs):
    """Add scheduled steps to the steps table of the database and get their ids.

    The ids are only determined once the steps are written to the database and the
    ids of all previously existing steps are known.

    Args:
        steps (list): List of dicts with entries for the steps table.
        logging (bool): Whether to actually write to the databes.
        db_kwargs (dict): Dict with the entries "database", "path" and "fast_logging"

    Returns:
        list: List of integers with the step ids.

    """
    default_row = {"status": "scheduled"}
    if logging:
        for row in steps:
            data = {**default_row, **row}

            append_row(
                data=data,
                table_name="steps",
                **db_kwargs,
            )

        step_ids = read_last_rows(
            table_name="steps",
            n_rows=len(steps),
            return_type="dict_of_lists",
            **db_kwargs,
        )["rowid"]
    else:
        step_ids = list(range(len(steps)))

    return step_ids


def update_step_status(step, new_status, db_kwargs):
    step = int(step)

    assert new_status in ["scheduled", "running", "complete", "skipped"]

    update_row(
        data={"status": new_status},
        rowid=step,
        table_name="steps",
        **db_kwargs,
    )
