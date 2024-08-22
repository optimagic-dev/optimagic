import pytest
from optimagic.logging.types import (
    IterationStateWithId,
    ProblemInitializationWithId,
    StepResultWithId,
)


def test_raise_on_missing_id():
    with pytest.raises(ValueError, match="rowid"):
        IterationStateWithId(1, 2, 3, True, None, None, None)

    with pytest.raises(ValueError, match="rowid"):
        StepResultWithId("n", "optimization", "skipped")

    with pytest.raises(ValueError, match="rowid"):
        ProblemInitializationWithId("minimize", 2)
