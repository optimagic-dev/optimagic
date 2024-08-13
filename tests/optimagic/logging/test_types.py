import pytest
from optimagic.logging.types import (
    CriterionEvaluationWithId,
    ProblemInitializationWithId,
    StepResultWithId,
)


def test_raise_on_missing_id():
    with pytest.raises(ValueError, match="rowid"):
        CriterionEvaluationWithId(1, 2, 3, True)

    with pytest.raises(ValueError, match="rowid"):
        StepResultWithId("n", "optimization", "skipped")

    with pytest.raises(ValueError, match="rowid"):
        ProblemInitializationWithId("minimize", 2)
