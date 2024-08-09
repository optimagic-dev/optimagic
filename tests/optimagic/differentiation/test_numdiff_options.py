import pytest
from optimagic.differentiation.numdiff_options import (
    NumDiffOptions,
    pre_process_numdiff_options,
)
from optimagic.exceptions import InvalidNumdiffError


def test_pre_process_numdiff_options_trivial_case():
    numdiff_options = NumDiffOptions(
        method="central",
        step_size=2,
        scaling_factor=0.5,
        min_steps=None,
        batch_evaluator="joblib",
    )
    got = pre_process_numdiff_options(numdiff_options)
    assert got == numdiff_options


def test_pre_process_numdiff_options_none_case():
    assert pre_process_numdiff_options(None) is None


def test_pre_process_numdiff_options_dict_case():
    got = pre_process_numdiff_options(
        {"method": "central", "step_size": 2, "batch_evaluator": "pathos"}
    )
    assert got == NumDiffOptions(
        method="central", step_size=2, batch_evaluator="pathos"
    )


def test_pre_process_numdiff_options_invalid_type():
    with pytest.raises(InvalidNumdiffError):
        pre_process_numdiff_options(numdiff_options="invalid")


def test_pre_process_numdiff_options_invalid_dict_key():
    with pytest.raises(InvalidNumdiffError, match="Invalid numdiff options"):
        pre_process_numdiff_options(numdiff_options={"wrong_key": "central"})


def test_pre_process_numdiff_options_invalid_dict_value():
    with pytest.raises(InvalidNumdiffError, match="Invalid numdiff `method`:"):
        pre_process_numdiff_options(numdiff_options={"method": "invalid"})


def test_numdiff_options_invalid_method():
    with pytest.raises(InvalidNumdiffError, match="Invalid numdiff `method`:"):
        NumDiffOptions(method="invalid")


def test_numdiff_options_invalid_step_size():
    with pytest.raises(InvalidNumdiffError, match="Invalid numdiff `step_size`:"):
        NumDiffOptions(step_size=0)


def test_numdiff_options_invalid_scaling_factor():
    with pytest.raises(InvalidNumdiffError, match="Invalid numdiff `scaling_factor`:"):
        NumDiffOptions(scaling_factor=-1)


def test_numdiff_options_invalid_min_steps():
    with pytest.raises(InvalidNumdiffError, match="Invalid numdiff `min_steps`:"):
        NumDiffOptions(min_steps=-1)


def test_numdiff_options_invalid_n_cores():
    with pytest.raises(InvalidNumdiffError, match="Invalid numdiff `n_cores`:"):
        NumDiffOptions(n_cores=-1)


def test_numdiff_options_invalid_batch_evaluator():
    with pytest.raises(InvalidNumdiffError, match="Invalid numdiff `batch_evaluator`:"):
        NumDiffOptions(batch_evaluator="invalid")
