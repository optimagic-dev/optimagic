"""Test helper functions for PyGAD optimizer."""

import warnings

import pytest

from optimagic.optimizers.pygad_optimizer import determine_effective_batch_size


@pytest.mark.parametrize(
    "batch_size, n_cores, expected",
    [
        (None, 1, None),
        (None, 4, 4),
        (10, 4, 10),
        (4, 4, 4),
        (2, 4, 2),
        (5, 1, 5),
        (0, 4, 0),
        (None, 100, 100),
        (1, 1, 1),
    ],
)
def test_determine_effective_batch_size_return_values(batch_size, n_cores, expected):
    result = determine_effective_batch_size(batch_size, n_cores)
    assert result == expected


@pytest.mark.parametrize(
    "batch_size, n_cores, should_warn",
    [
        (2, 4, True),
        (1, 8, True),
        (0, 4, True),
        (4, 4, False),
        (8, 4, False),
        (None, 4, False),
        (5, 1, False),
        (None, 1, False),
    ],
)
def test_determine_effective_batch_size_warnings(batch_size, n_cores, should_warn):
    if should_warn:
        warning_pattern = (
            f"batch_size \\({batch_size}\\) is smaller than "
            f"n_cores \\({n_cores}\\)\\. This may reduce parallel efficiency\\. "
            f"Consider setting batch_size >= n_cores\\."
        )
        with pytest.warns(UserWarning, match=warning_pattern):
            result = determine_effective_batch_size(batch_size, n_cores)
            assert result == batch_size
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = determine_effective_batch_size(batch_size, n_cores)
