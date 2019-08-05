import numpy as np
import pytest
from numpy.testing import assert_array_equal

from estimagic.visualization.comparison_plot import _catch_up_x
from estimagic.visualization.comparison_plot import _create_x_and_dodge
from estimagic.visualization.comparison_plot import _update_dodge
from estimagic.visualization.comparison_plot import _update_x

# ============================================================================
# _BUILD_DF_FROM_DATA_DICT
# ============================================================================


# ============================================================================
# _CREATE_GROUP_AND_HEIGHTS
# ============================================================================


# ============================================================================
# _CATCH_UP_X
# ============================================================================

fix_for_catch_up_x = [
    ([1, 2, 3], [], [1, 2, 3]),
    ([1, 2, 3], [4, 5, 6], [1, 2, 3, 5, 5, 5]),
    ([], [4, 5], [4.5, 4.5]),
]


@pytest.mark.parametrize("new_xs, stored_x, expected_new", fix_for_catch_up_x)
def test_catch_up_x(new_xs, stored_x, expected_new):
    _catch_up_x(new_xs, stored_x)
    assert new_xs == expected_new, "new_xs wrong"
    assert stored_x == [], "stored_x not empty"


# ============================================================================
# _UPDATE_X
# ============================================================================


fix_for_update_x = [
    # old_x, needs_dodge, new_xs, stored_x, expected_new, expected_stored
    (0, False, [], [], [], [0]),
    (1, True, [], [], [], [1]),
    (0, False, [1, 2, 3], [4, 5, 6], [1, 2, 3, 5, 5, 5], [0]),
    (1, True, [1, 2, 3], [4, 5, 6], [1, 2, 3], [4, 5, 6, 1]),
]


@pytest.mark.parametrize(
    "old_x, needs_dodge, new_xs, stored_x, expected_new, expected_stored",
    fix_for_update_x,
)
def test_update_x(old_x, needs_dodge, new_xs, stored_x, expected_new, expected_stored):
    _update_x(old_x, needs_dodge, new_xs, stored_x)
    assert new_xs == expected_new, "new_xs wrong"
    assert stored_x == expected_stored, "stored_x wrong"


# ============================================================================
# _UPDATE_DODGE
# ============================================================================

fix_for_update_dodge = [
    # dodge, needs_dodge, expected
    ([], False, [0]),
    ([], True, [1]),
    ([0], False, [0, 0]),
    ([0], True, [1, -1]),
    ([0, 1], False, [0, 1, 0]),
    ([0, 1], True, [0, 1, -1]),
]


@pytest.mark.parametrize("dodge, needs_dodge, expected", fix_for_update_dodge)
def test_update_dodge(dodge, needs_dodge, expected):
    _update_dodge(dodge, needs_dodge)
    assert dodge == expected, "dodge wrong"


# ============================================================================
# _CREATE_X_AND_DODGE
# ============================================================================

fix_for_create_x_and_dodge = [
    # name, val_arr, bool_arr, expected_x, expected_dodge
    (
        "no_dodge",
        np.arange(5),
        np.array([False, False, False, False, False]),
        np.arange(5),
        np.zeros(5),
    ),
    (
        "begin_dodge",
        np.arange(5),
        np.array([True, True, True, False, False]),
        np.array([1, 1, 1, 3, 4]),
        np.array([1, -1, 2, 0, 0]),
    ),
    (
        "middle_dodge",
        np.arange(5),
        np.array([False, True, True, True, False]),
        np.array([1.5, 1.5, 1.5, 1.5, 4]),
        np.array([1, -1, 2, -2, 0]),
    ),
    (
        "start_finish_dodge",
        np.arange(5),
        np.array([True, True, False, True, True]),
        np.array([0.5, 0.5, 3, 3, 3]),
        np.array([1, -1, 1, -1, 2]),
    ),
]


@pytest.mark.parametrize(
    "name, val_arr, bool_arr, expected_x, expected_dodge", fix_for_create_x_and_dodge
)
def test_create_x(name, val_arr, bool_arr, expected_x, expected_dodge):
    x_arr, dodge_arr = _create_x_and_dodge(val_arr, bool_arr)
    assert_array_equal(x_arr, expected_x)


@pytest.mark.parametrize(
    "name, val_arr, bool_arr, expected_x, expected_dodge", fix_for_create_x_and_dodge
)
def test_create_dodge(name, val_arr, bool_arr, expected_x, expected_dodge):
    x_arr, dodge_arr = _create_x_and_dodge(val_arr, bool_arr)
    assert_array_equal(dodge_arr, expected_dodge)
