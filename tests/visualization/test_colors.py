"""This only tests that we get the right number of colors, not the exact colors."""
import pytest
from estimagic.visualization.colors import get_colors

BLUE = "#4e79a7"
ORANGE = "#f28e2b"
RED = "#e15759"
TEAL = "#76b7b2"
GREEN = "#59a14f"
YELLOW = "#edc948"
PURPLE = "#b07aa1"
BROWN = "#9c755f"


def test_correct_number_categorical():
    for number in range(20):
        assert len(get_colors("categorical", number)) == number


def test_correct_ordered():
    res = get_colors("ordered", 4)
    expected = [BLUE, TEAL, YELLOW, ORANGE]
    assert res == expected


def test_too_many_ordered_raises_error():
    with pytest.raises(ValueError):
        get_colors("ordered", 15)


def test_negative_number_raises_error():
    with pytest.raises(ValueError):
        get_colors("ordered", -15)


def test_wrong_palette_raises_error():
    with pytest.raises(ValueError):
        get_colors("red-green", 3)


def test_correct_categorical():
    res = get_colors("categorical", 20)
    expected = [
        BLUE,
        ORANGE,
        RED,
        TEAL,
        GREEN,
        YELLOW,
        PURPLE,
        BROWN,
        BLUE,
        ORANGE,
        RED,
        TEAL,
        GREEN,
        YELLOW,
        PURPLE,
        BROWN,
        BLUE,
        ORANGE,
        RED,
        TEAL,
    ]
    assert res == expected
