"""This only tests that we get the right number of colors, not the exact colors."""
import pytest
from estimagic.visualization.colors import get_colors


def test_correct_number_categorical():
    for number in range(20):
        assert len(get_colors("categorical", number)) == number


def test_correct_number_ordered():
    for number in range(10):
        assert len(get_colors("ordered", number)) == number


def test_negative_number_raises_error():
    with pytest.raises(ValueError):
        get_colors("ordered", -15)


def test_wrong_palette_raises_error():
    with pytest.raises(ValueError):
        get_colors("red-green", 3)
