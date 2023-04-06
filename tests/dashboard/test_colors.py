"""This only tests that we get the right number of colors, not the exact colors."""
import pytest
from tranquilo.dashboard.colors import get_colors


def test_correct_number_categorical():
    for number in range(20):
        assert len(get_colors("categorical", number)) == number


def test_wrong_palette_raises_error():
    with pytest.raises(ValueError):
        get_colors("red-green", 3)
