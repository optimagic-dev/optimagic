"""This only tests that we get the right number of colors, not the exact colors."""
import pytest

from estimagic.visualization.colors import get_colors


palettes = [
    "categorical",
    "ordered",
    "blue",
    "red",
    "yellow",
    "green",
    "orange",
    "purple",
    "yellow-green",
    "red-blue",
]


@pytest.mark.parametrize("palette", palettes)
def test_correct_number_up_to_twelve(palette):
    for number in range(13):
        assert len(get_colors(palette, number)) == number


palettes = [
    "categorical",
    "yellow-green",
    "blue-red",
]


@pytest.mark.parametrize("palette", palettes)
def test_correct_number_up_to_24(palette):
    for number in range(12, 25):
        assert len(get_colors(palette, number)) == number
