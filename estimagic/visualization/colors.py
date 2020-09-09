import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def get_colors(palette, number, as_cmap=False, skip_dark=0, skip_bright=0):
    """Return a list with hex codes representing a color palette.

    Args:
        palette (str): One of ["categorical", "ordered", "blue", "red", "yellow",
            "green", "orange", "purple"] or combinations of two colors, e.g.
            "red-green".
        number (int): Number of colors needed.
            - For combined color palettes like (red-blue) it is between 0 and 24
            - For non-combined ordered palettes like blue it is between 0 and 12
            - For the categorical palette it can by any non-negative integer. The colors
              are repeated if more than 12 colors are requested.
        as_cmap (bool): If True, the result is returned as matplotlib cmap.
        skip_dark (int): How many colors to skip from the dark side. Only
            available for monochrome and combined color palettes.
        skip_bright (int): How many colors to skip from the bright side. Only
            available for monochrome and combined color palettes.

    Returns:
        list or cmap: List of hex codes or cmap.

    """
    if palette in ["categorical", "ordered"]:
        assert skip_bright == skip_dark == 0

    if number < 0:
        raise ValueError("Number must be non-negative")
    if number == 0:
        res = []
    elif "-" in palette:
        actual_number = int(number + 2 * (skip_bright + skip_dark))
        if actual_number > 24:
            raise ValueError("Too many colors requested.")
        pal1, pal2 = palette.split("-")
        num1 = np.ceil(number / 2)
        num2 = np.floor(number / 2)
        res1 = _get_mono_colors(pal1, num1, skip_dark, skip_bright)
        res2 = _get_mono_colors(pal2, num2, skip_dark, skip_bright)
        res = res1 + res2[::-1]
    elif palette in ["blue", "red", "green", "yellow", "orange", "purple"]:
        res = _get_mono_colors(palette, number, skip_dark, skip_bright)
    elif palette == "categorical":
        too_long = (1 + number // 12) * CAT_LIST
        res = too_long[:number]
    else:
        if number > 12:
            raise ValueError("Too many colors requested.")
        elif palette == "ordered":
            triangle = ORDERED
        else:
            raise NotImplementedError(f"{palette} is not implemented.")
        res = triangle[number]

    if as_cmap:
        res = LinearSegmentedColormap.from_list(palette, res)
    return res


def _get_mono_colors(palette, number, skip_dark, skip_bright):
    if number == 0:
        res = []
    else:
        triangle = _mono_list_to_triangle(MONO_COLORS[palette])
        actual_number = number + skip_dark + skip_bright
        res = triangle[actual_number][skip_dark:]
        # list[:-0] = [] but we need the full list in that case
        if skip_bright != 0:
            res = res[:-skip_bright]
    return res


def _mono_list_to_triangle(mono_list):
    indices_to_delete = [5, 6, 3, 8, 0, 11, 2, 9, 4, 7, 10]
    arr = np.array(mono_list)
    triangle = {}
    for i in range(12):
        subset = np.delete(arr.copy(), indices_to_delete[:i]).tolist()
        triangle[len(subset)] = subset
    return triangle


# =====================================================================================
# Hex codes for the basic color palettes
# =====================================================================================

CAT_LIST = [
    "#547482",
    "#C87259",
    "#C2D8C2",
    "#F1B05D",
    "#818662",
    "#6C4A4D",
    "#7A8C87",
    "#EE8445",
    "#C8B05C",
    "#3C2030",
    "#C89D64",
    "#2A3B49",
]

ORDERED = {
    1: ["#547482"],
    2: ["#547482", "#c87259"],
    3: ["#547482", "#F1B05D", "#c87259"],
    4: ["#547482", "#7A8C87", "#F1B05D", "#c87259"],
    5: ["#547482", "#7A8C87", "#C2D8C2", "#F1B05D", "#c87259"],
    6: ["#547482", "#7A8C87", "#C2D8C2", "#F1B05D", "#EE8445", "#c87259"],
    7: ["#547482", "#7A8C87", "#C2D8C2", "#C8B05C", "#F1B05D", "#EE8445", "#c87259"],
    8: [
        "#547482",
        "#7A8C87",
        "#C2D8C2",
        "#C8B05C",
        "#C89D64",
        "#F1B05D",
        "#EE8445",
        "#c87259",
    ],
    9: [
        "#547482",
        "#7A8C87",
        "#C2D8C2",
        "#818662",
        "#C8B05C",
        "#C89D64",
        "#F1B05D",
        "#EE8445",
        "#c87259",
    ],
    10: [
        "#547482",
        "#7A8C87",
        "#C2D8C2",
        "#818662",
        "#C8B05C",
        "#C89D64",
        "#F1B05D",
        "#EE8445",
        "#c87259",
        "#6c4a4d",
    ],
    11: [
        "#2A3B49",
        "#547482",
        "#7A8C87",
        "#C2D8C2",
        "#818662",
        "#C8B05C",
        "#C89D64",
        "#F1B05D",
        "#EE8445",
        "#c87259",
        "#6c4a4d",
    ],
    12: [
        "#2A3B49",
        "#547482",
        "#7A8C87",
        "#C2D8C2",
        "#818662",
        "#C8B05C",
        "#C89D64",
        "#F1B05D",
        "#EE8445",
        "#c87259",
        "#6c4a4d",
        "#3C2030",
    ],
}

MONO_COLORS = {
    "blue": [
        "#547482",
        "#5c7f8e",
        "#63899a",
        "#6f92a2",
        "#7b9baa",
        "#87a4b1",
        "#93adb9",
        "#9fb6c1",
        "#abbfc8",
        "#b6c8d0",
        "#c2d1d8",
        "#cedae0",
    ],
    "red": [
        "#a04d35",
        "#b3563b",
        "#c26246",
        "#c87259",
        "#ce826c",
        "#d5937f",
        "#dba392",
        "#e0b1a3",
        "#e5bdb1",
        "#eacac0",
        "#efd6cf",
        "#f4e3de",
    ],
    "yellow": [
        "#d98213",
        "#eb8d15",
        "#ec9627",
        "#efa74b",
        "#f1b05d",
        "#f3b96f",
        "#f4c281",
        "#f6ca93",
        "#f7d3a5",
        "#f9dcb7",
        "#fae5c9",
        "#fceedb",
    ],
    "green": [
        "#606449",
        "#6b6f51",
        "#767b5a",
        "#818662",
        "#8c916a",
        "#959a75",
        "#9ea280",
        "#a6ab8c",
        "#afb397",
        "#b8bba2",
        "#c1c4ae",
        "#c9ccb9",
    ],
    "orange": [
        "#d35b13",
        "#ea6516",
        "#ec752e",
        "#ee8445",
        "#f0935c",
        "#f2a374",
        "#f4b28b",
        "#f6bf9f",
        "#f8cbb1",
        "#f9d7c3",
        "#fbe3d5",
        "#fdefe7",
    ],
    "purple": [
        "#4e3537",
        "#5d4042",
        "#6c4a4d",
        "#7b5458",
        "#8a5f63",
        "#996a6e",
        "#a2777a",
        "#a98286",
        "#b18e91",
        "#b9999c",
        "#c1a5a8",
        "#c9b1b3",
    ],
}
