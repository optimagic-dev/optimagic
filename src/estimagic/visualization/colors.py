import seaborn as sns


def get_colors(palette, number):
    """Return a list with hex codes representing a color palette.

    Args:
        palette (str): One of ["categorical", "ordered"]
        number (int): Number of colors needed. Colors are repeated if more than eight
            colors are requested.

    Returns:
        list: List of hex codes.

    """
    blue = "#4e79a7"
    orange = "#f28e2b"
    red = "#e15759"
    teal = "#76b7b2"
    green = "#59a14f"
    yellow = "#edc948"
    purple = "#b07aa1"
    brown = "#9c755f"

    palette_to_colors = {
        "categorical": [blue, orange, red, teal, green, yellow, purple, brown],
        "ordered": [blue, teal, yellow, orange, red, purple],
    }

    if number < 0:
        raise ValueError("Number must be non-negative")
    if palette not in palette_to_colors.keys():
        raise ValueError(
            f"palette must be in {palette_to_colors.keys()}. You specified {palette}."
        )
    colors = palette_to_colors[palette]

    # if many ordered colors are requested switch to using the nipy_spectral color map
    if palette == "ordered" and number > len(colors):
        res = sns.color_palette("nipy_spectral", number)
    else:
        n_full_repetitions = number // len(colors)
        modulus = number % len(colors)
        res = n_full_repetitions * colors + colors[:modulus]
    return res
