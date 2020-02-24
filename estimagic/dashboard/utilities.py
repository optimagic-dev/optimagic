"""Helper functions for the dashboard."""
import random
import socket
from contextlib import closing
from pathlib import Path

import bokeh.palettes
from bokeh.models.widgets import Div
from bokeh.plotting import figure


# =====================================================================================
# Prettification and Making Compatible of Strings, Names and the Like
# =====================================================================================


def short_name_to_database_path(path_list):
    """Generate short but unique names from each path.

    Args:
        path_list (list):
            List of strings or pathlib.path to the optimizations' databases.

    Returns:
        list: List of strings with names.

    Example:

    >>> pl = ["bla/blubb/blabb.db", "a/b", "bla/blabb"]
    >>> short_and_unique_optimization_names(pl)
    ['blubb/blabb', 'b', 'bla/blabb']

    """
    without_suffixes = [Path(p).resolve().with_suffix("") for p in path_list]
    # The assert statement makes sure that the while loop terminates
    assert len(set(without_suffixes)) == len(
        without_suffixes
    ), "path_list must not contain duplicates."
    short_name_to_path = {}
    for path, path_with_suffix in zip(without_suffixes, path_list):
        parts = tuple(reversed(path.parts))
        needed_parts = 1
        candidate = parts[:needed_parts]
        while _name_clash(candidate, without_suffixes):
            needed_parts += 1
            candidate = parts[:needed_parts]

        short_name = "/".join(reversed(candidate))
        short_name_to_path[short_name] = path_with_suffix
    return short_name_to_path


def _name_clash(candidate, path_list, allowed_occurences=1):
    """Determine if candidate leads to a name clash.

    Args:
        candidate (tuple): tuple with parts of a path.
        path_list (list): List of pathlib.path
        allowed_occurences (int): How often a name can occur before
            we call it a clash.

    Returns:
        bool

    """
    duplicate_counter = -allowed_occurences
    for path in path_list:
        parts = tuple(reversed(path.parts))
        if len(parts) >= len(candidate) and parts[: len(candidate)] == candidate:
            duplicate_counter += 1
    return duplicate_counter > 0


# =====================================================================================
# Custom Bokeh Elements
# =====================================================================================


def dashboard_link(name):
    """Create a link refering to *name*'s monitoring app."""
    div_name = f"link_{name}"
    text = f"<a href=./{name}> {name}!</a>"
    return Div(text=text, name=div_name, width=400)


# =====================================================================================
# Styling Functions
# =====================================================================================


def create_wide_figure(title, tooltips=None):
    """Return a styled, empty figure of predetermined height and width."""
    fig = figure(plot_height=350, plot_width=700, title=title, tooltips=tooltips)
    fig.title.text_font_size = "15pt"
    fig.min_border_left = 50
    fig.min_border_right = 50
    fig.min_border_top = 20
    fig.min_border_bottom = 50
    fig.toolbar_location = None
    return fig


def get_color_palette(nr_colors):
    """Return list of colors depending on the number needed."""
    if nr_colors == 1:
        return ["firebrick"]
    elif nr_colors == 2:
        return ["darkslateblue", "goldenrod"]
    elif nr_colors <= 10:
        return bokeh.palettes.Category10[nr_colors]
    else:
        return random.choices(bokeh.palettes.Turbo256, k=nr_colors)


# =====================================================================================


def find_free_port():
    """Find a free port on the localhost.

    Adapted from https://stackoverflow.com/a/45690594
    """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("localhost", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
