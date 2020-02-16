"""Library with low level Bokeh functions for building the dashboard."""
import random

import bokeh.palettes
from bokeh.plotting import figure


def create_wide_figure(title, tooltips=None):
    """Return an empty figure of predetermined height and width."""
    fig = figure(plot_height=350, plot_width=700, title=title, tooltips=tooltips)
    fig.title.text_font_size = "15pt"
    fig.min_border_left = 50
    fig.min_border_right = 50
    fig.min_border_top = 20
    fig.min_border_bottom = 50
    fig.toolbar_location = None
    fig.xaxis.axis_label_text_font_style = "normal"
    fig.xaxis.axis_label_text_font_size = "12pt"
    return fig


def get_color_palette(nr_colors):
    """Return list of colors depending on the number needed."""
    # color tone palettes: bokeh.palettes.Blues9, Greens9, Reds9, Purples9.
    if nr_colors == 1:
        return ["firebrick"]
    elif nr_colors == 2:
        return ["darkslateblue", "goldenrod"]
    elif nr_colors < 20:
        return bokeh.palettes.Category20[nr_colors]
    else:
        return random.choices(bokeh.palettes.Category20[20], k=nr_colors)
