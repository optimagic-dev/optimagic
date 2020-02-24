"""Helper functions for the dashboard."""
import random

import bokeh.palettes
from bokeh.models import Toggle
from bokeh.models.widgets import Div
from bokeh.plotting import figure

from estimagic.optimization.utilities import index_element_to_string


# =====================================================================================
# Pandas to Bokeh Functions
# =====================================================================================


def map_groups_to_params(params):
    """Map the group name to the ColumnDataSource friendly parameter names.

    Args:
        params (pd.DataFrame):
            DataFrame with the parameter values and additional information such as the
            "group" column and Index.
    """
    group_to_params = {}
    for group in params["group"].unique():
        if group is not None:
            tup_params = params[params["group"] == group].index
            str_params = [index_element_to_string(tup) for tup in tup_params]
            group_to_params[group] = str_params
    return group_to_params


# =====================================================================================
# Custom Bokeh Elements
# =====================================================================================


def dashboard_link(name):
    """Create a link refering to *name*'s monitoring app."""
    div_name = f"link_{name}"
    text = f"<a href=./{name}> {name}!</a>"
    return Div(text=text, name=div_name, width=400)


def dashboard_toggle(database_name):
    """Create a Button that changes color when clicked displaying its boolean state.

    .. note::
        This should be a subclass but I did not get that to work.

    """
    toggle = Toggle(
        label=" Activate",
        button_type="danger",
        width=50,
        height=30,
        name=f"toggle_{database_name}",
    )

    def change_button_color(attr, old, new):
        if new is True:
            toggle.button_type = "success"
            toggle.label = "Deactivate"
        else:
            toggle.button_type = "danger"
            toggle.label = "Activate"

    toggle.on_change("active", change_button_color)
    return toggle


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
