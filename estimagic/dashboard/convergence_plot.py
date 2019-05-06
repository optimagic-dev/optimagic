"""Functions for creating and styling the convergence plot."""
import random
from datetime import datetime

import bokeh.palettes
from bokeh.models import ColumnDataSource
from bokeh.models.annotations import Legend
from bokeh.plotting import figure
from pandas import MultiIndex


def setup_convergence_plot(param_df, start_time):
    """
    Setup the convergence plot for later updating.

    Args:
        param_df (pandas DataFrame):
            DataFrame with the initial parameter values, constraints etc.

        start_time (datetime):
            time at which the optimization started

    """
    # ToDo: split up convergence plot depending on MultiIndex and/or nr of parameters
    # ToDo: plot upper and lower bounds
    # ToDo: only plot parameters that are not fixed.

    # this must only be modified from a Bokeh session callback
    assert "time" not in param_df.index, (
        "Estimagic uses the key 'time'. "
        + "Therefore, it may not be used in the index of the parameter DataFrame."
    )

    first_entry = data_dict_from_param_values(param_df["value"], start_time)
    colors = _choose_color_palettes(param_df)
    param_data = ColumnDataSource(data=first_entry)
    conv_p = figure(plot_height=700)
    named_lines = _add_convergence_lines(
        figure=conv_p, param_data=param_data, colors=colors
    )

    # Add legend manually as our update somehow messes up the legend
    legend = Legend(items=named_lines)
    conv_p.add_layout(legend)

    # Add interactions
    conv_p.legend.click_policy = "mute"

    return conv_p, param_data


def data_dict_from_param_values(param_sr, start_time):
    """
    Convert parameter Series for adding it to a ColumnDataSource.

    Args:
        param_sr (pd.Series):
            Series with the parameter values

        start_time (datetime):
            time at which the optimization started

    """
    entry = {"time": [datetime.now() - start_time]}
    entry.update({str(k): [v] for k, v in param_sr.to_dict().items()})
    return entry


def _choose_color_palettes(param_df):
    blues = bokeh.palettes.Blues9
    # other color palettes: Greens9, Reds9, Purples9
    long_colors = bokeh.palettes.Viridis256

    index = param_df.index
    if type(index) != MultiIndex:
        if len(index) < 9:
            return blues
        else:
            return random.sample(long_colors, len(param_df))
    else:
        raise NotImplementedError(
            "MultiIndex is not supported yet by the Estimagic dashboard!"
        )


def _add_convergence_lines(figure, param_data, colors):
    line_names = [str(x) for x in sorted(param_data.column_names) if x != "time"]

    named_lines = []

    for i, name in enumerate(line_names):
        renderer = figure.line(
            source=param_data,
            x="time",
            y=name,
            line_width=2,
            name=name,
            color=colors[i],
            nonselection_alpha=0,
        )
        named_lines.append((name, [renderer]))

    return named_lines
