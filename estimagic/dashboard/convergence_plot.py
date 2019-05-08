"""Functions for creating and styling the convergence plot."""
import random
from functools import partial

import bokeh.palettes
from bokeh.models import ColumnDataSource
from bokeh.models.annotations import Legend
from bokeh.plotting import figure
from tornado import gen


def setup_convergence_plot(param_df):
    """
    Setup the convergence plot for later updating.

    This function is called in _setup_dashboard.

    Args:
        param_df (pandas DataFrame):
            DataFrame with the initial parameter values, constraints etc.
    """
    # ToDo: split up convergence plot depending on MultiIndex and/or nr of parameters
    # ToDo: plot upper and lower bounds
    # ToDo: only plot parameters that are not fixed.

    assert "iteration" not in param_df.index, (
        "Estimagic uses the key 'iteration'. "
        + "Therefore, it may not be used in the index of the parameter DataFrame."
    )

    colors = _choose_color_palettes(param_df)
    first_entry = _convert_parmas_for_cds(param_sr=param_df["value"], iteration=0)
    param_data = ColumnDataSource(data=first_entry)

    conv_p = figure(plot_height=700, plot_width=1400)

    named_lines = _add_convergence_lines(
        figure=conv_p, param_data=param_data, colors=colors
    )

    # Add legend manually as our update somehow messes up the legend
    legend = Legend(items=named_lines)
    conv_p.add_layout(legend)

    # Add interactions
    conv_p.legend.click_policy = "mute"

    return conv_p, param_data


def update_convergence_plot(doc, queue, param_data):
    """
    Check for new param values and update the plot.

    This function is called in a never ending while loop in _update_dashboard.

    Args:
        doc (bokeh Document):
            document instance where the Dashboard will be stored.
            Note this must stay the first argument for the bokeh FunctionHandler
            to work properly.

        queue (Queue):
            queue to which originally the parameters DataFrame is supplied and to which
            the updated parameter Series will be supplied later.

        param_data (ColumnDataSource):
            ColumnDataSource storing earlier parameter iterations.

    """
    if queue.qsize() > 0:
        new_params = queue.get()

        doc.add_next_tick_callback(
            partial(_update_convergence_plot, data=param_data, new_params=new_params)
        )


def _choose_color_palettes(param_df):
    # color tone palettes: bokeh.palettes.Blues9, Greens9, Reds9, Purples9.
    long_colors = bokeh.palettes.Viridis256
    return random.sample(long_colors, len(param_df))


def _add_convergence_lines(figure, param_data, colors):
    line_names = [str(x) for x in sorted(param_data.column_names) if x != "iteration"]

    named_lines = []

    for i, name in enumerate(line_names):
        renderer = figure.line(
            source=param_data,
            x="iteration",
            y=name,
            line_width=2,
            name=name,
            color=colors[i],
            nonselection_alpha=0,
        )
        named_lines.append((name, [renderer]))

    return named_lines


def _convert_parmas_for_cds(param_sr, iteration):
    """
    Convert parameter Series for adding it to a ColumnDataSource.

    Args:
        param_sr (pd.Series):
            Series with the parameter values

        iteration (int):
            iteration of the parameter vector
    """
    entry = {"iteration": [iteration]}
    entry.update({str(k): [v] for k, v in param_sr.to_dict().items()})
    return entry


@gen.coroutine
def _update_convergence_plot(new_params, data):
    iteration = max(data.data["iteration"]) + 1
    to_add = _convert_parmas_for_cds(param_sr=new_params, iteration=iteration)
    data.stream(to_add)
