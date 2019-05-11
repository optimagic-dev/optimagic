"""Functions for creating and styling the convergence plot."""
import random
from functools import partial

import bokeh.palettes
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from tornado import gen


def setup_convergence_tab(param_df, initial_fitness):
    """
    Setup the convergence plot for later updating.

    This function is called in _setup_dashboard.

    Args:
        param_df (pandas DataFrame):
            DataFrame with the initial parameter values, constraints etc.
        initial_fitness (pd.Series):
            criterion function evaluated at the initial parameters

    """
    fitness_plot, fitness_data = _fitness_plot(initial_fitness=initial_fitness)
    param_plots, param_data = _parameter_plots(param_df=param_df)
    plots = [fitness_plot] + param_plots
    datasets = [fitness_data] + param_data
    return plots, datasets


def update_convergence_tab(doc, queue, datasets):
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

        datasets (list):
            list of ColumnDataSource storing earlier parameter iterations.

    """
    fitness_data, param_data = datasets

    if queue.qsize() > 0:
        new_params, fitness = queue.get()

        doc.add_next_tick_callback(
            partial(_update_convergence_plot, data=param_data, new_values=new_params)
        )
        doc.add_next_tick_callback(
            partial(_update_convergence_plot, data=fitness_data, new_values=fitness)
        )


def _fitness_plot(initial_fitness):
    fitness_data = ColumnDataSource(
        data=_convert_sr_for_cds(sr=initial_fitness, iteration=0)
    )
    fitness_p = _wide_figure(title="Fitness")
    fitness_p.line(
        source=fitness_data,
        x="XxXxITERATIONxXxX",
        y="fitness",
        line_width=1,
        name="fitness",
        color="firebrick",
    )

    return fitness_p, fitness_data


def _parameter_plots(param_df):
    """
    Create the plots that show the convergence of (groups of) parameters.

    ToDo: split up convergence plot depending on column in param_df.
    ToDo: plot upper and lower bounds.
    ToDo: only plot parameters that are not fixed.

    Args:
        param_df (pandas DataFrame):
            DataFrame with the initial parameter values, constraints etc.


    """

    colors = _choose_color_palettes(param_df)

    first_entry = _convert_sr_for_cds(sr=param_df["value"], iteration=0)
    param_data = ColumnDataSource(data=first_entry)

    conv_p = _wide_figure(title="Parameter Values")

    _add_convergence_lines(figure=conv_p, param_data=param_data, colors=colors)

    return [conv_p], [param_data]


def _choose_color_palettes(param_df):
    # color tone palettes: bokeh.palettes.Blues9, Greens9, Reds9, Purples9.
    nr_colors = len(param_df)
    if nr_colors < 20:
        return bokeh.palettes.Category20[nr_colors]
    else:
        random.sample(bokeh.pallettes.Category20[20], nr_colors)


def _add_convergence_lines(figure, param_data, colors):
    iteration_name = "XxXxITERATIONxXxX"
    line_names = [
        str(x) for x in sorted(param_data.column_names) if x != iteration_name
    ]
    for i, name in enumerate(line_names):
        figure.line(
            source=param_data,
            x=iteration_name,
            y=name,
            line_width=1,
            name=name,
            color=colors[i],
            nonselection_alpha=0,
        )


def _convert_sr_for_cds(sr, iteration):
    """
    Convert parameter Series for adding it to a ColumnDataSource.

    Args:
        sr (pd.Series):
            Series with the new value(s)

        iteration (int):
            iteration of the parameter vector
    """
    entry = {"XxXxITERATIONxXxX": [iteration]}
    entry.update({str(k): [v] for k, v in sr.to_dict().items()})
    return entry


@gen.coroutine
def _update_convergence_plot(new_values, data):
    iteration = max(data.data["XxXxITERATIONxXxX"]) + 1
    to_add = _convert_sr_for_cds(sr=new_values, iteration=iteration)
    data.stream(to_add)


def _wide_figure(title):
    return figure(plot_height=350, plot_width=700, title=title)
