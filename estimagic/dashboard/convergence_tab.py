"""Functions for creating and styling the convergence tab."""
from bokeh.layouts import column
from bokeh.models import ColumnDataSource
from bokeh.models import Panel
from tornado import gen

from estimagic.dashboard.plotting_functions import plot_with_lines
from estimagic.optimization.utilities import index_tuple_to_string

X_NAME = "XxXxITERATIONxXxX"


def setup_convergence_tab(params_df, initial_fitness):
    """
    Setup the convergence plot for later updating.

    This function is called in _setup_dashboard.

    Args:
        params_df (pandas DataFrame):
            DataFrame with the initial parameter values, constraints etc.
        initial_fitness (pd.Series):
            criterion function evaluated at the initial parameters

    """
    conv_data = _convergence_data(params_df, initial_fitness)

    fitness_plot = plot_with_lines(
        data=conv_data, y_keys=["fitness"], x_name=X_NAME, title="Fitness"
    )

    plots = [fitness_plot]

    # add plots for groups of parameters
    group_to_params = _map_groups_to_params(params_df)
    for g, params in group_to_params.items():
        group_plot = plot_with_lines(
            data=conv_data, y_keys=params, x_name=X_NAME, title=g
        )
        plots.append(group_plot)

    tab = Panel(child=column(plots), title="Convergence Plots")

    return conv_data, tab


@gen.coroutine
def update_convergence_data(new_fitness, new_params, data, rollover):
    """
    Update the convergence data with new parameters and a new fitness value.

    Args:
        new_fitness (float):
            fitness value of the new iteration
        new_params (Series):
            new parameter values
        data (ColumnDataSource):
            ColumnDataSource to stream to
        rollover (int):
            maximum number of entries to keep before dropping earlier entries
            to add new entries

    """
    iteration = max(data.data[X_NAME]) + 1
    to_add = {X_NAME: [iteration], "fitness": [new_fitness]}
    to_add.update({index_tuple_to_string(k): [new_params[k]] for k in new_params.index})
    data.stream(to_add, rollover)


def _convergence_data(params_df, initial_fitness):
    data_dict = {X_NAME: [0], "fitness": [initial_fitness]}
    params_dict = {
        index_tuple_to_string(k): [params_df["value"][k]] for k in params_df.index
    }
    data_dict.update(params_dict)
    return ColumnDataSource(data=data_dict)


def _map_groups_to_params(params_df):
    group_to_params = {}
    for group in params_df["group"].unique():
        if group is not None:
            tup_params = params_df[params_df["group"] == group].index
            str_params = [index_tuple_to_string(tup) for tup in tup_params]
            group_to_params[group] = str_params
    return group_to_params
