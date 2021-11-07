import numpy as np
import pandas as pd
import seaborn as sns
from estimagic.visualization.colors import get_colors


def plot_univariate_effects(
    criterion, params, n_gridpoints=21, n_random_values=2, plots_per_row=2, seed=5471
):
    """Plot criterion along coordinates at given and random values.

    Args:
        criterion (callable): criterion function. Takes a DataFrame and
            returns a scalar value or dictionary with the entry "value".
        params (pandas.DataFrame): See :ref:`params`. Must contain finite
            lower and upper bounds for all parameters.
        n_gridpoints (int): Number of gridpoints on which the criterion
            function is evaluated. This is the number per plotted line.
        n_random_values (int): Number of random parameter vectors that
            are used as center of the plots.
        plots_per_row (int): How many plots are plotted per row.


    """
    np.random.seed(seed)
    if (
        "lower_bound" not in params.columns
        or not np.isfinite(params["lower_bound"]).all()
    ):
        raise ValueError("All parameters need a finite lower bound.")
    if (
        "upper_bound" not in params.columns
        or not np.isfinite(params["upper_bound"]).all()
    ):
        raise ValueError("All parameters need a finite upper bound.")

    if "name" not in params.columns:
        names = [_index_element_to_string(tup) for tup in params.index]
        params["name"] = names

    plot_data = _get_plot_data(
        params=params,
        use_random_value=False,
        value_identifier="start values",
        n_gridpoints=n_gridpoints,
    )
    to_concat = [plot_data]

    for i in range(n_random_values):
        to_concat.append(
            _get_plot_data(
                params=params,
                use_random_value=True,
                value_identifier=f"random value {i}",
                n_gridpoints=n_gridpoints,
            )
        )

    plot_data = pd.concat(to_concat).reset_index()

    arguments = []
    for _, row in plot_data.iterrows():
        p = params.copy(deep=True)
        p["value"] = row[params.index].astype(float)
        arguments.append(p)

    function_values = [criterion(arg) for arg in arguments]
    if isinstance(function_values[0], dict):
        function_values = [val["value"] for val in function_values]

    plot_data["Criterion Value"] = function_values

    colors = get_colors("categorical", 1 + n_random_values)
    g = sns.FacetGrid(
        plot_data,
        col="name",
        hue="value_identifier",
        col_wrap=plots_per_row,
        palette=colors,
        aspect=1.5,
        sharex=False,
    )
    g.map(sns.lineplot, "Parameter Value", "Criterion Value", linewidth=2)

    return g


def _get_plot_data(params, use_random_value, value_identifier, n_gridpoints):
    if use_random_value:
        params = params.copy()
        params["value"] = np.random.uniform(
            params["lower_bound"], params["upper_bound"]
        )

    to_concat = []
    for loc in params.index:
        param_name = params.loc[loc, "name"]
        lb = params.loc[loc, "lower_bound"]
        ub = params.loc[loc, "upper_bound"]
        df = pd.DataFrame(
            data=[params["value"].to_numpy()] * n_gridpoints,
            columns=params.index,
        )
        grid = np.linspace(lb, ub, n_gridpoints)
        df[loc] = grid
        df["Parameter Value"] = grid
        df["name"] = param_name
        df["value_identifier"] = value_identifier
        to_concat.append(df)

    plot_data = pd.concat(to_concat).reset_index()
    return plot_data


def _index_element_to_string(element, separator="_"):
    if isinstance(element, (tuple, list)):
        as_strings = [str(entry).replace("-", "_") for entry in element]
        res_string = separator.join(as_strings)
    else:
        res_string = str(element)
    return res_string
