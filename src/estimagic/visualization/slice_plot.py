import numpy as np
import pandas as pd
import plotly.express as px


def slice_plot(
    criterion,
    params,
    n_gridpoints=21,
    n_random_values=2,
    plots_per_row=2,
    combine_plots_in_grid=True,
    seed=5471,
    template="plotly_white",
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
        combine_plots_in_grid (bool): decide whether to return a one
        figure containing subplots for each factor pair or a dictionary
        of individual plots. Default True.
        template (str): The template for the figure. Default is "plotly_white".

    Returns:
        plotly.Figure: The grid plot or dict of individual plots


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

    # common plotting parameters
    kws = {"x": "Parameter Value", "y": "Criterion Value", "color": "value_identifier"}

    # Plot with subplots
    if combine_plots_in_grid:

        g = px.line(
            plot_data,
            **kws,
            facet_col="name",
            facet_col_wrap=plots_per_row,
            width=400 * plots_per_row,
            height=200 * len(plot_data["name"].unique()) / plots_per_row,
        )
        g.update_layout(showlegend=False, template=template)

        out = g

    # Dictionary for individual plots
    if not combine_plots_in_grid:
        ind_dict = {}
        for i in plot_data["name"].unique():
            ind = px.line(
                plot_data[plot_data["name"] == i],
                **kws,
            )
            ind.update_layout(
                showlegend=False,
                title="name=" + str(i),
                height=300,
                width=500,
                title_x=0.5,
                template=template,
            )
            # adding to dictionary
            key = "name=" + str(i)
            ind_dict[key] = ind

        out = ind_dict

    return out


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
