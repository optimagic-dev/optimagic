import numpy as np
import pandas as pd
import plotly.express as px
from estimagic.batch_evaluators import process_batch_evaluator
from estimagic.config import DEFAULT_N_CORES
from estimagic.config import PLOTLY_PALETTE
from estimagic.config import PLOTLY_TEMPLATE
from estimagic.visualization.plotting_utilities import combine_plots
from estimagic.visualization.plotting_utilities import get_layout_kwargs


def slice_plot(
    criterion,
    params,
    batch_evaluator=None,
    n_cores=DEFAULT_N_CORES,
    param_name_mapping=None,
    n_gridpoints=21,
    n_random_values=0,
    seed=5471,
    share_yrange_all=True,
    expand_yrange=0.02,
    share_xrange_all=False,
    colorscale=PLOTLY_PALETTE,
    template=PLOTLY_TEMPLATE,
    showlegend=True,
    layout_kwargs=None,
    legend_kwargs=None,
    title_kwargs=None,
    return_dict=False,
    plots_per_row=2,
    sharex=False,
    sharey=True,
    make_subplot_kwargs=None,
    clean_legend=True,
):
    """Plot criterion along coordinates at given and random values.

    Generates individual plots for each parameter, combines them into a figure with
    subplots.

    Args:
        criterion (callable): criterion function. Takes a DataFrame and returns a
            scalar value or dictionary with the entry "value".
        params (pandas.DataFrame): See :ref:`params`. Must contain finite lower and
            upper bounds for all parameters.
        batch_evaluator (str or callable): See :ref:`batch_evaluators`.
        n_cores (int): Number of cores.
        param_name_mapping (dict or NoneType): Dictionary mapping old parameter names
            to new ones.
        n_gridpoins (int): Number of gridpoints on which the criterion function is
            evaluated. This is the number per plotted line.
        n_random_values (int): Number of random parameter vectors that are used as
            center of the plots.
        seed (int): Numpy randoms seed used when generating the random values.
        share_yrange_all (bool): If True, the individual plots share the scale on the
            yaxis.
        share_xrange_all (bool): If True, set the same range of x axis for all plots.
        expand_y (float): The ration by which to expand the range of the (shared) y
            axis, such that the axis is not cropped at exactly max of Criterion Value.
        colorscale: The coloring palette for traces. Default is "qualitative.Set2".
        template (str): The template for the figure. Default is "plotly_white".
        showlegend (bool): If True, show legend.
        layout_kwargs (dict or NoneType): Dictionary of key word arguments used to
            update layout of plotly Figure object. If None, the default kwargs defined
            in the function will be used.
        legend_kwargs (dict or NoneType): Dictionary of key word arguments used to
            update position, orientation and title of figure legend. If None, default
            position and orientation will be used with no title.
        title_kwargs (dict or NoneType): Dictionary of key word arguments used to
            update properties of the figure title. Use {'text': '<desired title>'}
            to set figure title.
        return_dict (bool): If True, return dictionary with individual plots of each
            parameter, else, ombine individual plots into a figure with subplots.
        plots_per_row (int): Number of plots per row.
        sharex (bool): Whether to share the properties of x-axis across subplots. In
            the sam column
        sharey (bool): If True, share the properties of y-axis across subplots in the
        make_subplot_kwargs (dict or NoneType): Dictionary of keyword arguments used
            to instantiate plotly Figure with multiple subplots. Is used to define
            properties such as, for example, the spacing between subplots (governed by
            'horizontal_spacing' and 'vertical_spacing'). If None, default arguments
            defined in the function are used.
        clean_legend (bool): If True, then cleans the legend from duplicates.

    Returns:
        out (dict or plotly.Figure): Returns either dictionary with individual slice
            plots for each parameter or a plotly Figure combining the individual plots.


    """

    params = params.copy(deep=True)

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
    param_names = plot_data["name"].unique()
    param_name_mapping = _process_names_mapping(param_name_mapping, param_names)
    arguments = []
    for _, row in plot_data.iterrows():
        p = params.copy(deep=True)
        p["value"] = row[params.index].astype(float)
        arguments.append(p)
    batch_evaluator = process_batch_evaluator(batch_evaluator)

    function_values = batch_evaluator(
        criterion,
        arguments=arguments,
        unpack_symbol=None,
        error_handling="raise",
        n_cores=n_cores,
    )
    if isinstance(function_values[0], dict):
        function_values = [val["value"] for val in function_values]

    plot_data["Criterion Value"] = function_values
    lb = plot_data["Criterion Value"].min()
    ub = plot_data["Criterion Value"].max()
    y_range = ub - lb
    yaxis_ub = ub + y_range * expand_yrange
    yaxis_lb = lb - y_range * expand_yrange
    layout_kwargs = get_layout_kwargs(
        layout_kwargs, legend_kwargs, title_kwargs, template, showlegend
    )

    plots_dict = {}
    for par_name in plot_data["name"].unique():
        df = plot_data[plot_data["name"] == par_name]
        subfig = px.line(
            df,
            y="Criterion Value",
            x="Parameter Value",
            color="value_identifier",
            color_discrete_sequence=colorscale,
        )
        subfig.update_layout(**layout_kwargs)
        subfig.update_xaxes(title={"text": param_name_mapping[par_name]})
        subfig.update_yaxes(title={"text": "Criterion Value"})
        if share_yrange_all is True:
            subfig.update_yaxes(range=[yaxis_lb, yaxis_ub])
        plots_dict[par_name] = subfig
    if return_dict:
        out = plots_dict
    else:
        plots = list(plots_dict.values())
        out = combine_plots(
            plots=plots,
            plots_per_row=plots_per_row,
            sharex=sharex,
            sharey=sharey,
            share_yrange_all=share_yrange_all,
            share_xrange_all=share_xrange_all,
            expand_yrange=expand_yrange,
            make_subplot_kwargs=make_subplot_kwargs,
            showlegend=showlegend,
            template=template,
            clean_legend=clean_legend,
            layout_kwargs=layout_kwargs,
            legend_kwargs=legend_kwargs,
            title_kwargs=title_kwargs,
        )
    return out


def _get_plot_data(params, use_random_value, value_identifier, n_gridpoints):
    params = params.copy(deep=True)
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


def _process_names_mapping(params_mapping, old_names):
    """Get dictionary mappping old parameter names to new ones."""
    if params_mapping is None:
        params_mapping = {par: par for par in old_names}
    else:
        for par in old_names:
            if par not in params_mapping:
                params_mapping[par] = par
    return params_mapping
