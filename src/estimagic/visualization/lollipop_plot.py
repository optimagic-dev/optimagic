import itertools
import math

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def lollipop_plot(
    data,
    sharex=True,
    plot_bar=True,
    n_rows=1,
    scatterplot_kws=None,
    barplot_kws=None,
    combine_plots_in_grid=True,
    template="plotly_white",
):
    """Make a lollipop plot.

    Args:
        data (pandas.DataFrame): The datapoints to be plotted. In contrast
            to many seaborn functions, the whole data will be plotted. Thus if you
            want to plot just some variables or rows you need to restrict the dataset
            before passing it.
        sharex (bool): Whether the x-axis is shared across variables, default True.
        plot_bar (bool): Whether thin bars are plotted, default True.
        n_rows (int): Number of rows for a grid if plots are combined
            in a grid, default 1. The number of columns is determined automatically.
        scatterplot_kws (dict): Keyword arguments to plot the dots of the lollipop plot
            via the scatter function.
        barplot_kws (dict): Keyword arguments to plot the lines of the lollipop plot
            via the barplot function.
        combine_plots_in_grid (bool): decide whether to return a one
        figure containing subplots for each factor pair or a dictionary
        of individual plots. Default True.
        template (str): The template for the figure. Default is "plotly_white".

    Returns:
        plotly.Figure: The grid plot or dict of individual plots

    """
    # adding coloring palette
    palette = px.colors.qualitative.Plotly

    data, varnames = _harmonize_data(data)

    scatter_dict = {
        "mode": "markers",
        "marker": {"color": palette[0]},
        "showlegend": False,
    }

    bar_dict = {
        "orientation": "h",
        "width": 0.03,
        "marker": {"color": palette[0]},
        "showlegend": False,
    }

    scatterplot_kws = (
        scatter_dict
        if scatterplot_kws is None
        else scatter_dict.update(
            {k: v for k, v in scatterplot_kws.items() if k not in scatter_dict}
        )
    )
    barplot_kws = (
        bar_dict
        if barplot_kws is None
        else bar_dict.update(
            {k: v for k, v in barplot_kws.items() if k not in bar_dict}
        )
    )

    # container for individual plots
    g_list = []
    # container for titles
    titles = []

    # creating data traces for plotting faceted/individual plots
    for indep_name in varnames:
        g_ind = []
        # dot plot using the scatter function
        to_plot = data[data["indep"] == indep_name]
        trace_1 = go.Scatter(x=to_plot["values"], y=to_plot["__name__"], **scatter_dict)
        g_ind.append(trace_1)

        # bar plot
        if plot_bar:
            trace_2 = go.Bar(x=to_plot["values"], y=to_plot["__name__"], **bar_dict)
        g_ind.append(trace_2)

        g_list.append(g_ind)
        titles.append(indep_name)

    if sharex:
        lower_candidate = data[["indep", "values"]].groupby("indep").min().min()
        upper_candidate = data[["indep", "values"]].groupby("indep").max().max()
        padding = (upper_candidate - lower_candidate) / 10
        lower = lower_candidate - padding
        upper = upper_candidate + padding

    # Plot with subplots
    if combine_plots_in_grid:
        n_cols = math.ceil(len(varnames) / n_rows)

        g = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=titles,
            column_widths=[100] * n_cols,
            row_heights=[60] * n_rows,
        )

        for ind, (facet_row, facet_col) in enumerate(
            itertools.product(range(1, n_rows + 1), range(1, n_cols + 1))
        ):
            if ind + 1 > len(g_list):
                break  # empty plot in a grid when odd number of ind_plots
            traces = g_list[ind]
            for trace in range(len(traces)):
                g.add_trace(traces[trace], row=facet_row, col=facet_col)
        if sharex:
            g.update_xaxes(range=[lower, upper])

        g.update_layout(
            template=template,
            height=150 * n_rows,
            width=150 * n_cols,
            margin={"l": 10, "r": 10, "t": 30, "b": 10},
        )

        out = g

    # Dictionary for individual plots
    if not combine_plots_in_grid:
        ind_dict = create_ind_dict(
            g_list,
            titles,
            share_xax=sharex,
            x_min=lower,
            x_max=upper,
            kws={
                "template": template,
                "height": 150,
                "width": 150,
                "title_x": 0.5,
                "margin": {"l": 10, "r": 10, "t": 30, "b": 10},
            },
        )
        out = ind_dict

    return out


def _harmonize_data(data):
    if not isinstance(data, list):
        data = [data]

    to_concat = []
    for i, df in enumerate(data):
        df = df.copy()
        df.columns = _make_string_index(df.columns)
        df.index = _make_string_index(df.index)
        df["__name__"] = df.index
        df["__hue__"] = i
        to_concat.append(df)

    combined = pd.concat(to_concat)
    # so that it is possibel to facet the strip plot
    new_data = pd.melt(
        combined, id_vars=["__name__", "__hue__"], var_name="indep", value_name="values"
    )

    varnames = new_data["indep"].unique()

    return new_data, varnames


def _make_string_index(ind):
    if isinstance(ind, pd.MultiIndex):
        out = ind.map(lambda tup: "_".join((str(name) for name in tup))).tolist()
    else:
        out = ind.map(str).tolist()
    return out


df = pd.DataFrame(
    np.arange(12).reshape(4, 3),
    index=pd.MultiIndex.from_tuples([(0, "a"), ("b", 1), ("a", "b"), (2, 3)]),
    columns=["a", "b", "c"],
)


def create_ind_dict(
    ind_list,
    names,
    kws,
    x_title=None,
    y_title=None,
    clean_legend=False,
    sci_notation=False,
    share_xax=False,
    x_min=None,
    x_max=None,
):
    """Create a dictionary for individual plots from a list of traces.

    Args:
        ind_list (iterable): The list of traces for each individual plot.
        names (iterable): The list of titles for the each plot.
        kws (dict): The dictionary for the layout.update, unified for each
        individual plot.
        x_title (iterable or None): The list of x-axis labels for each plot. If None,
        then no labels are added.
        y_title (iterable or None): The list of y-axis labels for each plot. If None,
        then no labels are added.
        clean_legend (bool): If True, then cleans the legend from duplicates.
        Default False.
        sci_notation (bool): If True then updates the ticks on x- and y-axis to
        be displayed in a scientific notation. Default False.
        share_xax (bool): If True, then the x-axis domain is the same
        for each individual plot.
        x_min (int or None): The lower bound for share_xax.
        x_max (int or None): The upped bound for share_xax.

    Returns:
        dictionary of individual plots

    """
    fig_dict = {}
    if x_title is None:
        x_title = ["" for ind in range(len(ind_list))]
    if y_title is None:
        y_title = ["" for ind in range(len(ind_list))]

    for ind in range(len(ind_list)):
        fig = go.Figure()
        traces = ind_list[ind]
        for trace in range(len(traces)):
            fig.add_trace(traces[trace])
        # adding title and styling axes and theme
        fig.update_layout(
            title=names[ind], xaxis_title=x_title[ind], yaxis_title=y_title[ind], **kws
        )
        # scientific notations for axis ticks
        if sci_notation:
            fig.update_yaxes(tickformat=".2e")
            fig.update_xaxes(tickformat=".2e")
        # deleting duplicates in legend
        if clean_legend:
            fig = clean_legend_duplicates(fig)
        if share_xax:
            fig.update_xaxes(range=[x_min, x_max])
        # adding to dictionary
        key = names[ind].replace(" ", "_").lower()
        fig_dict[key] = fig

    return fig_dict


def clean_legend_duplicates(fig):
    names = set()
    fig.for_each_trace(
        lambda trace: trace.update(showlegend=False)
        if (trace.name in names)
        else names.add(trace.name)
    )
    return fig
