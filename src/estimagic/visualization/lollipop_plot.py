import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def lollipop_plot(
    data,
    sharex=True,
    plot_bar=True,
    stripplot_kws=None,
    barplot_kws=None,
    combine_plots_in_grid=True,
):
    """Make a lollipop plot.

    Args:
        data (pandas.DataFrame): The datapoints to be plotted. In contrast
            to many seaborn functions, the whole data will be plotted. Thus if you
            want to plot just some variables or rows you need to restrict the dataset
            before passing it.
        sharex (bool): Whether the x-axis is shared across variables, default True.
        plot_bar (bool): Whether thin bars are plotted, default True.
        pairgrid_kws (dict): Keyword arguments for for the creation of a facet grid.
        Most notably, "facet_col_spacing" to control the space between facet columns.
        stripplot_kws (dict): Keyword arguments to plot the dots of the lollipop plot
            via the stripplot function. Most notably, "width" and "height".
        barplot_kws (dict): Keyword arguments to plot the lines of the lollipop plot
            via the barplot function. Most notably, "color" and "alpha". In contrast
            to seaborn, we allow for a "width" argument.
        dodge (bool): Wheter the lollipops for different datasets are plotted
            with an offset or on top of each other.
        combine_plots_in_grid (bool): decide whether to return a one
        figure containing subplots for each factor pair or a dictionary
        of individual plots. Default True.

    Returns:
        plotly.Figure: The grid plot or dict of individual plots

    """
    data, varnames = _harmonize_data(data)

    stripplot_kws = {} if stripplot_kws is None else stripplot_kws
    barplot_kws = {} if barplot_kws is None else barplot_kws

    # Draw a facet dot plot using the strip function
    g = go.Figure(
        px.strip(
            data,
            x="values",
            y="__name__",
            facet_col="indep",
            labels={"values": "", "__name__": "", "indep": ""},
            stripmode="overlay",
            **stripplot_kws
        )
    ).update_traces(jitter=0)

    g.update_layout(showlegend=False)

    # Use semantically meaningful titles for the facet
    g.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))

    # Draw lines to the plot using the barplot function
    if plot_bar:
        for col_facet, indep_name in enumerate(varnames):
            g.add_bar(
                x=data.loc[data["indep"] == indep_name, "values"],
                y=data.loc[data["indep"] == indep_name, "__name__"],
                orientation="h",
                **barplot_kws,
                row=1,
                col=col_facet + 1,
                width=0.025
            )

    # Use the same x axis limits on all columns
    if sharex:
        lower_candidate = data[["indep", "values"]].groupby("indep").min().min()
        upper_candidate = data[["indep", "values"]].groupby("indep").max().max()
        padding = (upper_candidate - lower_candidate) / 10
        lower = lower_candidate - padding
        upper = upper_candidate + padding
        g.update_yaxes(range=[lower, upper])

    # Accessing individual plots from the facet plot
    g_list = []

    for i in range(len(varnames)):

        temp = go.Figure(g["data"][i])
        if plot_bar:
            temp.add_trace(g["data"][i + len(varnames)])

        g_list.append(temp)

    if combine_plots_in_grid:
        out = g

    else:
        out = g_list

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
