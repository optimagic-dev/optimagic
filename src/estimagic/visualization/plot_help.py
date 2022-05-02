import itertools

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_grid_plot(
    rows,
    cols,
    ind_list,
    names,
    kws,
    x_title=None,
    y_title=None,
    clean_legend=False,
    scientific_notation=False,
    share_xax=False,
    x_min=None,
    x_max=None,
):

    """Create a dictionary for a grid plot from a list of traces.

    Args:
        rows (int): Number of rows in a plot.
        cols (int): Number of cols in a plot.
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
        plotly.Figure: The plot with subplots.

    """
    if x_title is None:
        x_title = ["" for ind in range(len(ind_list))]
    if y_title is None:
        y_title = ["" for ind in range(len(ind_list))]

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=names)
    for ind, (facet_row, facet_col) in enumerate(
        itertools.product(range(1, rows + 1), range(1, cols + 1))
    ):
        if ind + 1 > len(ind_list):
            break  # if there are empty individual plots

        traces = ind_list[ind]
        for trace in range(len(traces)):
            fig.add_trace(traces[trace], row=facet_row, col=facet_col)
            # style axis labels
            fig.update_xaxes(row=facet_row, col=facet_col, title=x_title[ind])
            fig.update_yaxes(row=facet_row, col=facet_col, title=y_title[ind])

    # deleting duplicates in legend
    if clean_legend:
        fig = clean_legend_duplicates(fig)

    # scientific notations for axis ticks
    if scientific_notation:
        fig.update_yaxes(tickformat=".2e")
        fig.update_xaxes(tickformat=".2e")

    if share_xax:
        fig.update_xaxes(range=[x_min, x_max])

    # setting template theme and size
    fig.update_layout(**kws)

    return fig


def create_ind_dict(
    ind_list,
    names,
    kws,
    x_title=None,
    y_title=None,
    clean_legend=False,
    scientific_notation=False,
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
        Dictionary of individual plots.

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
        if scientific_notation:
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
