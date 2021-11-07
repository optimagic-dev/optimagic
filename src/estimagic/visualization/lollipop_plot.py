import pandas as pd
import seaborn as sns
from estimagic.visualization.colors import get_colors


def lollipop_plot(
    data,
    sharex=True,
    plot_bar=True,
    pairgrid_kws=None,
    stripplot_kws=None,
    barplot_kws=None,
    style=("whitegrid"),
    dodge=True,
):
    """Make a lollipop plot.

    Args:
        data (pandas.DataFrame): The datapoints to be plotted. In contrast
            to many seaborn functions, the whole data will be plotted. Thus if you
            want to plot just some variables or rows you need to restrict the dataset
            before passing it.
        sharex (bool): Whether the x-axis is shared across variables, default True.
        plot_bar (bool): Whether thin bars are plotted, default True.
        pairgrid_kws (dict): Keyword arguments for for the creation of a Seaborn
            PairGrid. Most notably, "height" and "aspect" to control the sizes.
        stripplot_kws (dict): Keyword arguments to plot the dots of the lollipop plot
            via the stripplot function. Most notably, "color" and "size".
        barplot_kws (dict): Keyword arguments to plot the lines of the lollipop plot
            via the barplot function. Most notably, "color" and "alpha". In contrast
            to seaborn, we allow for a "width" argument.
        style (str): A seaborn style.
        dodge (bool): Wheter the lollipops for different datasets are plotted
            with an offset or on top of each other.

    Returns:
        seaborn.PairGrid

    """
    data, varnames = _harmonize_data(data)

    sns.set_style(style)
    pairgrid_kws = {} if pairgrid_kws is None else pairgrid_kws
    stripplot_kws = {} if stripplot_kws is None else stripplot_kws
    barplot_kws = {} if barplot_kws is None else barplot_kws

    colors = get_colors("categorical", len(data))

    # Make the PairGrid
    pairgrid_kws = {
        "aspect": 0.5,
        **pairgrid_kws,
    }

    g = sns.PairGrid(
        data,
        x_vars=varnames,
        y_vars=["__name__"],
        hue="__hue__",
        **pairgrid_kws,
    )

    # Draw a dot plot using the stripplot function
    combined_stripplot_kws = {
        "size": 8,
        "orient": "h",
        "jitter": False,
        "palette": colors,
        "edgecolor": "#0000ffff",
        "dodge": dodge,
        **stripplot_kws,
    }

    g.map(sns.stripplot, **combined_stripplot_kws)

    if plot_bar:
        # Draw lines to the plot using the barplot function
        combined_barplot_kws = {
            "palette": colors,
            "alpha": 0.5,
            "width": 0.1,
            "dodge": dodge,
            **barplot_kws,
        }
        bar_height = combined_barplot_kws.pop("width")
        g.map(sns.barplot, **combined_barplot_kws)

    # Adjust the width of the bars which seaborn.barplot does not allow
    for ax in g.axes.flat:
        for patch in ax.patches:
            current_height = patch.get_height()
            diff = current_height - bar_height
            # we change the bar width
            patch.set_height(bar_height)
            # we recenter the bar
            patch.set_y(patch.get_y() + diff * 0.5)

    # Use the same x axis limits on all columns and add better labels
    if sharex:
        lower_candidate = data[varnames].min().min()
        upper_candidate = data[varnames].max().max()
        padding = (upper_candidate - lower_candidate) / 10
        lower = lower_candidate - padding
        upper = upper_candidate + padding
        g.set(xlim=(lower, upper), xlabel=None, ylabel=None)

    # Use semantically meaningful titles for the columns
    for ax, title in zip(g.axes.flat, varnames):

        # Set a different title for each axes
        ax.set(title=title)

        # Make the grid horizontal instead of vertical
        ax.xaxis.grid(False)
        ax.yaxis.grid(False)

    sns.despine(left=False, bottom=False)
    return g


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

    varnames = [col for col in combined.columns if col not in ["__hue__", "__name__"]]

    return combined, varnames


def _make_string_index(ind):
    if isinstance(ind, pd.MultiIndex):
        out = ind.map(lambda tup: "_".join((str(name) for name in tup))).tolist()
    else:
        out = ind.map(str).tolist()
    return out
