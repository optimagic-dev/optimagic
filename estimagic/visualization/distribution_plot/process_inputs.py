"""Process inputs of the interactive distribution plot."""
import os
import warnings
from pathlib import Path

import pandas as pd

from estimagic.visualization.distribution_plot.manipulate_data import add_hist_cols
from estimagic.visualization.distribution_plot.manipulate_data import clean_data


def process_inputs(
    source, id_col, group_cols, subgroup_col, figure_height, x_padding, num_bins,
):
    df = _handle_source_type(source)
    group_cols = _process_group_cols(group_cols)
    df = clean_data(
        df=df, id_col=id_col, group_cols=group_cols, subgroup_col=subgroup_col,
    )
    df = add_hist_cols(
        df=df, group_cols=group_cols, x_padding=x_padding, num_bins=num_bins,
    )
    plot_height = _determine_plot_height(
        figure_height=figure_height, data=df, group_cols=group_cols
    )

    return df, group_cols, plot_height


def _process_group_cols(group_cols):
    if group_cols is None:
        group_cols = []
    elif isinstance(group_cols, str):
        group_cols = [group_cols]
    return group_cols


def _handle_source_type(source):
    if isinstance(source, pd.DataFrame):
        df = source
    elif isinstance(source, Path) or isinstance(source, str):
        assert os.path.exists(
            source
        ), "The path {} you specified does not exist.".format(source)
        database = load_database(path=source)  # noqa
        raise NotImplementedError("Databases not supported yet.")
    return df


def _determine_plot_height(figure_height, data, group_cols):
    """Calculate the height alloted to each plot in pixels.

    Args:
        figure_height (int): height of the entire figure in pixels
        data (pd.DataFrame): the data to be plotted

    Returns:
        plot_height (int): Plot height in pixels.

    """
    if figure_height is None:
        figure_height = 1000

    if len(group_cols) == 0:
        n_groups = 0
        n_plots = 1
    elif len(group_cols) == 1:
        n_groups = 0
        n_plots = len(data.groupby(group_cols))
    else:
        n_groups = len(data.groupby(group_cols[:-1]))
        n_plots = len(data.groupby(group_cols))
    space_of_titles = n_groups * 50
    available_space = figure_height - space_of_titles
    plot_height = int(available_space / n_plots)
    if plot_height < 20:
        warnings.warn(
            "The figure height you specified results in very small ({}) ".format(
                plot_height
            )
            + "plots which may not render well. Adjust the figure height "
            "to a larger value or set it to None to get a larger plot. "
            "Alternatively, you can click on the Reset button "
            "on the right of the plot and your plot should render correctly."
        )
    return plot_height
