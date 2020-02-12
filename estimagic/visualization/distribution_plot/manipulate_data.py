"""Functions to manipulate data for the interactive distribution plot."""
from functools import partial

import numpy as np
import pandas as pd

from estimagic.dashboard.plotting_functions import get_color_palette


def clean_data(df, group_cols, subgroup_col):
    cleaned = _drop_nans_and_sort(
        df=df, group_cols=group_cols, subgroup_col=subgroup_col
    )
    cleaned = _safely_reset_index(df=cleaned)
    if subgroup_col is not None:
        cleaned[subgroup_col] = _clean_subgroup_col(sr=cleaned[subgroup_col])
        cleaned["color"] = _create_color_col(sr=cleaned[subgroup_col])
    else:
        cleaned["color"] = "#035096"
    return cleaned


def add_hist_cols(df, group_cols, x_padding, num_bins):
    df = df.copy()
    df["unit_height"] = 1
    df[["binned_x", "rect_width", "xmin", "xmax"]] = _bin_width_and_midpoints(
        df=df, group_cols=group_cols, num_bins=num_bins, x_padding=x_padding,
    )

    df["dodge"] = 0.5 + df.groupby(group_cols + ["binned_x"]).cumcount()
    return df


# =====================================================================================


def _drop_nans_and_sort(df, group_cols, subgroup_col):
    """Only keep entries that have valid values in the important columns and sort.

    We sort them first by their group_cols and then by the subgroup_col.
    The later insures that bricks of the same subgroup column are stacked together.
    Within group stacks bricks are ordered by their value.
    """
    drop_and_sort_cols = group_cols.copy()
    if subgroup_col is not None:
        drop_and_sort_cols.append(subgroup_col)
    drop_and_sort_cols += ["value", "id"]
    df = df.dropna(subset=drop_and_sort_cols, how="any")
    df = df.sort_values(drop_and_sort_cols)
    return df


def _safely_reset_index(df):
    """Rename the index to avoid errors when the ColumnDataSource is constructed."""
    old_name = df.index.name
    if old_name is None or old_name in df.columns:
        i = 0
        if old_name is None:
            # double __ to ensure unique columns when bokeh transforms the DataFrame
            # to a ColumnDataSource
            new_name = "index__{}"
        else:
            new_name = old_name + "_{}"
        while new_name.format(i) in df.columns:
            i += 1
        new_df = df.copy()
        new_df.index.name = new_name.format(i)
        return new_df.reset_index()
    else:
        return df


def _clean_subgroup_col(sr):
    if len(sr.unique()) < 10:
        sr = sr.astype(str)
    else:
        try:
            sr = sr.astype(float)
        except ValueError:
            sr = sr.astype(str)
    return sr


def _create_color_col(sr):
    subgroup_vals = sorted(sr.unique())
    palette = get_color_palette(len(subgroup_vals))
    color_dict = {val: color for val, color in zip(subgroup_vals, palette)}
    return sr.replace(color_dict)


# =====================================================================================


def _bin_width_and_midpoints(df, group_cols, num_bins, x_padding):
    bin_width_and_midpoint_func = partial(
        _bin_width_and_midpoints_per_group, num_bins=num_bins, x_padding=x_padding,
    )
    if len(group_cols) <= 1:
        return bin_width_and_midpoint_func(df)
    else:
        # Exclude the last column because the last column identifies the plot
        # but we want the bins to be comparable across plots of the same (sub)group.
        grouped = df.groupby(group_cols[:-1])
        return grouped.apply(bin_width_and_midpoint_func)


def _bin_width_and_midpoints_per_group(df, num_bins, x_padding):
    xmin, xmax = _calculate_x_bounds(df, x_padding)
    bins, rect_width = np.linspace(
        start=xmin, stop=xmax, num=num_bins + 1, retstep=True
    )
    midpoints = bins[:-1] + rect_width / 2
    values_midpoints = pd.cut(df["value"], bins, labels=midpoints).astype(float)
    to_add = values_midpoints.to_frame(name="binned_x")
    to_add["rect_width"] = rect_width
    to_add["xmin"] = xmin
    to_add["xmax"] = xmax
    return to_add


def _calculate_x_bounds(df, padding):
    raw_min = df["value"].min()
    raw_max = df["value"].max()
    if "ci_lower" in df.columns:
        raw_min = min(raw_min, df["ci_lower"].min())
    if "ci_upper" in df.columns:
        raw_max = max(raw_max, df["ci_upper"].max())
    white_space = (raw_max - raw_min).clip(1e-50) * padding
    x_min = raw_min - white_space
    x_max = raw_max + white_space
    return x_min, x_max
