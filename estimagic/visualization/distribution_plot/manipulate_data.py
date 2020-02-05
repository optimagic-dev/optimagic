"""Functions to manipulate data for the interactive distribution plot."""
from functools import partial

import bokeh.palettes
import numpy as np
import pandas as pd


def clean_data(df, value_col, id_col, group_cols, subgroup_col):
    cleaned = _drop_nans_and_sort(
        df=df,
        group_cols=group_cols,
        subgroup_col=subgroup_col,
        value_col=value_col,
        id_col=id_col,
    )
    cleaned = _safely_reset_index(df=cleaned)
    if subgroup_col is not None:
        cleaned[subgroup_col] = _clean_subgroup_col(sr=cleaned[subgroup_col])
        cleaned["color"] = _create_color_col(sr=cleaned[subgroup_col])
    else:
        cleaned["color"] = "#035096"
    return cleaned


def add_hist_cols(
    df, value_col, group_cols, lower_bound_col, upper_bound_col, x_padding, num_bins
):
    df = df.copy()
    df[["binned_x", "rect_width", "xmin", "xmax"]] = _bin_width_and_midpoints(
        df=df,
        group_cols=group_cols,
        value_col=value_col,
        lower_bound_col=lower_bound_col,
        upper_bound_col=upper_bound_col,
        num_bins=num_bins,
        x_padding=x_padding,
    )

    df["dodge"] = 0.5 + df.groupby(group_cols + ["binned_x"]).cumcount()
    return df


# =====================================================================================


def _drop_nans_and_sort(df, group_cols, subgroup_col, value_col, id_col):
    drop_and_sort_cols = group_cols.copy()
    if subgroup_col is not None:
        drop_and_sort_cols.append(subgroup_col)
    drop_and_sort_cols += [value_col, id_col]
    df = df.dropna(subset=drop_and_sort_cols, how="any")
    df.sort_values(drop_and_sort_cols, inplace=True)
    return df


def _safely_reset_index(df):
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


def _bin_width_and_midpoints(
    df, group_cols, value_col, lower_bound_col, upper_bound_col, num_bins, x_padding
):
    bin_width_and_midpoint_func = partial(
        _bin_width_and_midpoints_per_group,
        value_col=value_col,
        num_bins=num_bins,
        x_padding=x_padding,
        lower_bound_col=lower_bound_col,
        upper_bound_col=upper_bound_col,
    )
    if len(group_cols) <= 1:
        return bin_width_and_midpoint_func(df)
    else:
        # Exclude the last column because the last column identifies the plot
        # but we want the bins to be comparable across plots of the same (sub)group.
        grouped = df.groupby(group_cols[:-1])
        return grouped.apply(bin_width_and_midpoint_func)


def _bin_width_and_midpoints_per_group(
    df, value_col, lower_bound_col, upper_bound_col, num_bins, x_padding
):
    xmin, xmax = _calculate_x_bounds(
        df, value_col, lower_bound_col, upper_bound_col, x_padding
    )
    bins, rect_width = np.linspace(
        start=xmin, stop=xmax, num=num_bins + 1, retstep=True
    )
    midpoints = bins[:-1] + rect_width / 2
    values_midpoints = pd.cut(df[value_col], bins, labels=midpoints).astype(float)
    to_add = values_midpoints.to_frame(name="binned_x")
    to_add["rect_width"] = rect_width
    to_add["xmin"] = xmin
    to_add["xmax"] = xmax
    return to_add


def _calculate_x_bounds(df, value_col, lower_bound_col, upper_bound_col, padding):
    raw_min = df[value_col].min()
    raw_max = df[value_col].max()
    if lower_bound_col is not None:
        raw_min = min(raw_min, df[lower_bound_col].min())
    if upper_bound_col is not None:
        raw_max = max(raw_max, df[upper_bound_col].max())
    white_space = (raw_max - raw_min).clip(1e-50) * padding
    x_min = raw_min - white_space
    x_max = raw_max + white_space
    return x_min, x_max


# =====================================================================================


def get_color_palette(nr_colors):
    """Return list of colors depending on the number needed."""
    # color tone palettes: bokeh.palettes.Blues9, Greens9, Reds9, Purples9.
    if nr_colors == 1:
        return ["firebrick"]
    elif nr_colors == 2:
        return ["darkslateblue", "goldenrod"]
    elif nr_colors < 20:
        return bokeh.palettes.Category20[nr_colors]
    else:
        return np.random.choice(bokeh.palettes.Category20[20], nr_colors)
