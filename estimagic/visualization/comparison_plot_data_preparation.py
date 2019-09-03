"""
Process a list of estimagic optimization results for drawing a comparison plot.

"""
import warnings

import numpy as np
import pandas as pd


MEDIUMELECTRICBLUE = "#035096"


def comparison_plot_inputs(results, x_padding, num_bins, color_dict, fig_height):
    """Generate the inputs for the comparison plot function.

    Args:
        results (list): List of estimagic optimization results where the info
            has been extended with 'model' and 'model_name'
        x_padding (float): the x_range is extended on each side by x_padding
            times the range of the data
        num_bins (int): number of bins
        color_dict (dict): mapping from the model class names to colors.
        fig_height (int): height the entire plot should have

    Returns:
        source_dfs (dict): map from parameter identifiers to DataFrames
            with everything we need for the comparison plot
        group_plot_info (dict): map from parameter group to x_range,
        general_plot_info (dict):
    """
    parameter_groups = _consolidate_parameter_attribute(results, "group")
    parameter_names = _consolidate_parameter_attribute(results, "name")
    all_data = _combine_params_data(
        results, parameter_groups, parameter_names, color_dict
    )

    x_min, x_max = _calculate_x_bounds(all_data, x_padding)
    bins, rect_width = _calculate_bins_and_rectangle_width(x_min, x_max, num_bins)

    groups = parameter_groups.unique()
    source_dfs = {group: {} for group in groups}
    y_max = 5
    for param in parameter_groups.index:
        group = parameter_groups.loc[param]
        sdf = all_data.loc[param]
        sdf = sdf.set_index("model", drop=True)
        sdf["binned_x"] = _replace_by_bin_midpoint(sdf["value"], bins.loc[group])
        sdf["dodge"] = _calculate_dodge(sdf["value"], bins.loc[group])
        sdf["dodge"] = sdf["dodge"].where(sdf["value"].notnull(), -10)
        source_dfs[group][param] = sdf.reset_index()
        y_max = int(max(y_max, sdf["dodge"].max() + 1))

    plot_height = _determine_plot_height(
        figure_height=fig_height,
        y_max=y_max,
        n_params=len(all_data.index.unique()),
        n_groups=len(groups),
    )

    plot_info = _create_plot_info(
        x_min=x_min,
        x_max=x_max,
        rect_width=rect_width,
        y_max=y_max,
        plot_height=plot_height,
    )

    return source_dfs, plot_info


def _consolidate_parameter_attribute(results, attribute, wildcards=None):
    """Consolidate attributes of parameters are specified in several results.

    Args:
        results (list): List of optimization results
        attribute (str): Name of the column in params that holds the attribute
        wildcards (list, optional): Values that are compatible with anything

    Returns:
        consolidated (pd.Series): The index are the union of all parameter indices.
        The values are the consolidated attribute of that parameter.

    """
    wildcards = [None, np.nan] if wildcards is None else wildcards
    data = pd.concat([res.params[attribute] for res in results], axis=1)

    def _consolidate(x):
        unique_values = set(x.unique()).difference(set(wildcards))
        assert len(unique_values) <= 1, (
            f"{attribute} have to be compatible across models, i.e. the {attribute} "
            f"for the same parameters have to be equal or {wildcards}."
        )
        if len(unique_values) == 0:
            return None
        else:
            return list(unique_values)[0]

    consolidated = data.apply(_consolidate, axis=1)
    consolidated.name = attribute
    return consolidated


def _combine_params_data(results, parameter_groups, parameter_names, color_dict):
    """Combine the params fields of the results across models.

     Args:
        results (list): List of estimagic optimization results where the info
            has been extended with 'model' and 'model_name'
        parameter_groups (Series): maps parameters to parameter group
        parameter_names (Series): maps parameters to pretty names

    Returns:
        df (DataFrame): A DataFrame in long format. The columns are
            - 'value': Parameter values
            - 'conf_int_lower': Lower bound of confidence intervals
            - 'conf_int_upper': Upper bound of confidence intervals
            - 'model': Name of the model
            - 'model_class': Class of the model
            - 'group': Parameter group
            - 'name': Pretty name of the parameter. Not necessarily unique.

    """
    relevant = ["value", "conf_int_lower", "conf_int_upper"]
    used_model_names = []
    res_dfs = []
    for i, res in enumerate(results):
        small_params = res.params[res.params.columns & relevant].copy()
        params = pd.concat([small_params, parameter_groups, parameter_names], axis=1)
        params, used_model_names = _add_model_name(
            df=params,
            info=res.info,
            missing_val=str(i),
            used_model_names=used_model_names,
        )
        params = _add_model_class_and_color(
            df=params, info=res.info, color_dict=color_dict
        )
        res_dfs.append(params)

    all_data = pd.concat(res_dfs)
    all_data = _ensure_correct_conf_ints(all_data)
    all_data["group"].replace({None: np.nan}, inplace=True)
    all_data.dropna(subset=["group"], inplace=True)
    return all_data


def _add_model_name(df, info, missing_val, used_model_names):
    model_name = info.get("model_name", missing_val)
    if model_name in used_model_names:
        model_name = model_name + "_" + missing_val
    df["model"] = model_name
    used_model_names.append(model_name)
    return df, used_model_names


def _add_model_class_and_color(df, info, color_dict):
    if color_dict is None:
        color_dict = {}
    model_class = info.get("model_class", "no model class")
    df["model_class"] = model_class
    df["color"] = color_dict.get(model_class, MEDIUMELECTRICBLUE)
    return df


def _ensure_correct_conf_ints(df):
    if "conf_int_upper" not in df.columns:
        df["conf_int_upper"] = np.nan
    if "conf_int_lower" not in df.columns:
        df["conf_int_lower"] = np.nan
    nr_nans_in_cis = df[["conf_int_lower", "conf_int_upper"]].isnull().sum(axis=1)
    assert all(
        nr_nans_in_cis.isin([0, 2])
    ), "For some models there is only one of the two confidence bounds given."
    return df


def _calculate_x_bounds(params_data, padding):
    """Calculate the lower and upper ends of the x-axis for each group.

    Args:
        params_data (df): see _combine_params_data
        padding (float): the x_range is extended on each side by x_padding
            times the range of the data

    Returns:
        x_min (Series): The index are the parameter groups. The values are
            the left bound of the x-axis for this parameter group
        x_max (Series): Same as x_min but for right bound

    """
    raw_min = params_data.groupby("group").min().min(axis=1)
    raw_max = params_data.groupby("group").max().max(axis=1)
    white_space = (raw_max - raw_min).clip(1e-50) * padding
    x_min = raw_min - white_space
    x_max = raw_max + white_space
    x_min.name = "x_min"
    x_max.name = "x_max"
    return x_min, x_max


def _calculate_bins_and_rectangle_width(x_min, x_max, num_bins):
    bins_transposed, stepsize = np.linspace(
        start=x_min, stop=x_max, num=num_bins + 1, retstep=True
    )
    bins = pd.DataFrame(data=bins_transposed.T, index=x_min.index)
    rectangle_width = pd.Series(data=stepsize, index=x_min.index, name="width")
    return bins, rectangle_width


def _replace_by_bin_midpoint(values, bins):
    midpoints = (bins + bins.shift(periods=-1))[:-1] / 2
    sr = pd.cut(values, bins, labels=midpoints).astype(float)
    sr.fillna(bins[0], inplace=True)
    return sr


def _calculate_dodge(values, bins):
    df = values.to_frame()
    df["bin"] = pd.cut(values, bins, labels=range(len(bins) - 1))
    dodge = 0.5 + df.groupby("bin").cumcount()
    return dodge


def _create_plot_info(x_min, x_max, rect_width, y_max, plot_height):
    group_plot_info = pd.concat([x_min, x_max, rect_width], axis=1)
    group_plot_info["x_range"] = group_plot_info.apply(
        lambda x: (x["x_min"], x["x_max"]), axis=1
    )
    group_plot_info.drop(columns=["x_min", "x_max"], inplace=True)
    group_plot_info = group_plot_info.T.to_dict()
    plot_info = {
        "plot_height": plot_height,
        "y_range": (0, y_max),
        "group_info": group_plot_info,
    }
    return plot_info


def _determine_plot_height(figure_height, y_max, n_params, n_groups):
    if figure_height is None:
        plot_height = int(max(min(y_max, 1000), 200))
    else:
        space_of_titles = n_groups * 50
        available_space = figure_height - space_of_titles
        plot_height = int(available_space / n_params)
        if plot_height < 50:
            warnings.warn(
                "\n\nThe height ({}) ".format(figure_height)
                + "you specified results in very small "
                + "plots which may not render well. \n"
                + "Adjust the figure height to a larger value "
                + "or set it to None to get a larger plot. \n"
                + "Alternatively, you can click on the Reset button "
                + "on the right of the plot and your plot should render correctly."
            )
    return plot_height
