"""
Process a list of estimagic optimization results for drawing a comparison plot.

"""
import numpy as np
import pandas as pd


MEDIUMELECTRICBLUE = "#035096"


def generate_comp_plot_inputs(results, x_padding, num_bins, color_dict):
    """Generate the inputs for the comparison plot function.

    Args:
        results (list): List of estimagic optimization results where the info
            has been extended with 'model' and 'model_name'
        x_padding (float): the x_range is extended on each side by x_padding
            times the range of the data
        num_bins (int): number of bins
        color_dict (dict): mapping from the model class names to colors.

    Returns:
        source_dfs (dict): map from parameter identifiers to DataFrames
            with everything we need for the comparison plot
        x_min (Series): The index are the parameter groups. The values are
            the left bound of the x-axis for this parameter group
        x_max (Series): Same as x_min but for right bound
        bins (DataFrame): The index are the parameter groups. Each row contains
            the edges of the bins for that group.
        rect_widths (Series): The index are the parameter groups. The value are
            the bin width for that parameter group.
    """
    parameter_groups = _consolidate_parameter_attribute(results, "group")
    parameter_names = _consolidate_parameter_attribute(results, "name")
    all_data = _combine_params_data(
        results, parameter_groups, parameter_names, color_dict
    )
    x_min, x_max = _calculate_x_bounds(all_data, x_padding)
    bins, rect_width = _calculate_bins_and_rectangle_width(x_min, x_max, num_bins)
    source_dfs = {group: {} for group in parameter_groups.unique()}
    y_max = 5
    for param in parameter_groups.index:
        group = parameter_groups.loc[param]
        sdf = all_data.loc[param]
        sdf = sdf.set_index("model", drop=True)
        sdf["binned_x"] = _replace_by_bin_midpoint(sdf["value"], bins.loc[group])
        sdf["dodge"] = _calculate_dodge(sdf["value"], bins.loc[group])
        sdf["dodge"] = sdf["dodge"].where(sdf["value"].notnull(), -10)
        source_dfs[group][param] = sdf
        y_max = int(max(y_max, sdf["dodge"].max() + 1))
    return source_dfs, x_min, x_max, rect_width, y_max


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
    to_concat = []
    for res in results:
        params = res.params[res.params.columns & relevant].copy()
        params["model"] = res.info["model_name"]
        params["model_class"] = res.info["model_class"]
        params["color"] = color_dict.get(res.info["model_class"], MEDIUMELECTRICBLUE)
        to_concat.append(params)
    df = pd.concat(to_concat)
    for attr in [parameter_groups, parameter_names]:
        df = pd.merge(df, attr, left_index=True, right_index=True)
    df["group"].replace({None: np.nan}, inplace=True)
    df.dropna(subset=["group"], inplace=True)
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
    return x_min, x_max


def _calculate_bins_and_rectangle_width(x_min, x_max, num_bins):
    bins_transposed, stepsize = np.linspace(
        start=x_min, stop=x_max, num=num_bins + 1, retstep=True
    )
    bins = pd.DataFrame(data=bins_transposed.T, index=x_min.index)
    rectangle_width = pd.Series(data=stepsize, index=x_min.index)
    return bins, rectangle_width


def _replace_by_bin_midpoint(values, bins):
    midpoints = (bins + bins.shift(periods=-1))[:-1] / 2
    return pd.cut(values, bins, labels=midpoints).astype(float)


def _calculate_dodge(values, bins):
    df = values.to_frame()
    df["bin"] = pd.cut(values, bins, labels=range(len(bins) - 1))
    dodge = 0.5 + df.groupby("bin").cumcount()
    return dodge
