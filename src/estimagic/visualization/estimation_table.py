import re
from collections import namedtuple
from copy import deepcopy
from functools import partial
from warnings import warn

import numpy as np
import pandas as pd


def estimation_table(
    models,
    *,
    return_type="data_frame",
    render_options=None,
    custom_param_names=None,
    show_col_names=True,
    custom_col_names=None,
    endog_names_as_col_names=False,
    endog_names_as_col_level=False,
    custom_endog_names=None,
    custom_model_names=None,
    custom_index_names=None,
    show_inference=True,
    confidence_intervals=False,
    show_stars=True,
    significance_levels=(0.1, 0.05, 0.01),
    number_format=("{0:.3g}", "{0:.5f}", "{0:.4g}"),
    add_trailing_zeros=True,
    padding=1,
    show_footer=True,
    stat_keys=None,
    append_notes=True,
    notes_label="Note:",
    custom_notes=None,
    siunitx_warning=True,
    alignment_warning=True,
):
    r"""Generate html and LaTex tables provided (lists of) of models.

    Can return strings of LaTex/html scripts or dictionaries with processed dataframes
    to be passed to tabular functions, or save tables to path.

    Args:
        models (list): list of estimation results. The estimation results should either
            have attributes info(dict) with summary statistics of the model and params
            (DataFrame) with parameter values, standard erors and/or confidence
            intervals and p values, or a sm regression result can be passed.
        return_type (str): Can be "latex", "html", "python" or a file path with the
            extension .tex or .html. If "python", a dictionary with the entries
            "paramaters_df", "footer_df" and "footer (html and latex)" is returned.
        custom_param_names (dict): a dictionary with old names of parameters that should
            be renamed as keys and respective new names as values. Default is None.
        render_options (dict): a dictionary with keyword arguments that are passed to
            df.to_latex or df.to_html, depending on the return_type.
            The default is None.
        show_col_names (bool): a boolean variable for printing column numbers.
            Default is True
        custom_col_names (list): a list of strings to print as column names.
            Default is None.
        endog_names_as_col_names (bool): if True, use the name of endogenous variables
            as column names. Default is False.
        endog_names_as_col_level (bool): if True, use the name of endogenous variables
            as additional level of column index.
        custom_endog_names (dict): a dictionary with old names of endogenous variabels
            that should be renamed as keys and respective new names as values.
        custom_model_names (dictionary): a dictionary with keys to print as model names,
            and values as columns to be combined under the respective model names.
            Default is None.
        custom_index_names (list): a list of strings to print as the name of the
            parameter/variable column. To print index names, add index_names = True
            in the render options. Default is None.
        show_inference (bool): a boolean variable for printing precision (standard
            error/confidence intervals). Defalut is True.
        show_stars (bool): a boolean variable for printing significance stars.
            Default is True.
        significance_levels (list): a list of floats for p value's significance cutt-off
            values. Default is [0.1,0.05,0.01].
        number_format (int): an integer for the number of digits to the right of the
            decimal point to round to. Default is 2.
        padding (int): an integer used for aligning LaTex columns. Affects the
            alignment of the columns to the left of the decimal point of numerical
            entries. Default is 1. If the number of models is more than 2, set the
            value of padding to 3 or more to avoid columns overlay in the tex output.

        confidence_intervals (bool): a boolean variable for printin confidence
            intervals or standard errors as precision. If False standard errors
            are printed. Default is False.
        show_footer (bool): a boolean variable for printing statistics, e.g. R2,
            Obs numbers. Default is True.
        stat_keys (dict): a dictionary with printed statistics names as keys,
            and statistics statistics names to be retrieved from model.info as values.
            Default is dictionary with common statistics of stats model linear
            regression.
        append_notes (bool): a boolean variable for printing p value cutoff explanation
            and additional notes, if applicable. Default is True.
        notes_label (str): a sting to print as the title of the notes section, if
            applicable. Default is 'Notes'
        custom_notes (list): a list of strings for additional notes. Default is None.

    Returns:
        res_table (data frame, str or dictionary): depending on the rerturn type,
            data frame with formatted strings, a string for html or latex tables,
            or a dictionary with statistics and parameters dataframes, and strings
            for footers is returned. If the return type is a path, the function saves
            the resulting table at the given path.

    Notes:
        - Compiling LaTex tables requires the package siunitx.
        - Add \sisetup{input-symbols = ()} to your main tex file for proper
            compilation

    """
    assert isinstance(models, list), "Please, provide models as a list"
    models = [_process_model(mod) for mod in models]
    # if the value of custom_col_names is the default and EVERY models' attribute
    # info has the key estimation_name, replace custom col names with the  value
    # of this key.
    endog_names = {
        model.info.get("dependent_variable", ""): model.info.get(
            "dependent_variable", ""
        )
        for model in models
    }
    if custom_endog_names:
        endog_names = endog_names.update(custom_endog_names)
    endog_names = endog_names.values()
    if not custom_col_names:
        if not endog_names_as_col_names:
            name_list = [model.info.get("estimation_name", "") for model in models]
        else:
            name_list = endog_names
        if "" not in name_list:
            custom_col_names = name_list
    if not custom_model_names:
        if endog_names_as_col_level:
            custom_model_names = {
                dep_var: [col_num] for col_num, dep_var in enumerate(endog_names)
            }
    # Set some defaults:
    if not stat_keys:
        stat_keys = {
            "Observations": "n_obs",
            "R$^2$": "rsquared",
            "Adj. R$^2$": "rsquared_adj",
            "Residual Std. Error": "resid_std_err",
            "F Statistic": "fvalue",
            "show_dof": None,
        }

    for_index = [mod.params for mod in models]
    com_ind = []
    for d_ in for_index:
        com_ind += [ind for ind in d_.index.to_list() if ind not in com_ind]
    format_cols = ["value"]
    if show_inference:
        if confidence_intervals:
            format_cols += ["ci_lower", "ci_upper"]
        else:
            format_cols.append("standard_error")
    df_list = [mod.params.reindex(com_ind)[format_cols] for mod in models]
    raw_formatted = [_apply_number_format(df, number_format) for df in df_list]
    max_trail = int(max([_get_digits_after_decimal(df) for df in raw_formatted]))
    if add_trailing_zeros:
        formatted = [_apply_number_format(df, max_trail) for df in raw_formatted]
    else:
        formatted = raw_formatted
    to_convert = []
    if show_stars:
        for df, mod in zip(formatted, models):
            to_convert.append(
                pd.concat([df, mod.params.reindex(com_ind)["p_value"]], axis=1)
            )
    else:
        to_convert = formatted
    to_concat = [
        _convert_model_to_series(
            df,
            significance_levels,
            show_stars,
        )
        for df in to_convert
    ]
    body_df = pd.concat(to_concat, axis=1)
    body_df = _process_body_df(
        body_df,
        custom_param_names,
        custom_index_names,
        show_col_names,
        custom_col_names,
        custom_model_names,
    )
    to_concat = [
        _create_statistics_sr(
            mod,
            stat_keys,
            significance_levels,
            show_stars,
            number_format,
            add_trailing_zeros,
            max_trail,
        )
        for mod in models
    ]
    footer_df = pd.concat(to_concat, axis=1)
    footer_df.columns = body_df.columns
    if str(return_type).endswith("tex"):
        if siunitx_warning:
            warn(
                r"""LaTeX compilation requires the package siunitx and adding
                    \sisetup{input-symbols =()} to your main tex file. To turn
                    this warning off set value of siunitx_warning = False"""
            )
        if len(models) > 2:
            if alignment_warning:
                warn(
                    """Set the value of padding to 3 or higher to avoid overlay
                        of columns. To turn this warning off set value of
                        alignment_warning = False"""
                )
        notes_tex = _generate_notes_latex(
            append_notes, notes_label, significance_levels, custom_notes, body_df
        )
        out = render_latex(
            body_df,
            footer_df,
            max_trail,
            notes_tex,
            render_options,
            custom_index_names,
            custom_model_names,
            padding,
            show_footer,
        )
    elif str(return_type).endswith("html"):
        footer = _generate_notes_html(
            append_notes, notes_label, significance_levels, custom_notes, body_df
        )
        out = render_html(
            body_df, footer_df, footer, render_options, custom_index_names, show_footer
        )
    elif return_type == "render_inputs":
        out = {
            "body_df": body_df,
            "footer_df": footer_df,
            "notes_tex": _generate_notes_latex(
                append_notes, notes_label, significance_levels, custom_notes, body_df
            ),
            "latex_right_alig": max_trail,
            "notes_html": _generate_notes_html(
                append_notes, notes_label, significance_levels, custom_notes, body_df
            ),
        }
    elif return_type == "data_frame":
        if show_footer:
            footer_df.index.names = body_df.index.names
            out = pd.concat([body_df.reset_index(), footer_df.reset_index()]).set_index(
                body_df.index.names
            )
        else:
            out = body_df
    else:
        raise TypeError("Invalid return type")
    if str(return_type).endswith((".html", ".tex")):
        with open(return_type, "w") as t:
            t.write(out)

    return out


def render_latex(
    body_df,
    footer_df,
    right_align,
    notes_tex,
    render_options=None,
    custom_index_names=None,
    custom_model_names=None,
    padding=1,
    show_footer=True,
):
    """Return estimation table in LaTeX format as string.

    Args:
        body_df (pandas.DataFrame): the processed dataframe with parameter values and
            precision (if applied) as strings.
        footer_df (pandas.DataFrame): the processed dataframe with summary statistics as
            strings.
        notes_tex (str): a string with LaTex code for the notes section
        render_options(dict): the pd.to_latex() kwargs to apply if default options
            need to be updated.
        lef_decimals (int): see main docstring
        number_format (int): see main docstring
        show_footer (bool): see main docstring

    Returns:
        latex_str (str): the string for LaTex table script.

    """
    body_df = body_df.copy(deep=True)
    body_df = body_df.applymap(_add_latex_syntax_around_scientfic_number_string)
    n_levels = body_df.index.nlevels
    n_columns = len(body_df.columns)
    # here you add all arguments of df.to_latex for which you want to change the default
    default_options = {
        "index_names": False,
        "escape": False,
        "na_rep": "",
        "column_format": "l" * n_levels
        + "S[table-format ={}.{},table-space-text-post={{-**}}]".format(
            padding, right_align
        )
        * n_columns,
        "multicolumn_format": "c",
    }
    if custom_index_names:
        default_options.update({"index_names": True})
    if render_options:
        default_options.update(render_options)
    if not default_options["index_names"]:
        body_df.index.names = [None] * body_df.index.nlevels
    latex_str = body_df.to_latex(**default_options)
    if custom_model_names:
        temp_str = "\n"
        for k in custom_model_names:
            max_col = max(custom_model_names[k]) + n_levels + 1
            min_col = min(custom_model_names[k]) + n_levels + 1
            temp_str += f"\\cmidrule(lr){{{min_col}-{max_col}}}"
            temp_str += "\n"
        latex_str = (
            latex_str.split("\\\\", 1)[0]
            + "\\\\"
            + temp_str
            + latex_str.split("\\\\", 1)[1]
        )
    latex_str = latex_str.split("\\bottomrule")[0]
    if show_footer:
        stats_str = footer_df.to_latex(**default_options)
        stats_str = (
            "\\midrule" + stats_str.split("\\midrule")[1].split("\\bottomrule")[0]
        )
        latex_str += stats_str
    latex_str += notes_tex
    latex_str += "\\bottomrule\n\\end{tabular}\n"
    if latex_str.startswith("\\begin{table}"):
        latex_str += "\n\\end{table}\n"
    return latex_str


def render_html(
    body_df, footer_df, notes_html, render_options, custom_index_names, show_footer
):
    """Return estimation table in html format as string.

    Args:
        body_df (DataFrame): the processed dataframe with parameter values and
            precision (if applied) as strings.
        footer_df (DataFrame): the processed dataframe with summary statistics
            as strings.
        notes_html (str): a string with html code for the notes section
        render_options(dict): the pd.to_html() kwargs to apply if default options
            need to be updated.
        show_footer (bool): see main docstring

    Returns:
        html_str (str): the string for html table script.

    """
    n_levels = body_df.index.nlevels
    n_columns = len(body_df.columns)
    default_options = {"index_names": False, "na_rep": "", "justify": "center"}
    if custom_index_names:
        default_options.update({"index_names": True})
    html_str = ""
    if render_options:
        default_options.update(render_options)
        if "caption" in default_options:
            html_str += default_options["caption"] + "<br>"
            default_options.pop("caption")
    html_str += body_df.to_html(**default_options).split("</tbody>\n</table>")[0]
    # this line removes all the curly braces that were placed in order to get nice latex
    # output. Since Html does not escape them, they need to be removed.
    html_str = re.sub(
        r"(?<=[\d)}{)a-zA-Z])}", "", re.sub(r"{(?=[}\d(a-zA-Z-])", "", html_str)
    ).replace(r"\,", " ")
    if show_footer:
        stats_str = """<tr><td colspan="{}" style="border-bottom: 1px solid black">
            </td></tr>""".format(
            n_levels + n_columns
        )
        stats_str += (
            footer_df.to_html(**default_options)
            .split("</thead>\n")[1]
            .split("</tbody>\n</table>")[0]
        )
        stats_str = re.sub(r"(?<=[\d)}{)])}", "", re.sub(r"{(?=[}\d(])", "", stats_str))
        html_str += stats_str
    html_str += notes_html
    html_str += "</tbody>\n</table>"
    return html_str


def _process_model(model):
    """Check model validity, convert to namedtuple."""
    NamedTup = namedtuple("NamedTup", "params info")
    if hasattr(model, "params") and hasattr(model, "info"):
        assert isinstance(model.info, dict)
        assert isinstance(model.params, pd.DataFrame)
        info_dict = model.info
        params_df = model.params.copy(deep=True)
    else:
        if isinstance(model, dict):
            params_df = model["params"].copy(deep=True)
            info_dict = model.get("info", {})
        elif isinstance(model, pd.DataFrame):
            params_df = model.copy(deep=True)
            info_dict = {}
        else:
            try:
                params_df = _extract_params_from_sm(model)
                info_dict = {**_extract_info_from_sm(model)}
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException:
                raise TypeError("Model {} does not have valid format".format(model))
    if "pvalue" in params_df.columns:
        params_df = params_df.rename(columns={"pvalue": "p_value"})
    processed_model = NamedTup(params=params_df, info=info_dict)
    return processed_model


def _convert_model_to_series(
    df,
    significance_levels,
    show_stars,
):
    """Return processed value series with significance stars and inference information.

    Args:

        df (DataFrame): params DataFrame of the model
        significance_levels (list): see main docstring
        number_format (int): see main docstring
        show_inference (bool): see main docstring
        confidence_intervals (bool): see main docstring
        show_stars (bool): see main docstring

    Returns:
        sr (pd.Series): string series with values and inferences.
    """
    value_sr = df["value"]
    if show_stars:
        sig_bins = [-1] + sorted(significance_levels) + [2]
        value_sr += "$^{"
        value_sr += (
            pd.cut(
                df["p_value"],
                bins=sig_bins,
                labels=[
                    "*" * (len(significance_levels) - i)
                    for i in range(len(significance_levels) + 1)
                ],
            )
            .astype("str")
            .replace("nan", "")
            .replace(np.nan, "")
        )
        value_sr += " }$"
    if "ci_lower" in df:
        ci_lower = df["ci_lower"]
        ci_upper = df["ci_upper"]
        inference_sr = "{("
        inference_sr += ci_lower
        inference_sr += r"\,;\,"
        inference_sr += ci_upper
        inference_sr += ")}"
        sr = _combine_series(value_sr, inference_sr)
    elif "standard_error" in df:
        standard_error = df["standard_error"]
        inference_sr = "(" + standard_error + ")"
        sr = _combine_series(value_sr, inference_sr)
    else:
        sr = value_sr
    # replace empty braces with empty string
    sr = sr.where(sr.apply(lambda x: bool(re.search(r"\d", x))), "")
    sr.name = ""
    return sr


def _combine_series(value_sr, inference_sr):
    """Merge value and inference series. Return string series
    with parameter values and precision values below respective param
    values.

    Args:
        values_sr (Series): string series of estimated parameter values
        inference_sr (Series): string series of inference values
    Returns:
        series: combined string series of param and inference values.


    """

    value_df = value_sr.to_frame(name="")
    original_cols = value_df.columns
    value_df.reset_index(drop=False, inplace=True)
    index_names = [item for item in value_df.columns if item not in original_cols]
    # set the index to even numbers, starting at 0
    value_df.index = value_df.index * 2
    inference_df = inference_sr.to_frame(name="")
    inference_df.reset_index(drop=False, inplace=True)
    # set the index to odd numbers, starting at 1
    inference_df.index = (inference_df.index * 2) + 1
    inference_df[index_names[-1]] = ""
    df = pd.concat([value_df, inference_df]).sort_index()
    df.set_index(index_names, inplace=True, drop=True)
    return df[""]


def _create_statistics_sr(
    model,
    stat_keys,
    significance_levels,
    show_stars,
    number_format,
    add_trailing_zeros,
    max_trail,
):
    """Process statistics values, return string series.

    Args:
        model (estimation result): see main docstring
        stat_keys (dict): see main docstring
        significance_levels (list): see main docstring
        show_stars (bool): see main docstring
        number_format (int): see main focstring

    Returns:
        series: string series with summary statistics values and additional info
            if applicable.

    """
    stat_values = {}
    stat_keys = deepcopy(stat_keys)
    if "show_dof" in stat_keys:
        show_dof = stat_keys.pop("show_dof")
    else:
        show_dof = None
    for k in stat_keys:
        if not stat_keys[k] == "n_obs":
            stat_values[k] = model.info.get(stat_keys[k], np.nan)
    raw_formatted = _apply_number_format(
        pd.DataFrame(pd.Series(stat_values)), number_format
    )
    if add_trailing_zeros:
        formatted = _apply_number_format(raw_formatted, max_trail)
    else:
        formatted = raw_formatted
    stat_values = formatted.to_dict()[0]
    if "n_obs" in stat_keys.values():
        n_obs = model.info.get("n_obs", np.nan)
        if not np.isnan(n_obs):
            n_obs = int(n_obs)
        stat_values[
            list(stat_keys.keys())[list(stat_keys.values()).index("n_obs")]
        ] = n_obs

    if "fvalue" in model.info and "F Statistic" in stat_values:
        if show_stars and "f_pvalue" in model.info:
            sig_bins = [-1] + sorted(significance_levels) + [2]
            sig_icon_fstat = "*" * (
                len(significance_levels)
                - np.digitize(model.info["f_pvalue"], sig_bins)
                + 1
            )
            stat_values["F Statistic"] = (
                stat_values["F Statistic"] + "$^{" + sig_icon_fstat + "}$"
            )
        if show_dof:
            fstat_str = "{{{}(df={};{})}}"
            stat_values["F Statistic"] = fstat_str.format(
                stat_values["F Statistic"],
                int(model.info["df_model"]),
                int(model.info["df_resid"]),
            )
    if "resid_std_err" in model.info and "Residual Std. Error" in stat_values:
        if show_dof:
            rse_str = "{{{}(df={})}}"
            stat_values["Residual Std. Error"] = rse_str.format(
                stat_values["Residual Std. Error"], int(model.info["df_resid"])
            )
    stat_sr = pd.Series(stat_values)
    # the follwing is to make sure statistics dataframe has as many levels of
    # indices as the parameters dataframe.
    stat_ind = np.empty((len(stat_sr), model.params.index.nlevels - 1), dtype=str)
    stat_ind = np.concatenate(
        [stat_sr.index.values.reshape(len(stat_sr), 1), stat_ind], axis=1
    ).T
    stat_sr.index = pd.MultiIndex.from_arrays(stat_ind)
    return stat_sr.astype("str").replace("nan", "")


def _process_body_df(
    df,
    custom_param_names,
    custom_index_names,
    show_col_names,
    custom_col_names,
    custom_model_names,
):
    """Process body DataFrame, customize the header.

    Args:
        df (DataFrame): string DataFrame with parameter values and inferences.
        custom_param_names (dict): see main docstring
        custom_index_names (list): see main docstring
        show_col_names (bool): see main docstring
        custom_col_names (list): see main docstring
        custom_model_names (dict): see main docstring

    Returns:
        processed_df (DataFrame): string DataFrame with customized header.

    """

    if custom_index_names:
        df.index.names = custom_index_names
    if custom_param_names:
        ind = df.index.to_frame()
        ind = ind.replace(custom_param_names)
        df.index = pd.MultiIndex.from_frame(ind)
    if show_col_names:
        if custom_col_names:
            df.columns = ["{" + cn + "}" for cn in custom_col_names]
        else:
            df.columns = ["{(" + str(col + 1) + ")}" for col in range(len(df.columns))]
    if custom_model_names:
        assert isinstance(
            custom_model_names, dict
        ), """Please provide a dictionary with model names as keys and lists of
            respective column numbers as values"""
        for val in custom_model_names.values():
            assert isinstance(
                val, list
            ), """Provide list of integers for columns to be combined under a common
                model name"""
            if len(val) > 1:
                assert all(
                    i == j - 1 for i, j in zip(val, val[1:])
                ), "Under common model name you can combine only adjacent columns"
        cols = df.columns
        custom_model_names = {
            k: v
            for k, v in sorted(custom_model_names.items(), key=lambda item: item[1])
        }
        flev = ["{}"] * len(cols)
        for k in custom_model_names:
            for v in custom_model_names[k]:
                flev[v] = "{" + k + "}"
        df.columns = pd.MultiIndex.from_tuples(list(zip(flev, cols)))
    return df


def _generate_notes_latex(
    append_notes, notes_label, significance_levels, custom_notes, df
):
    """Generate the LaTex script of the notes section.

    Args:
        append_notes (bool): see main docstring
        notes_label (str): see main docstring
        significance_levels (list): see main docstring
        custom_notes (str): see main docstring
        df (DataFrame): params DataFrame of estimation model

    Returns:
        notes_latex (str): a string with LaTex script

    """
    n_levels = df.index.nlevels
    n_columns = len(df.columns)
    significance_levels = sorted(significance_levels)
    notes_text = "\\midrule\n"
    if append_notes:
        notes_text += "\\textit{{{}}} & \\multicolumn{{{}}}{{r}}{{".format(
            notes_label, str(n_columns + n_levels - 1)
        )
        # iterate over penultimate sig_level since last item of legend is not
        # followed by a semi column
        for i in range(len(significance_levels) - 1):
            star = "*" * (len(significance_levels) - i)
            notes_text += "$^{{{}}}$p$<${};".format(star, str(significance_levels[i]))
        notes_text += "$^{*}$p$<$" + str(significance_levels[-1]) + "} \\\\\n"
        if custom_notes:
            amp_n = "&" * n_levels
            if isinstance(custom_notes, list):
                assert all(
                    isinstance(n, str) for n in custom_notes
                ), "Data type of custom notes can only be string"
                for n in custom_notes:
                    notes_text += """
                    {}\\multicolumn{{{}}}{{r}}\\textit{{{}}}\\\\\n""".format(
                        amp_n, n_columns, n
                    )
            elif isinstance(custom_notes, str):
                notes_text += "{}\\multicolumn{{{}}}{{r}}\\textit{{{}}}\\\\\n".format(
                    amp_n, n_columns, custom_notes
                )
            else:
                raise ValueError(
                    "Custom notes can be either a string or a list of strings"
                )
    return notes_text


def _generate_notes_html(
    append_notes, notes_label, significance_levels, custom_notes, df
):
    """Generate the html script of the notes section of the estimation table.

    Args:
        append_notes (bool): see main docstring
        notes_label (str): see main docstring
        significance_levels (list): see main docstring
        custom_notes (str): see main docstring
        df (DataFrame): params DataFrame of estimation model

    Returns:
        notes_latex (str): a string with html script

    """
    n_levels = df.index.nlevels
    n_columns = len(df.columns)
    significance_levels = sorted(significance_levels)
    notes_text = """<tr><td colspan="{}" style="border-bottom: 1px solid black">
        </td></tr>""".format(
        n_columns + n_levels
    )
    if append_notes:
        notes_text += """
        <tr><td style="text-align: left">{}</td><td colspan="{}"
        style="text-align: right">""".format(
            notes_label, n_columns + n_levels - 1
        )
        for i in range(len(significance_levels) - 1):
            stars = "*" * (len(significance_levels) - i)
            notes_text += "<sup>{}</sup>p&lt;{}; ".format(stars, significance_levels[i])
        notes_text += """<sup>*</sup>p&lt;{} </td>""".format(significance_levels[-1])
        if custom_notes:
            if isinstance(custom_notes, list):
                assert all(
                    isinstance(n, str) for n in custom_notes
                ), "Data type of custom notes can only be string"
                notes_text += """
                    <tr><td></td><td colspan="{}"style="text-align: right">{}</td></tr>
                    """.format(
                    n_columns + n_levels - 1, custom_notes[0]
                )
                if len(custom_notes) > 1:
                    for i in range(1, len(custom_notes)):
                        notes_text += """
                        <tr><td></td><td colspan="{}"style="text-align: right">
                        {}</td></tr>
                        """.format(
                            n_columns + n_levels - 1, custom_notes[i]
                        )
            elif isinstance(custom_notes, str):
                notes_text += """
                    <tr><td></td><td colspan="{}"style="text-align: right">{}</td></tr>
                    """.format(
                    n_columns + n_levels - 1, custom_notes
                )
            else:
                raise ValueError(
                    "Custom notes can be either a string or a list of strings"
                )

    return notes_text


def _extract_params_from_sm(model):
    """Convert statsmodels like estimation result to estimagic like params dataframe."""
    to_concat = []
    params_list = ["params", "pvalues", "bse"]
    for col in params_list:
        to_concat.append(getattr(model, col))
    to_concat.append(model.conf_int())
    params_df = pd.concat(to_concat, axis=1)
    params_df.columns = ["value", "p_value", "standard_error", "ci_lower", "ci_upper"]
    return params_df


def _extract_info_from_sm(model):
    """Process statsmodels estimation result to retrieve summary statistics as dict."""
    info = {}
    key_values = [
        "rsquared",
        "rsquared_adj",
        "fvalue",
        "f_pvalue",
        "df_model",
        "df_resid",
    ]
    for kv in key_values:
        info[kv] = getattr(model, kv)
    info["dependent_variable"] = model.model.endog_names
    info["resid_std_err"] = np.sqrt(model.scale)
    info["n_obs"] = model.df_model + model.df_resid + 1
    return info


def _apply_number_format(df, number_format):
    processed_format = _process_number_format(number_format)
    if isinstance(processed_format, (list, tuple)):
        df_formatted = df.copy(deep=True).astype("float")
        for formatter in processed_format[:-1]:
            df_formatted = df_formatted.applymap(formatter.format).astype("float")
        df_formatted = df_formatted.astype("float").applymap(
            processed_format[-1].format
        )
    elif isinstance(processed_format, str):
        df_formatted = df.applymap(
            partial(_format_non_scientific_numbers, format_string=processed_format)
        )
    elif callable(processed_format):
        df_formatted = df.applymap(processed_format)
    return df_formatted


def _format_non_scientific_numbers(number_string, format_string):
    if "e" in number_string:
        out = number_string
    else:
        out = format_string.format(float(number_string))
    return out


def _process_number_format(raw_format):
    if isinstance(raw_format, str):
        processed_format = [raw_format]
    elif isinstance(raw_format, int):
        processed_format = f"{{0:.{raw_format}f}}"
    elif callable(raw_format) or isinstance(raw_format, (list, tuple)):
        processed_format = raw_format
    else:
        raise TypeError("Invalid number format")
    return processed_format


def _get_digits_after_decimal(df):
    max_trail = 0
    for c in df.columns:
        try:
            trail_length = (
                (
                    df[c][~df[c].astype("str").str.contains("e")]
                    .astype("str")
                    .str.split(".", expand=True)[1]
                    .astype("str")
                    .replace("None", "")
                )
                .str.len()
                .max()
            )
        except KeyError:
            trail_length = 0
        if trail_length > max_trail:
            max_trail = trail_length
    return max_trail


def _add_latex_syntax_around_scientfic_number_string(string):
    if "e" not in string:
        out = string
    else:
        prefix, *num_parts, suffix = re.split(r"([+-.\d+])", string)
        number = "".join(num_parts)
        out = f"{prefix}\\num{{{number}}}{suffix}"
    return out
