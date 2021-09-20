import re
from collections import namedtuple
from copy import copy
from warnings import warn

import numpy as np
import pandas as pd


def estimation_table(
    models,
    return_type,
    render_options=None,
    custom_param_names=None,
    show_col_names=True,
    custom_col_names=None,
    custom_model_names=None,
    custom_index_names=None,
    show_inference=True,
    confidence_intervals=False,
    show_stars=True,
    sig_levels=(0.1, 0.05, 0.01),
    sig_digits=2,
    left_decimals=1,
    show_footer=True,
    stats_dict=None,
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
        render_options (dict): A dictionary with keyword arguments that are passed to
            df.to_latex or df.to_html, depending on the return_type.
            The default is None.
        show_col_names (bool): a boolean variable for printing column numbers.
            Default is True
        custom_col_names (list): a list of strings to print as column names.
            Default is None.
        custom_model_names (list): a list of strings to print as model names,
            possibly combining columns under common model names. Default is None.
        custom_index_names (list): a list of strings to print as the name of the
            parameter/variable column. To print index names, add index_names = True
            in the render options. Default is None.
        show_inference (bool): a boolean variable for printing precision (standard
            error/confidence intervals). Defalut is True.
        show_stars (bool): a boolean variable for printing significance stars.
            Default is True.
        sig_levels (list): a list of floats for p value's significance cutt-off values.
            Default is [0.1,0.05,0.01].
        sig_digits (int): an integer for the number of digits to the right of the
            decimal point to round to. Default is 2.
        left_decimals (int): an integer used for aligning LaTex columns. Affects the
            alignment of the columns to the left of the decimal point of numerical
            entries. Default is 1.
        confidence_intervals (bool): a boolean variable for printin confidence
            intervals or standard errors as precision. If False standard errors
            are printed. Default is False.
        show_footer (bool): a boolean variable for printing statistics, e.g. R2,
            Obs numbers. Default is True.
        stats_dict (dict): a dictionary with printed statistics names as keys,
            and statistics statistics names to be retrieved from model.info as values.
            Default is dictionary with common statistics of stats model linear
            regression.
        append_notes (bool): a boolean variable for printing p value cutoff explanation
            and additional notes, if applicable. Default is True.
        notes_label (str): a sting to print as the title of the notes section, if
            applicable. Default is 'Notes'
        custom_notes (list): a list of strings for additional notes. Default is None.

    Returns:
        res_table (str or dictionary): depending on the rerturn type, a string
            for html or latex tables, or a dictionary with atatistics and
            parameters dataframes, and strings for footers is returned. If the
            return type is a path, the function saves the resulting table at the
            given path.

    Notes:
        - Compiling LaTex tables requires the package siunitx.
        - Add \sisetup{input-symbols = ()} to your main tex file for proper
            compilation
        - If the number of models is more than 2, set the value of left_decimals
            to 3 or more to avoid columns overlay in the tex output.

    """
    assert isinstance(models, list), "Please, provide models as a list"
    models = [_process_model(mod) for mod in models]
    # if the value of custom_col_names is the default and EVERY models' attribute
    # info has the key estimation_name, replace custom col names with the  value
    # of this key.
    if not custom_col_names:
        name_list = []
        for model in models:
            name_list.append(model.info.get("estimation_name", ""))
        if "" not in name_list:
            custom_col_names = name_list
    # Set some defaults:
    if not stats_dict:
        stats_dict = {
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
    df_list = [mod.params.reindex(com_ind) for mod in models]
    to_concat = [
        _convert_model_to_series(
            df,
            sig_levels,
            sig_digits,
            show_inference,
            confidence_intervals,
            show_stars,
        )
        for df in df_list
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
        _create_statistics_sr(mod, stats_dict, sig_levels, show_stars, sig_digits)
        for mod in models
    ]
    footer_df = pd.concat(to_concat, axis=1)
    footer_df.columns = body_df.columns
    if return_type == "latex" or str(return_type).endswith(".tex"):
        if siunitx_warning:
            warn(
                r"""LaTeX compilation requires the package siunitx and adding
                    \sisetup{input_symbols =()} to your main tex file. To turn
                    this warning off set value of siunitx_warning = False"""
            )
        if len(models) > 2:
            if alignment_warning:
                warn(
                    """Set the value of left_decimals to 3 or higher to avoid overlay
                        of columns. To turn this warning off set value of
                        alignment_warning = False"""
                )
        notes_tex = _generate_notes_latex(
            append_notes, notes_label, sig_levels, custom_notes, body_df
        )
        res_table = tabular_tex(
            body_df,
            footer_df,
            notes_tex,
            render_options,
            custom_index_names,
            custom_model_names,
            left_decimals,
            sig_digits,
            show_footer,
        )
    elif return_type == "html" or str(return_type).endswith(".html"):
        footer = _generate_notes_html(
            append_notes, notes_label, sig_levels, custom_notes, body_df
        )
        res_table = tabular_html(
            body_df, footer_df, footer, render_options, custom_index_names, show_footer
        )
    else:
        res_table = {
            "body_df": body_df,
            "footer_df": footer_df,
            "notes_tex": _generate_notes_latex(
                append_notes, notes_label, sig_levels, custom_notes, body_df
            ),
            "notes_html": _generate_notes_html(
                append_notes, notes_label, sig_levels, custom_notes, body_df
            ),
        }
    if str(return_type).endswith((".html", ".tex")):
        with open(return_type, "w") as t:
            t.write(res_table)

    return res_table


def tabular_tex(
    body_df,
    footer_df,
    notes_tex,
    render_options,
    custom_index_names,
    custom_model_names,
    left_decimals,
    sig_digits,
    show_footer,
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
        sig_digits (int): see main docstring
        show_footer (bool): see main docstring

    Returns:
        latex_str (str): the string for LaTex table script.

    """
    n_levels = body_df.index.nlevels
    n_columns = len(body_df.columns)
    # here you add all arguments of df.to_latex for which you want to change the default
    default_options = {
        "index_names": False,
        "escape": False,
        "na_rep": "",
        "column_format": "l" * n_levels
        + "S[table-format ={}.{}]".format(left_decimals, sig_digits) * n_columns,
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


def tabular_html(
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
    sig_levels,
    sig_digits,
    show_inference,
    confidence_intervals,
    show_stars,
):
    """Return processed value series with significance stars and inference information.

    Args:

        df (DataFrame): params DataFrame of the model
        sig_levels (list): see main docstring
        sig_digits (int): see main docstring
        show_inference (bool): see main docstring
        confidence_intervals (bool): see main docstring
        show_stars (bool): see main docstring

    Returns:
        sr (pd.Series): string series with values and inferences.
    """

    if show_stars:
        sig_bins = [-1] + sorted(sig_levels) + [2]
        value_sr = round(df["value"], sig_digits).replace(np.nan, "").astype("str")
        value_sr += "$^{"
        value_sr += (
            pd.cut(
                df["p_value"],
                bins=sig_bins,
                labels=[
                    "*" * (len(sig_levels) - i) for i in range(len(sig_levels) + 1)
                ],
            )
            .astype("str")
            .replace("nan", "")
            .replace(np.nan, "")
        )
        value_sr += " }$"
    else:
        value_sr = round(df["value"], sig_digits).replace(np.nan, "").astype("str")

    if show_inference:
        if confidence_intervals:
            inference_sr = "{("
            inference_sr += (
                round(df["ci_lower"], sig_digits).replace(np.nan, "").astype("str")
            )
            inference_sr += r"\,;\,"
            inference_sr += (
                round(df["ci_upper"], sig_digits).replace(np.nan, "").astype("str")
            )
            inference_sr += ")}"
        else:
            inference_sr = (
                "("
                + round(df["standard_error"], sig_digits)
                .replace(np.nan, "")
                .astype("str")
                + ")"
            )

        # replace empty braces with empty string
        # combine the two into one series Done
        sr = _combine_series(value_sr, inference_sr)
    else:
        sr = value_sr
    sr[~sr.apply(lambda x: bool(re.search(r"\d", x)))] = ""
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


def _create_statistics_sr(model, stats_dict, sig_levels, show_stars, sig_digits):
    """Process statistics values, return string series.

    Args:
        model (estimation result): see main docstring
        stats_dict (dict): see main docstring
        sig_levels (list): see main docstring
        show_stars (bool): see main docstring
        sig_digits (int): see main focstring

    Returns:
        series: string series with summary statistics values and additional info
            if applied.

    """
    series_dict = {}
    stats_dict = copy(stats_dict)
    if "show_dof" in stats_dict:
        show_dof = stats_dict.pop("show_dof")
    else:
        show_dof = None
    for k in stats_dict:
        series_dict[k] = str(
            round(model.info.get(stats_dict[k], np.nan), sig_digits)
        ).replace("nan", "")
    if "fvalue" in model.info and "F Statistic" in series_dict:
        if show_stars and "f_pvalue" in model.info:
            sig_bins = [-1] + sorted(sig_levels) + [2]
            sig_icon_fstat = "*" * (
                len(sig_levels) - np.digitize(model.info["f_pvalue"], sig_bins) + 1
            )
            series_dict["F Statistic"] = (
                series_dict["F Statistic"] + "$^{" + sig_icon_fstat + "}$"
            )
        if show_dof:
            fstat_str = "{{{}(df={};{})}}"
            series_dict["F Statistic"] = fstat_str.format(
                series_dict["F Statistic"],
                model.info["df_model"],
                model.info["df_resid"],
            )
    if "resid_std_err" in model.info and "Residual Std. Error" in series_dict:
        if show_dof:
            rse_str = "{{{}(df={})}}"
            series_dict["Residual Std. Error"] = rse_str.format(
                series_dict["Residual Std. Error"], model.info["df_resid"]
            )
    stat_sr = pd.Series(series_dict)
    # the follwing is to make sure statistics dataframe has as many levels of
    # indices as the parameters dataframe.
    stat_ind = np.empty((len(stat_sr), model.params.index.nlevels - 1), dtype=str)
    stat_ind = np.concatenate(
        [stat_sr.index.values.reshape(len(stat_sr), 1), stat_ind], axis=1
    ).T
    stat_sr.index = pd.MultiIndex.from_arrays(stat_ind)
    return stat_sr


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
        for k in custom_param_names:
            ind = ind.replace(k, custom_param_names[k])
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


def _generate_notes_latex(append_notes, notes_label, sig_levels, custom_notes, df):
    """Generate the LaTex script of the notes section.

    Args:
        append_notes (bool): see main docstring
        notes_label (str): see main docstring
        sig_levels (list): see main docstring
        custom_notes (str): see main docstring
        df (DataFrame): params DataFrame of estimation model

    Returns:
        notes_latex (str): a string with LaTex script

    """
    n_levels = df.index.nlevels
    n_columns = len(df.columns)
    sig_levels = sorted(sig_levels)
    notes_text = "\\midrule\n"
    if append_notes:
        notes_text += "\\textit{{{}}} & \\multicolumn{{{}}}{{r}}{{".format(
            notes_label, str(n_columns + n_levels - 1)
        )
        # iterate over penultimate sig_level since last item of legend is not
        # followed by a semi column
        for i in range(len(sig_levels) - 1):
            star = "*" * (len(sig_levels) - i)
            notes_text += "$^{{{}}}$p$<${};".format(star, str(sig_levels[i]))
        notes_text += "$^{*}$p$<$" + str(sig_levels[-1]) + "} \\\\\n"
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


def _generate_notes_html(append_notes, notes_label, sig_levels, custom_notes, df):
    """Generate the html script of the notes section of the estimation table.

    Args:
        append_notes (bool): see main docstring
        notes_label (str): see main docstring
        sig_levels (list): see main docstring
        custom_notes (str): see main docstring
        df (DataFrame): params DataFrame of estimation model

    Returns:
        notes_latex (str): a string with html script

    """
    n_levels = df.index.nlevels
    n_columns = len(df.columns)
    sig_levels = sorted(sig_levels)
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
        for i in range(len(sig_levels) - 1):
            stars = "*" * (len(sig_levels) - i)
            notes_text += "<sup>{}</sup>p&lt;{}; ".format(stars, sig_levels[i])
        notes_text += """<sup>*</sup>p&lt;{} </td>""".format(sig_levels[-1])
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
                    <td></td><td colspan="{}"style="text-align: right">{}</td></tr>
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
