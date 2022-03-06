import io
import textwrap
from collections import namedtuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from estimagic.config import EXAMPLE_DIR
from estimagic.visualization.estimation_table import _apply_number_format
from estimagic.visualization.estimation_table import _convert_frame_to_string_series
from estimagic.visualization.estimation_table import _create_group_to_col_position
from estimagic.visualization.estimation_table import _create_statistics_sr
from estimagic.visualization.estimation_table import _customize_col_groups
from estimagic.visualization.estimation_table import (
    _get_default_column_names_and_groups,
)
from estimagic.visualization.estimation_table import _get_digits_after_decimal
from estimagic.visualization.estimation_table import _get_model_names
from estimagic.visualization.estimation_table import _process_frame_indices
from estimagic.visualization.estimation_table import _process_model
from estimagic.visualization.estimation_table import estimation_table
from estimagic.visualization.estimation_table import render_html
from estimagic.visualization.estimation_table import render_latex
from pandas.testing import assert_frame_equal as afe
from pandas.testing import assert_series_equal as ase

# test process_model for different model types
ProcessedModel = namedtuple("ProcessedModel", "params info name")

fix_path = EXAMPLE_DIR / "diabetes.csv"

df_ = pd.read_csv(fix_path, index_col=0)
est = sm.OLS(endog=df_["target"], exog=sm.add_constant(df_[df_.columns[0:4]])).fit()
est1 = sm.OLS(endog=df_["target"], exog=sm.add_constant(df_[df_.columns[0:5]])).fit()


def test_estimation_table():
    models = [est]
    res = estimation_table(models, return_type="render_inputs", append_notes=False)
    exp = {}
    body = """
        index,target
        const,152.00$^{*** }$
        ,(2.85)
        Age,37.20$^{ }$
        ,(64.10)
        Sex,-107.00$^{* }$
        ,(62.10)
        BMI,787.00$^{*** }$
        ,(65.40)
        ABP,417.00$^{*** }$
        ,(69.50)
    """
    exp["params"] = _read_csv_string(body).fillna("")
    exp["params"].set_index("index", inplace=True)
    footer_str = """
         ,target
        R$^2$,0.40
        Adj. R$^2$,0.40
        Residual Std. Error,60.00
        F Statistic,72.90$^{***}$
        Observations,442

    """
    exp["stats"] = _read_csv_string(footer_str).fillna("")
    exp["stats"].set_index(" ", inplace=True)
    exp["stats"].index.names = [None]
    exp["stats"].index = pd.MultiIndex.from_arrays([exp["stats"].index])
    exp["notes_tex"] = "\\midrule\n"
    exp[
        "notes_html"
    ] = """<tr><td colspan="2" style="border-bottom: 1px solid black">
        </td></tr>"""
    afe(exp["stats"], res["stats"])
    afe(exp["params"], res["params"], check_index_type=False)


def test_render_latex():
    models = [_process_model(mod) for mod in [est, est1]]
    render_inputs = estimation_table(models, return_type="render_inputs")
    out_render_latex = render_latex(**render_inputs, siunitx_warning=False)
    out_estimation_table = estimation_table(
        models, return_type="latex", siunitx_warning=False
    )
    assert out_render_latex == out_estimation_table


def test_render_html():
    models = [_process_model(mod) for mod in [est, est1]]
    render_inputs = estimation_table(models, return_type="render_inputs")
    out_render_latex = render_html(**render_inputs)
    out_estimation_table = estimation_table(models, return_type="html")
    assert out_render_latex == out_estimation_table


def test_process_model_namedtuple():
    # checks that process_model doesn't alter values
    df = pd.DataFrame(columns=["value", "p_value", "ci_lower", "ci_upper"])
    df["value"] = np.arange(10)
    df["p_value"] = np.arange(10)
    df["ci_lower"] = np.arange(10)
    df["ci_upper"] = np.arange(10)
    info = {"stat1": 0, "stat2": 0}
    name = "model_name"
    model = ProcessedModel(params=df, info=info, name=name)
    res = _process_model(model)
    afe(res.params, df)
    ase(pd.Series(res.info), pd.Series(info))
    assert name == res.name


def test_process_model_stats_model():
    params = pd.DataFrame(
        columns=["value", "p_value", "standard_error", "ci_lower", "ci_upper"],
        index=["const", "Age", "Sex", "BMI", "ABP"],
    )
    params["value"] = [152.133484, 37.241211, -106.577520, 787.179313, 416.673772]
    params["p_value"] = [
        2.048808e-193,
        5.616557e-01,
        8.695658e-02,
        5.345260e-29,
        4.245663e-09,
    ]
    params["standard_error"] = [2.852749, 64.117433, 62.125062, 65.424126, 69.494666]
    params["ci_lower"] = [146.526671, -88.775663, -228.678572, 658.594255, 280.088446]
    params["ci_upper"] = [157.740298, 163.258084, 15.523532, 915.764371, 553.259097]
    info = {}
    info["rsquared"] = 0.40026108237714
    info["rsquared_adj"] = 0.39477148130050055
    info["fvalue"] = 72.91259907398705
    info["f_pvalue"] = 2.700722880950139e-47
    info["df_model"] = 4.0
    info["df_resid"] = 437.0
    info["resid_std_err"] = 59.97560860753488
    info["n_obs"] = 442.0
    res = _process_model(est)
    afe(res.params, params)
    ase(pd.Series(res.info), pd.Series(info))
    assert res.name == "target"


def test_process_model_dict():
    df = pd.DataFrame(columns=["value", "p_value", "standard_error"])
    df["value"] = np.arange(10)
    df["p_value"] = np.arange(10)
    df["standard_error"] = np.arange(10)
    info = {"stat1": 0, "stat2": 0}
    mod = {}
    mod["params"] = df
    mod["info"] = info
    res = _process_model(mod)
    afe(res.params, mod["params"])
    ase(pd.Series(res.info), pd.Series(mod["info"]))


# test convert_model_to_series for different arguments
def test_convert_model_to_series_with_ci():
    df = pd.DataFrame(
        np.array(
            [[0.6, 2.3, 3.3], [0.11, 0.049, 0.009], [0.6, 2.3, 3.3], [1.2, 3.3, 4.33]]
        ).T,
        columns=["value", "p_value", "ci_lower", "ci_upper"],
        index=["a", "b", "c"],
    ).astype("str")
    df["p_value"] = df["p_value"].astype("float")
    significance_levels = [0.1, 0.05, 0.01]
    show_stars = True
    res = _convert_frame_to_string_series(df, significance_levels, show_stars)
    exp = pd.Series(
        [
            "0.6$^{ }$",
            r"{(0.6\,;\,1.2)}",
            "2.3$^{** }$",
            r"{(2.3\,;\,3.3)}",
            "3.3$^{*** }$",
            r"{(3.3\,;\,4.33)}",
        ],
        index=["a", "", "b", "", "c", ""],
        name="",
    )
    exp.index.name = "index"
    ase(exp, res)


def test_convert_model_to_series_with_se():
    df = pd.DataFrame(
        np.array([[0.6, 2.3, 3.3], [0.11, 0.049, 0.009], [0.6, 2.3, 3.3]]).T,
        columns=["value", "p_value", "standard_error"],
        index=["a", "b", "c"],
    ).astype("str")
    df["p_value"] = df["p_value"].astype("float")
    significance_levels = [0.1, 0.05, 0.01]
    show_stars = True
    res = _convert_frame_to_string_series(df, significance_levels, show_stars)
    exp = pd.Series(
        ["0.6$^{ }$", "(0.6)", "2.3$^{** }$", "(2.3)", "3.3$^{*** }$", "(3.3)"],
        index=["a", "", "b", "", "c", ""],
        name="",
    )
    exp.index.name = "index"
    ase(exp, res)


def test_convert_model_to_series_without_inference():
    df = pd.DataFrame(
        np.array([[0.6, 2.3, 3.3], [0.11, 0.049, 0.009]]).T,
        columns=["value", "p_value"],
        index=["a", "b", "c"],
    ).astype("str")
    df["p_value"] = df["p_value"].astype("float")
    significance_levels = [0.1, 0.05, 0.01]
    show_stars = True
    res = _convert_frame_to_string_series(df, significance_levels, show_stars)
    exp = pd.Series(
        ["0.6$^{ }$", "2.3$^{** }$", "3.3$^{*** }$"], index=["a", "b", "c"], name=""
    )
    ase(exp, res)


# test create stat series
def test_create_statistics_sr():
    df = pd.DataFrame(np.empty((10, 3)), columns=["a", "b", "c"])
    df.index = pd.MultiIndex.from_arrays(np.array([np.arange(10), np.arange(10)]))
    info = {"rsquared": 0.45, "n_obs": 400, "rsquared_adj": 0.0002}
    number_format = ("{0:.3g}", "{0:.5f}", "{0:.4g}")
    add_trailing_zeros = True
    sig_levels = [0.1, 0.2]
    show_stars = False
    model = ProcessedModel(params=df, info=info, name="target")
    stats_dict = {
        "Observations": "n_obs",
        "R2": "rsquared",
        "show_dof": False,
        "R2 Adj.": "rsquared_adj",
    }
    res = _create_statistics_sr(
        model,
        stats_dict,
        sig_levels,
        show_stars,
        number_format,
        add_trailing_zeros,
        max_trail=4,
    )
    exp = pd.Series(["0.4500", "0.0002", "400"])
    exp.index = pd.MultiIndex.from_arrays(
        np.array([np.array(["R2", "R2 Adj.", "Observations"]), np.array(["", "", ""])])
    )
    ase(exp, res)


# test _process_frame_axes for different arguments
def test_process_frame_axes_indices():
    df = pd.DataFrame(np.ones((3, 3)), columns=["", "", ""])
    df.index = pd.MultiIndex.from_arrays(
        np.array([["today", "today", "today"], ["var1", "var2", "var3"]])
    )
    df.index.names = ["l1", "l2"]
    par_name_map = {"today": "tomorrow", "var1": "1stvar"}
    index_name_map = ["period", "variable"]
    column_names = list("abc")
    res = _process_frame_indices(
        df,
        custom_param_names=par_name_map,
        custom_index_names=index_name_map,
        column_names=column_names,
        show_col_names=True,
        show_col_groups=False,
        column_groups=None,
    )
    # expected:
    params = """
        period,variable,a,b,c
        tomorrow,1stvar,1,1,1
        tomorrow,var2,1,1,1
        tomorrow,var3,1,1,1
    """
    exp = _read_csv_string(params).fillna("")
    exp.set_index(["period", "variable"], inplace=True)
    afe(res, exp, check_dtype=False)


def test_process_frame_axes_columns():
    df = pd.DataFrame(np.ones((3, 3)), columns=["", "", ""])
    col_names = list("abc")
    col_groups = ["first", "first", "second"]
    res = _process_frame_indices(
        df=df,
        custom_index_names=None,
        custom_param_names=None,
        show_col_groups=True,
        show_col_names=True,
        column_names=col_names,
        column_groups=col_groups,
    )
    arrays = [np.array(col_groups), np.array(col_names)]
    exp = pd.DataFrame(data=np.ones((3, 3)), columns=arrays)
    afe(res, exp, check_dtype=False)


def test_apply_number_format_tuple():
    number_format = ("{0:.2g}", "{0:.2f}", "{0:.2g}")
    raw = pd.DataFrame(data=[1234.2332, 0.0001])
    exp = pd.DataFrame(data=["1.2e+03", "0"])
    res = _apply_number_format(df=raw, number_format=number_format)
    afe(exp, res)


def test_apply_number_format_int():
    number_format = 3
    raw = pd.DataFrame(data=["1234.2332", "1.2e+03"])
    exp = pd.DataFrame(data=["1234.233", "1.2e+03"])
    res = _apply_number_format(df=raw, number_format=number_format)
    afe(exp, res)


def test_apply_number_format_callable():
    def nsf(num, n=3):
        """n-Significant Figures"""
        numstr = ("{0:.%ie}" % (n - 1)).format(num)
        return numstr

    raw = pd.DataFrame(data=[1234.2332, 0.0001])
    exp = pd.DataFrame(data=["1.23e+03", "1.00e-04"])
    res = _apply_number_format(df=raw, number_format=nsf)
    afe(exp, res)


def test_get_digits_after_decimal():
    df = pd.DataFrame(
        data=[["12.456", "0.00003", "1.23e+05"], ["16", "0.03", "1.2e+05"]]
    ).T
    exp = 5
    res = _get_digits_after_decimal(df)
    assert exp == res


def test_create_group_to_col_position():
    col_groups = [
        "a_name",
        "a_name",
        "a_name",
        "second_name",
        "second_name",
        "third_name",
    ]
    exp = {"a_name": [0, 1, 2], "second_name": [3, 4], "third_name": [5]}
    res = _create_group_to_col_position(col_groups)
    assert exp == res


def test_get_model_names():
    m1 = ProcessedModel(params=None, info=None, name="a_name")
    m3 = ProcessedModel(params=None, info=None, name=None)
    m5 = ProcessedModel(params=None, info=None, name="third_name")
    models = [m1, m3, m5]
    res = _get_model_names(models)
    exp = ["a_name", "(2)", "third_name"]
    assert res == exp


def test_get_default_column_names_and_groups():
    model_names = ["a_name", "a_name", "(3)", "(4)", "third_name"]
    res_names, res_groups = _get_default_column_names_and_groups(model_names)
    exp_names = [f"({i+1})" for i in range(len(model_names))]
    exp_groups = ["a_name", "a_name", "(3)", "(4)", "third_name"]
    assert res_names == exp_names
    assert res_groups == exp_groups


def test_get_default_column_names_and_groups_undefined_groups():
    model_names = ["a_name", "second_name", "(3)", "(4)", "third_name"]
    res_names, res_groups = _get_default_column_names_and_groups(model_names)
    exp_names = model_names
    assert res_names == exp_names
    assert pd.isna(res_groups)


def test_customize_col_groups_default():
    default = ["a_name", "a_name", "(3)", "(4)", "third_name"]
    mapping = {"a_name": "first_name", "third_name": "fifth_name"}
    exp = ["first_name", "first_name", "(3)", "(4)", "fifth_name"]
    res = _customize_col_groups(default, mapping)
    assert exp == res


def _read_csv_string(string, index_cols=None):
    string = textwrap.dedent(string)
    return pd.read_csv(io.StringIO(string), index_col=index_cols)
