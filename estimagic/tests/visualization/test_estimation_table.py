import io
import textwrap
from collections import namedtuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pandas.testing import assert_frame_equal as afe
from pandas.testing import assert_series_equal as ase

from estimagic.config import DOCS_DIR
from estimagic.visualization.estimation_table import _convert_model_to_series
from estimagic.visualization.estimation_table import _create_statistics_sr
from estimagic.visualization.estimation_table import _process_body_df
from estimagic.visualization.estimation_table import _process_model
from estimagic.visualization.estimation_table import estimation_table

# test process_model for different model types
NamedTup = namedtuple("NamedTup", "params info")

fix_path = DOCS_DIR / "source/visualization/diabetes.csv"

df_ = pd.read_csv(fix_path, index_col=0)
est = sm.OLS(endog=df_["target"], exog=sm.add_constant(df_[df_.columns[0:4]])).fit()


def test_estimation_table():
    models = [est]
    return_type = "python"
    res = estimation_table(models, return_type, append_notes=False)
    exp = {}
    body_str = """
        index,{(1)}
        const,152.13$^{*** }$
        ,(2.85)
        Age,37.24$^{ }$
        ,(64.12)
        Sex,-106.58$^{* }$
        ,(62.13)
        BMI,787.18$^{*** }$
        ,(65.42)
        ABP,416.67$^{*** }$
        ,(69.49)
    """
    exp["body_df"] = _read_csv_string(body_str).fillna("")
    exp["body_df"].set_index("index", inplace=True)
    footer_str = """
         ,{(1)}
        Observations,442.0
        R$^2$,0.4
        Adj. R$^2$,0.39
        Residual Std. Error,59.98
        F Statistic,72.91$^{***}$
    """
    exp["footer_df"] = _read_csv_string(footer_str).fillna("")
    exp["footer_df"].set_index(" ", inplace=True)
    exp["footer_df"].index.names = [None]
    exp["footer_df"].index = pd.MultiIndex.from_arrays([exp["footer_df"].index])
    exp["notes_tex"] = "\\midrule\n"
    exp[
        "notes_html"
    ] = """<tr><td colspan="2" style="border-bottom: 1px solid black">
        </td></tr>"""

    afe(exp["footer_df"], res["footer_df"])
    afe(exp["body_df"], res["body_df"], check_index_type=False)
    ase(pd.Series(exp["notes_html"]), pd.Series(res["notes_html"]))
    ase(pd.Series(exp["notes_tex"]), pd.Series(res["notes_tex"]))


def test_process_model_namedtuple():
    # checks that process_model doesn't alter values
    df = pd.DataFrame(columns=["value", "pvalue", "ci_lower", "ci_upper"])
    df["value"] = np.arange(10)
    df["pvalue"] = np.arange(10)
    df["ci_lower"] = np.arange(10)
    df["ci_upper"] = np.arange(10)
    info = {"stat1": 0, "stat2": 0}
    model = NamedTup(params=df, info=info)
    res = _process_model(model)
    afe(res.params, df)
    ase(pd.Series(res.info), pd.Series(info))


def test_process_model_stats_model():
    par_df = pd.DataFrame(
        columns=["value", "pvalue", "standard_error", "ci_lower", "ci_upper"],
        index=["const", "Age", "Sex", "BMI", "ABP"],
    )
    par_df["value"] = [152.133484, 37.241211, -106.577520, 787.179313, 416.673772]
    par_df["pvalue"] = [
        2.048808e-193,
        5.616557e-01,
        8.695658e-02,
        5.345260e-29,
        4.245663e-09,
    ]
    par_df["standard_error"] = [2.852749, 64.117433, 62.125062, 65.424126, 69.494666]
    par_df["ci_lower"] = [146.526671, -88.775663, -228.678572, 658.594255, 280.088446]
    par_df["ci_upper"] = [157.740298, 163.258084, 15.523532, 915.764371, 553.259097]
    info_dict = {}
    info_dict["rsquared"] = 0.40026108237714
    info_dict["rsquared_adj"] = 0.39477148130050055
    info_dict["fvalue"] = 72.91259907398705
    info_dict["f_pvalue"] = 2.700722880950139e-47
    info_dict["df_model"] = 4.0
    info_dict["df_resid"] = 437.0
    info_dict["dependent_variable"] = "target"
    info_dict["resid_std_err"] = 59.97560860753488
    info_dict["n_obs"] = 442.0
    res = _process_model(est)
    afe(res.params, par_df)
    ase(pd.Series(res.info), pd.Series(info_dict))


def test_process_model_dict():
    df = pd.DataFrame(columns=["value", "pvalue", "standard_error"])
    df["value"] = np.arange(10)
    df["pvalue"] = np.arange(10)
    df["standard_error"] = np.arange(10)
    info = {"stat1": 0, "stat2": 0}
    mod = {}
    mod["params"] = df
    mod["info"] = info
    res = _process_model(mod)
    afe(res.params, mod["params"])
    ase(pd.Series(res.info), pd.Series(mod["info"]))


# test convert_model_to_series for different arguments
def test_convert_model_to_series_conf_int():
    df = pd.DataFrame(
        np.array(
            [[0.6, 2.3, 3.3], [0.11, 0.049, 0.009], [0.6, 2.3, 3.3], [1.2, 3.3, 4.3]]
        ).T,
        columns=["value", "pvalue", "ci_lower", "ci_upper"],
        index=["a", "b", "c"],
    )
    si_lev = [0.1, 0.05, 0.01]
    si_dig = 2
    ci = True
    si = True
    ss = True
    res = _convert_model_to_series(df, si_lev, si_dig, si, ci, ss)
    exp = pd.Series(
        [
            "0.6$^{ }$",
            r"{(0.6\,;\,1.2)}",
            "2.3$^{** }$",
            r"{(2.3\,;\,3.3)}",
            "3.3$^{*** }$",
            r"{(3.3\,;\,4.3)}",
        ],
        index=["a", "", "b", "", "c", ""],
        name="",
    )
    exp.index.name = "index"
    ase(exp, res)


def test_convert_model_to_series_std_err():
    df = pd.DataFrame(
        np.array([[0.6, 2.3, 3.3], [0.11, 0.049, 0.009], [0.6, 2.3, 3.3]]).T,
        columns=["value", "pvalue", "standard_error"],
        index=["a", "b", "c"],
    )
    si_lev = [0.1, 0.05, 0.01]
    si_dig = 2
    ci = False
    si = True
    ss = True
    res = _convert_model_to_series(df, si_lev, si_dig, si, ci, ss)
    exp = pd.Series(
        ["0.6$^{ }$", "(0.6)", "2.3$^{** }$", "(2.3)", "3.3$^{*** }$", "(3.3)"],
        index=["a", "", "b", "", "c", ""],
        name="",
    )
    exp.index.name = "index"
    ase(exp, res)


def test_convert_model_to_series_no_inference():
    df = pd.DataFrame(
        np.array([[0.6, 2.3, 3.3], [0.11, 0.049, 0.009], [0.6, 2.3, 3.3]]).T,
        columns=["value", "pvalue", "standard_error"],
        index=["a", "b", "c"],
    )
    si_lev = [0.1, 0.05, 0.01]
    si_dig = 2
    ci = False
    si = False
    ss = True
    res = _convert_model_to_series(df, si_lev, si_dig, si, ci, ss)
    exp = pd.Series(
        ["0.6$^{ }$", "2.3$^{** }$", "3.3$^{*** }$"], index=["a", "b", "c"], name=""
    )
    ase(exp, res)


# test create stat series
def test_create_statistics_sr():
    df = pd.DataFrame(np.empty((10, 3)), columns=["a", "b", "c"])
    df.index = pd.MultiIndex.from_arrays(np.array([np.arange(10), np.arange(10)]))
    info_dict = {"rsquared": 0.45, "n_obs": 400}
    sig_dig = 2
    sig_levels = [0.1, 0.2]
    show_stars = False
    model = NamedTup(params=df, info=info_dict)
    stats_dict = {"Observations": "n_obs", "R2": "rsquared", "show_dof": False}
    res = _create_statistics_sr(model, stats_dict, sig_levels, show_stars, sig_dig)
    exp = pd.Series([str(400), str(0.45)])
    exp.index = pd.MultiIndex.from_arrays(
        np.array([np.array(["Observations", "R2"]), np.array(["", ""])])
    )
    ase(exp, res)


# test process_params_df for different arguments
def test_process_body_df_indices():
    df = pd.DataFrame(np.ones((3, 3)), columns=list("abc"))
    df.index = pd.MultiIndex.from_arrays(
        np.array([["today", "today", "today"], ["var1", "var2", "var3"]])
    )
    df.index.names = ["l1", "l2"]
    par_map = {"today": "tomorrow", "var1": "1stvar"}
    name_map = ["period", "variable"]
    res = _process_body_df(
        df,
        par_map,
        name_map,
        show_col_names=False,
        custom_col_names=None,
        custom_model_names=None,
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


def test_process_params_df_columns():

    df = pd.DataFrame(np.ones((3, 6)), columns=list("abcdef"))
    custom_col_names = ["c" + str(i) for i in range(1, 7)]
    custom_model_names = {"m3-5": [2, 3, 4]}
    exp = _process_body_df(df, None, None, True, custom_col_names, custom_model_names)
    df.columns = pd.MultiIndex.from_arrays(
        np.array(
            [
                ["{}", "{}", "{m3-5}", "{m3-5}", "{m3-5}", "{}"],
                ["{c1}", "{c2}", "{c3}", "{c4}", "{c5}", "{c6}"],
            ]
        )
    )

    afe(df, exp, check_dtype=False)


def _read_csv_string(string, index_cols=None):
    string = textwrap.dedent(string)
    return pd.read_csv(io.StringIO(string), index_col=index_cols)
