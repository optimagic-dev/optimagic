import io
import textwrap
from collections import namedtuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from estimagic.config import EXAMPLE_DIR
from estimagic.visualization.estimation_table import _convert_model_to_series
from estimagic.visualization.estimation_table import _create_statistics_sr
from estimagic.visualization.estimation_table import _process_body_df
from estimagic.visualization.estimation_table import _process_model
from estimagic.visualization.estimation_table import estimation_table
from pandas.testing import assert_frame_equal as afe
from pandas.testing import assert_series_equal as ase

# test process_model for different model types
NamedTup = namedtuple("NamedTup", "params info")

fix_path = EXAMPLE_DIR / "diabetes.csv"

df_ = pd.read_csv(fix_path, index_col=0)
est = sm.OLS(endog=df_["target"], exog=sm.add_constant(df_[df_.columns[0:4]])).fit()


def test_estimation_table():
    models = [est]
    res = estimation_table(models, return_type="render_inputs", append_notes=False)
    exp = {}
    body_str = """
        index,(1)
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
    exp["body_df"] = _read_csv_string(body_str).fillna("")
    exp["body_df"].set_index("index", inplace=True)
    footer_str = """
         ,(1)
        R$^2$,0.40
        Adj. R$^2$,0.40
        Residual Std. Error,60.00
        F Statistic,72.90$^{***}$
        Observations,442

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
    df = pd.DataFrame(columns=["value", "p_value", "ci_lower", "ci_upper"])
    df["value"] = np.arange(10)
    df["p_value"] = np.arange(10)
    df["ci_lower"] = np.arange(10)
    df["ci_upper"] = np.arange(10)
    info = {"stat1": 0, "stat2": 0}
    model = NamedTup(params=df, info=info)
    res = _process_model(model)
    afe(res.params, df)
    ase(pd.Series(res.info), pd.Series(info))


def test_process_model_stats_model():
    par_df = pd.DataFrame(
        columns=["value", "p_value", "standard_error", "ci_lower", "ci_upper"],
        index=["const", "Age", "Sex", "BMI", "ABP"],
    )
    par_df["value"] = [152.133484, 37.241211, -106.577520, 787.179313, 416.673772]
    par_df["p_value"] = [
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
    res = _convert_model_to_series(df, significance_levels, show_stars)
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
    res = _convert_model_to_series(df, significance_levels, show_stars)
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
    res = _convert_model_to_series(df, significance_levels, show_stars)
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
    model = NamedTup(params=df, info=info)
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


def test_process_body_df_columns():

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
