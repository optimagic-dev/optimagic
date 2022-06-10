import numpy as np
import pandas as pd
import pytest
from estimagic.inference.shared import calculate_inference_quantities
from estimagic.inference.shared import process_pandas_arguments
from estimagic.parameters.tree_conversion import FlatParams
from estimagic.parameters.tree_registry import get_registry
from pandas.testing import assert_frame_equal
from pybaum import leaf_names
from pybaum import tree_equal
from pybaum import tree_just_flatten
from pybaum import tree_map


@pytest.fixture
def inputs():
    jac = pd.DataFrame(np.ones((5, 3)), columns=["a", "b", "c"])
    hess = pd.DataFrame(np.eye(3) / 2, columns=list("abc"), index=list("abc"))
    weights = pd.DataFrame(np.eye(5))
    moments_cov = 1 / weights
    out = {"jac": jac, "hess": hess, "weights": weights, "moments_cov": moments_cov}
    return out


def test_process_pandas_arguments_all_pd(inputs):
    *arrays, names = process_pandas_arguments(**inputs)
    for arr in arrays:
        assert isinstance(arr, np.ndarray)

    expected_names = {"moments": list(range(5)), "params": ["a", "b", "c"]}

    for key, value in expected_names.items():
        assert names[key].tolist() == value


def test_process_pandas_arguments_incompatible_names(inputs):
    inputs["jac"].columns = ["c", "d", "e"]

    with pytest.raises(ValueError):
        process_pandas_arguments(**inputs)


def test_calculate_inference_quantities():
    """Test calculate inference quantities for many relevant cases."""

    # ==================================================================================
    # create test inputs
    # ==================================================================================

    estimates = [
        0,
        {
            "a": 1,
            "b": (2, 3),
            "c": np.array([4, 5]),
            "d": pd.Series([6, 7], index=["a", "b"]),
            "e": pd.DataFrame(
                {"col1": [8, 9, 10], "col2": [11, 12, 13]},
                index=pd.MultiIndex.from_tuples([("a", "b"), ("c", "d"), ("e", "f")]),
            ),
            "f": np.array([[15, 16], [17, 18]]),
            "g": pd.DataFrame({"value": [19, 20], "col2": [-1, -1]}, index=["g", "h"]),
        },
    ]

    registry = get_registry(extended=True)
    estimates_flat = tree_just_flatten(estimates, registry=registry)
    names = leaf_names(estimates, registry=registry)

    flat_estimates = FlatParams(
        values=estimates_flat, lower_bounds=None, upper_bounds=None, names=names
    )

    free_cov = pd.DataFrame(
        np.diag(np.arange(len(estimates_flat))), index=names, columns=names
    )

    ci_level = 0  # implies that ci_lower = value = ci_upper

    # ==================================================================================
    # create expected output
    # ==================================================================================

    df = pd.DataFrame(
        {
            "value": flat_estimates.values,
            "standard_error": np.sqrt(np.diag(free_cov)),
            "ci_lower": flat_estimates.values,
            "ci_upper": flat_estimates.values,
        },
        index=names,
    )
    df[["ci_lower", "ci_upper"]] = df[["ci_lower", "ci_upper"]].astype(float)

    df_c = df.loc[["1_c_0", "1_c_1"]]
    df_c.index = pd.RangeIndex(stop=2)

    df_e = df.loc[
        [
            "1_e_a_b_col1",
            "1_e_a_b_col2",
            "1_e_c_d_col1",
            "1_e_c_d_col2",
            "1_e_e_f_col1",
            "1_e_e_f_col2",
        ]
    ]
    df_e.index = pd.MultiIndex.from_tuples(
        [
            ("a", "b", "col1"),
            ("a", "b", "col2"),
            ("c", "d", "col1"),
            ("c", "d", "col2"),
            ("e", "f", "col1"),
            ("e", "f", "col2"),
        ]
    )

    df_f = df.loc[["1_f_0_0", "1_f_0_1", "1_f_1_0", "1_f_1_1"]]
    df_f.index = pd.MultiIndex.from_tuples([(0, 0), (0, 1), (1, 0), (1, 1)])

    expected = [
        df.loc[["0"]].set_axis([0]),
        {
            "a": df.loc[["1_a"]].set_axis([0]),
            "b": (df.loc[["1_b_0"]].set_axis([0]), df.loc[["1_b_1"]].set_axis([0])),
            "c": df_c,
            "d": df.loc[["1_d_a", "1_d_b"]].set_axis(["a", "b"]),
            "e": df_e,
            "f": df_f,
            "g": df.loc[["1_g_g", "1_g_h"]].set_axis(["g", "h"]),
        },
    ]

    # ==================================================================================
    # compute and compare
    # ==================================================================================

    got = calculate_inference_quantities(estimates, flat_estimates, free_cov, ci_level)

    # drop irrelevant columns
    got = tree_map(lambda df: df.drop(columns=["stars", "p_value", "free"]), got)

    # for debugging purposes we first compare each leaf
    for got_leaf, exp_leaf in zip(tree_just_flatten(got), tree_just_flatten(expected)):
        assert_frame_equal(got_leaf, exp_leaf)

    assert tree_equal(expected, got)
