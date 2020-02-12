"""Test the data manipulation done by the interactive distribution plot."""
from io import StringIO

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest  # noqa

import estimagic.visualization.distribution_plot.manipulate_data as md


@pytest.fixture()
def simple_test_df():
    csv = """
    ,g1,g2,value,id,sub
    0, 1, b, 0.18, 1, s1
    1, 0, a, 0.1, 0, s1
    2, 1, b, 0.11, 0, s2
    3, 0, a, 0.11, 1, s1
    4, 1, a, 0.2, 0, s1
    5, 0, b, 0.15, 0, s2
    """
    df = pd.read_csv(StringIO(csv))
    return df


# True Unit Tests
# ================


def test_safely_reset_index_unnamed_index():
    df = pd.DataFrame(np.arange(10).reshape(5, 2))
    df.index = df.index * 2
    res = md._safely_reset_index(df)
    expected = df.copy(deep=True)
    expected["index__0"] = expected.index
    expected.index = [0, 1, 2, 3, 4]
    pdt.assert_frame_equal(res, expected, check_like=True)


def test_safely_reset_index_non_present_index_name():
    df = pd.DataFrame(np.arange(10).reshape(5, 2))
    df.index = df.index * 2
    df.index.name = "my_index"
    res = md._safely_reset_index(df)
    pdt.assert_frame_equal(res, df)


def test_safely_reset_index_already_present_index_name():
    df = pd.DataFrame(np.arange(10).reshape(5, 2))
    df.index = df.index * 2
    df.index.name = "my_index"
    df["my_index"] = ["a"] * 5
    res = md._safely_reset_index(df)
    expected = df.copy(deep=True)
    expected["my_index_0"] = expected.index
    expected = expected.reset_index(drop=True)
    pdt.assert_frame_equal(res, expected, check_like=True)


def test_drop_nans_and_sort_no_delete(simple_test_df):
    df = simple_test_df
    res = md._drop_nans_and_sort(df=df, subgroup_col="sub", group_cols=["g1", "g2"])
    expected = df.sort_values(["g1", "g2", "sub", "value", "id"])
    assert set(res.columns) == set(expected.columns)
    expected = expected[res.columns]
    pdt.assert_frame_equal(res, expected)


## def test_clean_subgroup_col():
##     assert True is False
##
##
## def test_create_color_col():
##     assert True is False
##
##
## def test_bin_width_and_midpoints_per_group():
##     assert True is False
##
##
## def test_calculate_x_bounds():
##     assert True is False
##
##
## # Integration Tests
## # ==================
##
##
## def test_bin_width_and_midpoints():
##     assert True is False
##
##
## def test_add_hist_cols():
##     assert True is False
##
##
## def test_clean_data():
##     assert True is False
