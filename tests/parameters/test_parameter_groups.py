import numpy as np
import pandas as pd
from estimagic.parameters.parameter_groups import _get_group_and_name
from estimagic.parameters.parameter_groups import _replace_too_common_groups
from estimagic.parameters.parameter_groups import _split_long_group
from estimagic.parameters.parameter_groups import get_params_groups_and_short_names


def test_get_params_groups_and_short_names_dict():
    params = {
        "alone": 30,
        "list_of_2": [10, 11],
        "fixed": [5, 10],
        "nested": {"c": [20, 21], "d": [22, 23], "e": 26},
        "to_be_split": 40 + np.arange(15),
    }
    free_mask = [True] * 3 + [False] * 2 + [True] * (5 + 15)
    res_groups, res_names = get_params_groups_and_short_names(
        params=params, free_mask=free_mask
    )
    expected_groups = (
        [
            "alone",
            "list_of_2",
            "list_of_2",
            None,
            None,
            "nested, c",
            "nested, c",
            "nested, d",
            "nested, d",
            "nested",
        ]
        + ["to_be_split, 1"] * 8
        + ["to_be_split, 2"] * 7
    )
    expected_names = [
        "alone",
        "0",
        "1",
        "fixed_0",
        "fixed_1",
        "0",
        "1",
        "0",
        "1",
        "e",
    ] + [str(i) for i in range(15)]

    assert res_groups == expected_groups
    assert res_names == expected_names


def test_get_params_groups_and_short_names_numpy():
    params = np.arange(15).reshape(5, 3)
    expected_groups = ["Parameters, 1"] * 8 + ["Parameters, 2"] * 7
    expected_names = [f"{j}_{i}" for j in range(5) for i in range(3)]
    res_groups, res_names = get_params_groups_and_short_names(
        params=params, free_mask=[True] * 15
    )
    assert expected_groups == res_groups
    assert expected_names == res_names


def test_get_params_groups_and_short_names_dataframe():
    params = pd.DataFrame({"value": np.arange(15)})
    expected_groups = ["Parameters, 1"] * 8 + ["Parameters, 2"] * 7
    expected_names = [str(i) for i in range(15)]
    res_groups, res_names = get_params_groups_and_short_names(
        params=params, free_mask=[True] * 15
    )
    assert expected_groups == res_groups
    assert expected_names == res_names


def test_get_group_and_name_not_free():
    res = _get_group_and_name(["a", "test", "hello"], is_free=False)
    assert res == (None, "a_test_hello")


def test_get_group_and_name_free():
    res = _get_group_and_name(["a", "test", "hello"], is_free=True)
    assert res == ("a, test", "hello")


def test_get_group_and_name_just_one():
    res = _get_group_and_name(["hello"], is_free=True)
    assert res == ("hello", "hello")


def test_split_long_group_short():
    res = _split_long_group("bla", 15)
    expected = np.array(["bla, 1"] * 8 + ["bla, 2"] * 7)
    assert (res == expected).all()


def test_split_long_group_very_short():
    res = _split_long_group("bla", 7)
    expected = np.array(["bla, 1"] * 7)
    assert (res == expected).all()


def test_split_long_group_20():
    res = _split_long_group("bla", 20)
    expected = np.array(["bla, 1"] * 7 + ["bla, 2"] * 7 + ["bla, 3"] * 6)
    assert (res == expected).all()


def test_split_long_group_23():
    res = _split_long_group("bla", 23)
    expected = np.array(["bla, 1"] * 8 + ["bla, 2"] * 8 + ["bla, 3"] * 7)
    assert (res == expected).all()


def test_replace_too_common_groups():
    groups = ["a", "b", "c", "b", "b"]
    split_group_names = ["d", "e", "f"]
    res = _replace_too_common_groups(groups, "b", split_group_names)
    assert res == ["a", "d", "c", "e", "f"]
