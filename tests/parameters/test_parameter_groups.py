import numpy as np
from estimagic.parameters.parameter_groups import _get_group_and_name
from estimagic.parameters.parameter_groups import _split_long_group
from estimagic.parameters.parameter_groups import get_params_groups
from estimagic.parameters.tree_conversion import FlatParams
from pybaum import leaf_names
from pybaum import tree_just_flatten


def test_get_params_groups():
    params = {
        "alone": 30,
        "list_of_2": [10, 11],
        "fixed": [5, 10],
        "nested": {"c": [20, 21], "d": [22, 23], "e": 26},
        "to_be_split": 40 + np.arange(15),
    }
    param_values = tree_just_flatten(params)
    flat_params = FlatParams(
        values=param_values,
        lower_bounds=[-np.inf] * len(param_values),
        upper_bounds=[np.inf] * len(param_values),
        names=leaf_names(params),
        # adjust free_mask if the order in the definition of params changes
        free_mask=[True, True, True, False, False] + [True] * (5 + 15),
    )
    res = get_params_groups(params=params, flat_params=flat_params)
    expected = (
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

    assert res == expected


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
    expected = ["bla, 1"] * 8 + ["bla, 2"] * 7
    assert res == expected


def test_split_long_group_very_short():
    res = _split_long_group("bla", 7)
    expected = ["bla, 1"] * 7
    assert res == expected


def test_split_long_group_20():
    res = _split_long_group("bla", 20)
    expected = ["bla, 1"] * 7 + ["bla, 2"] * 7 + ["bla, 3"] * 6
    assert res == expected


def test_split_long_group_23():
    res = _split_long_group("bla", 23)
    expected = ["bla, 1"] * 8 + ["bla, 2"] * 8 + ["bla, 3"] * 7
    assert res == expected
