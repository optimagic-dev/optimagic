import pytest
from estimagic.optimization.tranquilo.get_component import (
    _add_redundant_argument_handling,
    _fail_if_mandatory_argument_is_missing,
    _get_function_and_name,
    _get_valid_options,
    get_component,
)


@pytest.fixture()
def func_dict():
    out = {
        "f": lambda x: x,
        "g": lambda x, y: x + y,
    }
    return out


def test_get_component(func_dict):
    got = get_component(
        name_or_func="g",
        component_name="component",
        func_dict=func_dict,
        default_options={"x": 1},
        user_options={"y": 2},
        redundant_option_handling="ignore",
        redundant_argument_handling="ignore",
        mandatory_arguments=["x"],
    )

    assert got() == 3
    assert got(bla=15) == 3


def test_get_function_and_name_valid_string(func_dict):
    _func, _name = _get_function_and_name(
        name_or_func="f",
        component_name="component",
        func_dict=func_dict,
    )
    assert _func == func_dict["f"]
    assert _name == "f"


def test_get_function_and_name_invalid_string():
    with pytest.raises(ValueError, match="If component is a string, it must be one of"):
        _get_function_and_name(
            name_or_func="h",
            component_name="component",
            func_dict={"f": lambda x: x, "g": lambda x, y: x + y},
        )


def test_get_function_and_name_valid_function():
    def _f(x):
        return x

    _func, _name = _get_function_and_name(
        name_or_func=_f,
        component_name="component",
        func_dict=None,
    )
    assert _func == _f
    assert _name == "_f"


def test_get_function_and_string_wrong_type():
    with pytest.raises(TypeError, match="name_or_func must be a string or a callable."):
        _get_function_and_name(
            name_or_func=1,
            component_name="component",
            func_dict=None,
        )


def test_get_valid_options_ignore():
    got = _get_valid_options(
        default_options={"a": 1, "b": 2},
        user_options={"a": 3, "c": 4},
        signature=["a", "c"],
        name="bla",
        component_name="component",
        redundant_option_handling="ignore",
    )
    expected = {"a": 3, "c": 4}

    assert got == expected


def test_get_valid_options_raise():
    with pytest.raises(ValueError, match="The following options are not supported"):
        _get_valid_options(
            default_options={"a": 1, "b": 2},
            user_options={"a": 3, "c": 4},
            signature=["a", "c"],
            name="bla",
            component_name="component",
            redundant_option_handling="raise",
        )


def test_get_valid_options_warn():
    with pytest.warns(UserWarning, match="The following options are not supported"):
        _get_valid_options(
            default_options={"a": 1, "b": 2},
            user_options={"a": 3, "c": 4},
            signature=["a", "c"],
            name="bla",
            component_name="component",
            redundant_option_handling="warn",
        )


def test_fail_if_mandatory_argument_is_missing():
    with pytest.raises(
        ValueError, match="The following mandatory arguments are missing"
    ):
        _fail_if_mandatory_argument_is_missing(
            mandatory_arguments=["a", "c"],
            signature=["a", "b"],
            name="bla",
            component_name="component",
        )


def test_add_redundant_argument_handling_ignore():
    def f(a, b):
        return a + b

    _f = _add_redundant_argument_handling(
        func=f,
        signature=["a", "b"],
        warn=False,
    )

    assert _f(1, b=2, c=3) == 3


def test_add_redundant_argument_handling_warn():
    def f(a, b):
        return a + b

    _f = _add_redundant_argument_handling(
        func=f,
        signature=["a", "b"],
        warn=True,
    )
    with pytest.warns(UserWarning, match="The following arguments are not supported"):
        _f(1, b=2, c=3)
