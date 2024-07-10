import pytest
from optimagic.exceptions import InvalidKwargsError
from optimagic.process_user_function import process_func_of_params


def test_process_func_of_params():
    def f(params, b, c):
        return params + b + c

    func = process_func_of_params(f, {"b": 2, "c": 3})

    assert func(1) == 6


def test_process_func_of_params_too_many_kwargs():
    def f(params, b, c):
        return params + b + c

    with pytest.raises(InvalidKwargsError):
        process_func_of_params(f, {"params": 1, "b": 2, "c": 3})


def test_process_func_of_params_too_few_kwargs():
    def f(params, b, c):
        return params + b + c

    with pytest.raises(InvalidKwargsError):
        process_func_of_params(f, {"c": 3})
