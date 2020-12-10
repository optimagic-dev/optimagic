import itertools
import sys
import warnings

import pytest

from estimagic.batch_evaluators import joblib_batch_evaluator


batch_evaluators = [
    joblib_batch_evaluator,
]

n_core_list = [1, 2]

test_cases = list(itertools.product(batch_evaluators, n_core_list))


def double(x):
    return 2 * x


def buggy_func(x):
    raise AssertionError()


def add_x_and_y(x, y):
    return x + y


@pytest.mark.skipif(sys.platform == "darwin", reason="Too slow on Mac OS CI server.")
@pytest.mark.parametrize("batch_evaluator, n_cores", test_cases)
def test_batch_evaluator_without_exceptions(batch_evaluator, n_cores):

    calculated = batch_evaluator(
        func=double,
        arguments=list(range(10)),
        n_cores=n_cores,
    )

    expected = list(range(0, 20, 2))

    assert calculated == expected


@pytest.mark.skipif(sys.platform == "darwin", reason="Too slow on Mac OS CI server.")
@pytest.mark.parametrize("batch_evaluator, n_cores", test_cases)
def test_batch_evaluator_with_unhandled_exceptions(batch_evaluator, n_cores):
    with pytest.raises(AssertionError):
        batch_evaluator(
            func=buggy_func,
            arguments=list(range(10)),
            n_cores=n_cores,
            error_handling="raise",
        )


@pytest.mark.slow
@pytest.mark.parametrize("batch_evaluator, n_cores", test_cases)
def test_batch_evaluator_with_handled_exceptions(batch_evaluator, n_cores):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        calculated = batch_evaluator(
            func=buggy_func,
            arguments=list(range(10)),
            n_cores=n_cores,
            error_handling="continue",
        )

        for calc in calculated:
            assert isinstance(calc, str)


@pytest.mark.skipif(sys.platform == "darwin", reason="Too slow on Mac OS CI server.")
@pytest.mark.parametrize("batch_evaluator, n_cores", test_cases)
def test_batch_evaluator_with_list_unpacking(batch_evaluator, n_cores):
    calculated = batch_evaluator(
        func=add_x_and_y,
        arguments=[(1, 2), (3, 4)],
        n_cores=n_cores,
        unpack_symbol="*",
    )
    expected = [3, 7]
    assert calculated == expected


@pytest.mark.skipif(sys.platform == "darwin", reason="Too slow on Mac OS CI server.")
@pytest.mark.parametrize("batch_evaluator, n_cores", test_cases)
def test_batch_evaluator_with_dict_unpacking(batch_evaluator, n_cores):
    calculated = batch_evaluator(
        func=add_x_and_y,
        arguments=[{"x": 1, "y": 2}, {"x": 3, "y": 4}],
        n_cores=n_cores,
        unpack_symbol="**",
    )
    expected = [3, 7]
    assert calculated == expected
