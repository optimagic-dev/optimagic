import itertools
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import optimagic as om
from optimagic.logging import SQLiteLogOptions
from optimagic.optimization.optimize import minimize
from optimagic.parameters.bounds import Bounds
from optimagic.visualization.history_plots import (
    LineData,
    _extract_criterion_plot_lines,
    _harmonize_inputs_to_dict,
    _PlottingMultistartHistory,
    _retrieve_optimization_data,
    criterion_plot,
    params_plot,
)


@pytest.fixture()
def minimize_result():
    bounds = Bounds(soft_lower=np.full(5, -1), soft_upper=np.full(5, 6))
    out = {}
    for multistart in [True, False]:
        res = []
        for algorithm in ["scipy_neldermead", "scipy_lbfgsb"]:
            _res = minimize(
                fun=lambda x: x @ x,
                params=np.arange(5),
                algorithm=algorithm,
                bounds=bounds,
                multistart=(
                    om.MultistartOptions(n_samples=1000, convergence_max_discoveries=5)
                    if multistart
                    else None
                ),
            )
            res.append(_res)
        out[multistart] = res
    return out


# ======================================================================================
# Params plot
# ======================================================================================


TEST_CASES = list(
    itertools.product(
        [True, False],  # multistart
        [None, lambda x: x[:2]],  # selector
        [None, 50],  # max_evaluations
        [True, False],  # show_exploration
    )
)


@pytest.mark.parametrize(
    "multistart, selector, max_evaluations, show_exploration", TEST_CASES
)
def test_params_plot_multistart(
    minimize_result, multistart, selector, max_evaluations, show_exploration
):
    for _res in minimize_result[multistart]:
        params_plot(
            _res,
            selector=selector,
            max_evaluations=max_evaluations,
            show_exploration=show_exploration,
        )


# ======================================================================================
# Test criterion plot
# ======================================================================================


TEST_CASES = list(itertools.product([True, False], repeat=4))


@pytest.mark.parametrize(
    "multistart, monotone, stack_multistart, exploration", TEST_CASES
)
def test_criterion_plot_list_input(
    minimize_result, multistart, monotone, stack_multistart, exploration
):
    res = minimize_result[multistart]

    criterion_plot(
        res,
        monotone=monotone,
        stack_multistart=stack_multistart,
        show_exploration=exploration,
    )


def test_criterion_plot_name_input(minimize_result):
    result = minimize_result[False]
    criterion_plot(result[0], names="neldermead", palette="blue")


def test_criterion_plot_wrong_results():
    with pytest.raises(TypeError):
        criterion_plot([10, np.array([1, 2, 3])])


def test_criterion_plot_different_input_types():
    bounds = Bounds(soft_lower=np.full(5, -1), soft_upper=np.full(5, 6))
    # logged result
    minimize(
        fun=lambda x: x @ x,
        params=np.arange(5),
        algorithm="scipy_lbfgsb",
        bounds=bounds,
        multistart=om.MultistartOptions(n_samples=1000, convergence_max_discoveries=5),
        logging=SQLiteLogOptions("test.db", fast_logging=True),
    )

    res = minimize(
        fun=lambda x: x @ x,
        params=np.arange(5),
        algorithm="scipy_lbfgsb",
        bounds=bounds,
        multistart=om.MultistartOptions(n_samples=1000, convergence_max_discoveries=5),
    )

    results = ["test.db", res]

    criterion_plot(results)
    criterion_plot(results, monotone=True)
    criterion_plot(results, stack_multistart=True)
    criterion_plot(results, monotone=True, stack_multistart=True)
    criterion_plot(results, show_exploration=True)
    criterion_plot("test.db")


def test_criterion_plot_wrong_inputs():
    with pytest.raises(ValueError):
        criterion_plot("bla", names=[1, 2])

    with pytest.raises(ValueError):
        criterion_plot(["bla", "bla"], names="blub")


def test_harmonize_inputs_to_dict_single_result():
    res = minimize(fun=lambda x: x @ x, params=np.arange(5), algorithm="scipy_lbfgsb")
    assert _harmonize_inputs_to_dict(results=res, names=None) == {"0": res}


def test_harmonize_inputs_to_dict_single_result_with_name():
    res = minimize(fun=lambda x: x @ x, params=np.arange(5), algorithm="scipy_lbfgsb")
    assert _harmonize_inputs_to_dict(results=res, names="bla") == {"bla": res}


def test_harmonize_inputs_to_dict_list_results():
    res = minimize(fun=lambda x: x @ x, params=np.arange(5), algorithm="scipy_lbfgsb")
    results = [res, res]
    assert _harmonize_inputs_to_dict(results=results, names=None) == {
        "0": res,
        "1": res,
    }


def test_harmonize_inputs_to_dict_dict_input():
    res = minimize(fun=lambda x: x @ x, params=np.arange(5), algorithm="scipy_lbfgsb")
    results = {"bla": res, om.algos.scipy_lbfgsb(): res, om.algos.scipy_neldermead: res}
    got = _harmonize_inputs_to_dict(results=results, names=None)
    expected = {"bla": res, "scipy_lbfgsb": res, "scipy_neldermead": res}
    assert got == expected


def test_harmonize_inputs_to_dict_dict_input_with_names():
    res = minimize(fun=lambda x: x @ x, params=np.arange(5), algorithm="scipy_lbfgsb")
    results = {"bla": res, "blub": res}
    got = _harmonize_inputs_to_dict(results=results, names=["a", "b"])
    expected = {"a": res, "b": res}
    assert got == expected


def test_harmonize_inputs_to_dict_invalid_names():
    results = [None]
    names = ["a", "b"]
    with pytest.raises(ValueError):
        _harmonize_inputs_to_dict(results=results, names=names)


def test_harmonize_inputs_to_dict_str_input():
    assert _harmonize_inputs_to_dict(results="test.db", names=None) == {"0": "test.db"}


def test_harmonize_inputs_to_dict_path_input():
    path = Path("test.db")
    assert _harmonize_inputs_to_dict(results=path, names=None) == {"0": path}


def _compare_plotting_multistart_history_with_result(
    data: _PlottingMultistartHistory, res: om.OptimizeResult, res_name: str
):
    assert_array_equal(data.history.fun, res.history.fun)
    assert data.name == res_name
    assert_array_equal(data.start_params, res.start_params)
    assert data.is_multistart == (res.multistart_info is not None)


def test_retrieve_data_from_result(minimize_result):
    res = minimize_result[False][0]
    results = {"bla": res}

    data = _retrieve_optimization_data(
        results=results, stack_multistart=False, show_exploration=False
    )

    assert isinstance(data, list) and len(data) == 1
    assert isinstance(data[0], _PlottingMultistartHistory)
    _compare_plotting_multistart_history_with_result(
        data=data[0], res=res, res_name="bla"
    )


def test_retrieve_data_from_logged_result(tmp_path):
    res = minimize(
        fun=lambda x: x @ x,
        params=np.arange(2),
        algorithm="scipy_lbfgsb",
        logging=SQLiteLogOptions(tmp_path / "test.db", fast_logging=True),
    )
    results = {"logged": tmp_path / "test.db"}

    data = _retrieve_optimization_data(
        results=results, stack_multistart=False, show_exploration=False
    )

    assert isinstance(data, list) and len(data) == 1
    assert isinstance(data[0], _PlottingMultistartHistory)
    _compare_plotting_multistart_history_with_result(
        data=data[0], res=res, res_name="logged"
    )


@pytest.mark.parametrize("stack_multistart", [True, False])
def test_retrieve_data_from_multistart_result(minimize_result, stack_multistart):
    res = minimize_result[True][0]
    results = {"multistart": res}

    data = _retrieve_optimization_data(
        results=results, stack_multistart=stack_multistart, show_exploration=False
    )

    assert isinstance(data, list) and len(data) == 1

    assert data[0].is_multistart
    assert len(data[0].local_histories) == 5

    if stack_multistart:
        assert_array_equal(
            data[0].stacked_local_histories.fun,
            np.concatenate([hist.fun for hist in data[0].local_histories]),
        )
    else:
        assert data[0].stacked_local_histories is None


def test_extract_criterion_plot_lines(minimize_result):
    res = minimize_result[True][0]
    results = {"multistart": res}
    data = _retrieve_optimization_data(
        results=results, stack_multistart=False, show_exploration=False
    )

    palette_cycle = itertools.cycle(["red", "green", "blue"])

    lines, multistart_lines = _extract_criterion_plot_lines(
        data=data,
        max_evaluations=None,
        palette_cycle=palette_cycle,
        stack_multistart=False,
        monotone=False,
    )

    history = res.history.fun

    assert isinstance(lines, list) and len(lines) == 1
    assert isinstance(lines[0], LineData)

    assert_array_equal(lines[0].x, np.arange(len(history)))
    assert_array_equal(lines[0].y, history)

    assert isinstance(multistart_lines, list) and all(
        isinstance(line, LineData) for line in multistart_lines
    )
    assert len(multistart_lines) == 5
