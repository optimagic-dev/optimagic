from itertools import product

import numpy as np
import pytest

import optimagic as om
from optimagic import (
    convergence_report,
    get_benchmark_problems,
    rank_report,
    traceback_report,
)
from optimagic.benchmarking.benchmark_result import BenchmarkResult


def _result(params_history, fun_history, time_history, batches_history, traceback):
    return BenchmarkResult(
        params_history=np.array(params_history, dtype=np.float64),
        fun_history=np.array(fun_history, dtype=np.float64),
        time_history=np.array(time_history, dtype=np.float64),
        batches_history=np.array(batches_history, dtype=np.int64),
        optimize_result=None,
        traceback=traceback,
    )


@pytest.fixture
def benchmark_example():
    all_problems = get_benchmark_problems("example")
    problems = {
        k: v
        for k, v in all_problems.items()
        if k in ["bard_good_start", "box_3d", "rosenbrock_good_start"]
    }
    optimizers = {
        "lbfgsb": om.algos.scipy_lbfgsb(stopping_maxfun=10, stopping_maxiter=10),
        "nm": om.algos.scipy_neldermead(stopping_maxfun=10, stopping_maxiter=10),
    }

    results = {
        # success
        ("bard_good_start", "lbfgsb"): _result(
            params_history=[
                [1.0, 1.0, 1.0],
                [0.48286315298120086, 1.6129119244711858, 1.5974181569859445],
                [0.09754340799557773, 1.7558262514618663, 1.7403560082627973],
            ],
            fun_history=[4.16816959e01, 3.20813118e00, 9.97263708e-03],
            time_history=[0.0, 0.0003762839987757616, 0.0007037959985609632],
            batches_history=[0, 1, 2],
            traceback=None,
        ),
        # failed
        ("box_3d", "lbfgsb"): _result(
            params_history=[
                [0.0, 10.0, 20.0],
                [-0.6579976970071755, 10.014197643614924, 19.247113914560085],
                [-3.2899884850358774, 10.070988218074623, 16.235569572800433],
            ],
            fun_history=[1.03115381e03, 8.73640769e02, 9.35093416e02],
            time_history=[0.0, 0.000555748996703187, 0.0009771709992492106],
            batches_history=[0, 1, 2],
            traceback=None,
        ),
        # error
        ("rosenbrock_good_start", "lbfgsb"): _result(
            params_history=[[-1.2, 1.0], [0.0, 0.0]],
            fun_history=[1.795769e6, 1e3],
            time_history=[0.0, 5.73799989069812e-04],
            batches_history=[0, 1],
            traceback="lbfgsb traceback",
        ),
        # success
        ("bard_good_start", "nm"): _result(
            params_history=[
                [1.0, 1.0, 1.0],
                [1.05, 1.0, 1.0],
                [0.7999999999999998, 1.1999999999999993, 1.0499999999999994],
                [0.08241056, 1.13303608, 2.34369519],
            ],
            fun_history=[41.68169586, 43.90748158, 23.92563745, 0.00821487730657897],
            time_history=[
                0.0,
                3.603900040616281e-05,
                0.0004506860022956971,
                0.00015319500016630627,
            ],
            batches_history=[0, 1, 2, 4],
            traceback=None,
        ),
        # error
        ("box_3d", "nm"): _result(
            params_history=[[0.0, 10.0, 20.0], [0.025, 10.0, 20.0], [0.0, 10.5, 20.0]],
            fun_history=[1031.15381061, 1031.17836473, 1030.15033678],
            time_history=[0.0, 5.73799989069812e-05, 0.00010679600018193014],
            batches_history=[0, 1, 2],
            traceback="some traceback",
        ),
        # error
        ("rosenbrock_good_start", "nm"): _result(
            params_history=[[-1.2, 1.0], [0.0, 0.0]],
            fun_history=[1.795769e6, 1e3],
            time_history=[0.0, 5.73799989069812e-04],
            batches_history=[0, 1],
            traceback="another traceback",
        ),
    }

    return problems, optimizers, results


# ====================================================================================
# Convergence report
# ====================================================================================

keys = ["stopping_criterion"]
stopping_criterion = ["x_and_y", "x_or_y", "x", "y"]
x_precision = [1e-4, 1e-6]
y_precision = [1e-4, 1e-6]
CONVERGENCE_REPORT_OPTIONS = [
    dict(zip(keys, value, strict=False))
    for value in product(stopping_criterion, x_precision, y_precision)
]


@pytest.mark.parametrize("options", CONVERGENCE_REPORT_OPTIONS)
def test_convergence_report(options, benchmark_example):
    problems, optimizers, results = benchmark_example

    df = convergence_report(problems=problems, results=results, **options)

    expected_columns = list(optimizers.keys()) + ["dimensionality"]
    assert df.shape == (len(problems), len(expected_columns))
    assert set(df.columns) == set(expected_columns)

    assert df["lbfgsb"].loc["box_3d"] == "failed"
    assert df["nm"].loc["box_3d"] == "error"


def test_convergence_report_values(benchmark_example):
    problems, _, results = benchmark_example
    df = convergence_report(problems=problems, results=results, stopping_criterion="y")

    assert df["nm"].to_dict() == {
        "bard_good_start": "success",
        "box_3d": "error",
        "rosenbrock_good_start": "error",
    }
    assert df["lbfgsb"].to_dict() == {
        "bard_good_start": "success",
        "box_3d": "failed",
        "rosenbrock_good_start": "error",
    }
    assert df["dimensionality"].to_dict() == {
        "bard_good_start": 3,
        "box_3d": 3,
        "rosenbrock_good_start": 2,
    }


# ====================================================================================
# Rank report
# ====================================================================================

keys = ["runtime_measure", "stopping_criterion"]
runtime_measure = ["n_evaluations", "walltime", "n_batches"]
RANK_REPORT_OPTIONS = [
    dict(zip(keys, value, strict=False))
    for value in product(runtime_measure, stopping_criterion)
]


@pytest.mark.parametrize("options", RANK_REPORT_OPTIONS)
def test_rank_report(options, benchmark_example):
    problems, optimizers, results = benchmark_example

    df = rank_report(problems=problems, results=results, **options)

    assert df.shape == (len(problems), len(optimizers) + 1)  # +1 for dimensionality
    assert set(df.columns) == set(optimizers.keys()) | {"dimensionality"}

    assert df["lbfgsb"].loc["box_3d"] == "failed"
    assert df["nm"].loc["box_3d"] == "error"


# ====================================================================================
# Traceback report
# ====================================================================================


@pytest.mark.parametrize("return_type", ["text", "markdown", "dict", "dataframe"])
def test_traceback_report(return_type, benchmark_example):
    problems, optimizers, results = benchmark_example
    n_failed_problems = 3

    report = traceback_report(
        problems=problems, results=results, return_type=return_type
    )

    if return_type in ["text", "dict"]:
        assert len(report) == n_failed_problems

    elif return_type == "markdown":
        for algorithm_name in optimizers:
            assert algorithm_name in report

    elif return_type == "dataframe":
        assert report.shape == (n_failed_problems, 2)
        assert list(report.index.names) == ["algorithm", "problem"]


def test_traceback_report_contents(benchmark_example):
    problems, _, results = benchmark_example

    as_dict = traceback_report(problems=problems, results=results, return_type="dict")
    assert as_dict == {
        ("rosenbrock_good_start", "lbfgsb"): "lbfgsb traceback",
        ("box_3d", "nm"): "some traceback",
        ("rosenbrock_good_start", "nm"): "another traceback",
    }

    as_text = traceback_report(problems=problems, results=results, return_type="text")
    assert as_text == ["lbfgsb traceback", "some traceback", "another traceback"]


def test_traceback_report_with_invalid_return_type_raises(benchmark_example):
    problems, _, results = benchmark_example
    with pytest.raises(ValueError, match="return_type bla is not supported"):
        traceback_report(  # ty: ignore[no-matching-overload]
            problems=problems, results=results, return_type="bla"
        )


def test_reports_with_real_benchmark_results():
    all_problems = get_benchmark_problems("example")
    problems = {name: all_problems[name] for name in ["box_3d", "bard_good_start"]}
    results = om.run_benchmark(
        problems, {"lbfgsb": om.algos.scipy_lbfgsb, "nm": om.algos.scipy_neldermead}
    )

    convergence = convergence_report(problems, results)
    assert set(convergence["lbfgsb"]) <= {"success", "failed"}

    ranks = rank_report(problems, results)
    assert ranks.shape == (2, 3)

    tracebacks = traceback_report(problems, results)
    assert len(tracebacks) == 0
