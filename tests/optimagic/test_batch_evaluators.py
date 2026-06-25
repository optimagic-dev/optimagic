import importlib.util
import itertools
import multiprocessing
import shutil
import subprocess
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import pytest

from optimagic.batch_evaluators import (
    executor_batch_evaluator,
    mpi_batch_evaluator,
    process_batch_evaluator,
)

_MPI_HELPER = Path(__file__).parent / "_mpi_helper.py"

_mpiexec = shutil.which("mpiexec")
_has_mpi4py = importlib.util.find_spec("mpi4py") is not None

batch_evaluators = ["joblib", "threading"]

n_core_list = [1, 2]

test_cases = list(itertools.product(batch_evaluators, n_core_list))


def double(x):
    return 2 * x


def buggy_func(x):  # noqa: ARG001
    raise AssertionError()


def add_x_and_y(x, y):
    return x + y


@pytest.mark.slow()
@pytest.mark.parametrize("batch_evaluator, n_cores", test_cases)
def test_batch_evaluator_without_exceptions(batch_evaluator, n_cores):
    batch_evaluator = process_batch_evaluator(batch_evaluator)

    calculated = batch_evaluator(
        func=double,
        arguments=list(range(10)),
        n_cores=n_cores,
    )

    expected = list(range(0, 20, 2))

    assert calculated == expected


@pytest.mark.slow()
@pytest.mark.parametrize("batch_evaluator, n_cores", test_cases)
def test_batch_evaluator_with_unhandled_exceptions(batch_evaluator, n_cores):
    batch_evaluator = process_batch_evaluator(batch_evaluator)
    with pytest.raises(AssertionError):
        batch_evaluator(
            func=buggy_func,
            arguments=list(range(10)),
            n_cores=n_cores,
            error_handling="raise",
        )


@pytest.mark.slow()
@pytest.mark.parametrize("batch_evaluator, n_cores", test_cases)
def test_batch_evaluator_with_handled_exceptions(batch_evaluator, n_cores):
    batch_evaluator = process_batch_evaluator(batch_evaluator)
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


@pytest.mark.slow()
@pytest.mark.parametrize("batch_evaluator, n_cores", test_cases)
def test_batch_evaluator_with_list_unpacking(batch_evaluator, n_cores):
    batch_evaluator = process_batch_evaluator(batch_evaluator)
    calculated = batch_evaluator(
        func=add_x_and_y,
        arguments=[(1, 2), (3, 4)],
        n_cores=n_cores,
        unpack_symbol="*",
    )
    expected = [3, 7]
    assert calculated == expected


@pytest.mark.slow()
@pytest.mark.parametrize("batch_evaluator, n_cores", test_cases)
def test_batch_evaluator_with_dict_unpacking(batch_evaluator, n_cores):
    batch_evaluator = process_batch_evaluator(batch_evaluator)
    calculated = batch_evaluator(
        func=add_x_and_y,
        arguments=[{"x": 1, "y": 2}, {"x": 3, "y": 4}],
        n_cores=n_cores,
        unpack_symbol="**",
    )
    expected = [3, 7]
    assert calculated == expected


def test_get_batch_evaluator_invalid_value():
    with pytest.raises(ValueError):
        process_batch_evaluator("bla")


def test_get_batch_evaluator_invalid_type():
    with pytest.raises(TypeError):
        process_batch_evaluator(3)


def test_get_batch_evaluator_with_callable():
    assert callable(process_batch_evaluator(lambda x: x))


def _make_closure_multiplier():
    factor = 3
    return lambda x: x * factor


def _spawn_context():
    return ProcessPoolExecutor(mp_context=multiprocessing.get_context("spawn"))


@pytest.mark.slow()
def test_executor_batch_evaluator_transports_closure_across_processes():
    """A closure survives spawn-process transport and yields ordered results."""
    closure = _make_closure_multiplier()
    batch_evaluator = executor_batch_evaluator(_spawn_context())

    calculated = batch_evaluator(func=closure, arguments=list(range(5)))

    assert calculated == [0, 3, 6, 9, 12]


def test_executor_batch_evaluator_with_threads_handles_closure():
    """A closure run through a thread pool yields ordered results."""
    closure = _make_closure_multiplier()
    batch_evaluator = executor_batch_evaluator(ThreadPoolExecutor(max_workers=2))

    calculated = batch_evaluator(func=closure, arguments=list(range(5)))

    assert calculated == [0, 3, 6, 9, 12]


@pytest.mark.slow()
def test_executor_batch_evaluator_continue_sets_traceback_string():
    """With error_handling='continue', a failing slot becomes a traceback string."""
    batch_evaluator = executor_batch_evaluator(_spawn_context())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calculated = batch_evaluator(
            func=buggy_func,
            arguments=list(range(4)),
            error_handling="continue",
        )

    assert all(isinstance(calc, str) for calc in calculated)


@pytest.mark.slow()
def test_executor_batch_evaluator_raise_reraises_exception():
    """With error_handling='raise', the original exception is reraised."""
    batch_evaluator = executor_batch_evaluator(_spawn_context())
    with pytest.raises(AssertionError):
        batch_evaluator(
            func=buggy_func,
            arguments=list(range(4)),
            error_handling="raise",
        )


def test_process_batch_evaluator_resolves_mpi():
    """The string 'mpi' resolves to the mpi batch evaluator."""
    assert process_batch_evaluator("mpi") is mpi_batch_evaluator


def test_mpi_batch_evaluator_without_mpi4py_raises_clear_error():
    """Calling the mpi evaluator without mpi4py installed raises a clear ImportError."""
    try:
        import mpi4py  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("mpi4py is installed; cannot test the missing-dependency path.")

    with pytest.raises(ImportError, match="optimagic\\[mpi\\]"):
        mpi_batch_evaluator(func=double, arguments=[1, 2])


@pytest.mark.mpi()
@pytest.mark.skipif(
    _mpiexec is None or not _has_mpi4py,
    reason="requires mpiexec on PATH and an importable mpi4py.",
)
def test_mpi_batch_evaluator_integration():
    """The mpi evaluator fans a closure out over MPI worker ranks, in input order.

    Launches the helper under ``mpiexec -n 3 python -m mpi4py.futures`` so that
    real worker ranks exist, proving cloudpickle-over-MPI transport of a locally
    defined closure works end to end.
    """
    result = subprocess.run(
        [
            _mpiexec,
            "-n",
            "3",
            sys.executable,
            "-m",
            "mpi4py.futures",
            str(_MPI_HELPER),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )

    assert result.returncode == 0, (
        f"mpiexec run failed (returncode {result.returncode}).\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def test_executor_batch_evaluator_is_exported_at_top_level():
    """`executor_batch_evaluator` is reachable from the top-level namespace."""
    import optimagic

    assert optimagic.executor_batch_evaluator is executor_batch_evaluator
