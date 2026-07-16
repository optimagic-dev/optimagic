"""A collection of batch evaluators for process based parallelism.

All batch evaluators have the same interface and any function with the same interface
can be used used as batch evaluator in optimagic.

"""

from joblib import Parallel, delayed

try:
    from pathos.pools import ProcessPool

    pathos_is_available = True
except ImportError:
    pathos_is_available = False

import threading
from concurrent.futures import Executor
from typing import Any, Callable, Literal, TypeVar, cast

import cloudpickle

from optimagic import deprecations
from optimagic.config import DEFAULT_N_CORES as N_CORES
from optimagic.decorators import catch, unpack
from optimagic.typing import BatchEvaluator, BatchEvaluatorLiteral, ErrorHandling

T = TypeVar("T")


def pathos_mp_batch_evaluator(
    func: Callable[..., T],
    arguments: list[Any],
    *,
    n_cores: int = N_CORES,
    error_handling: ErrorHandling
    | Literal["raise", "continue"] = ErrorHandling.CONTINUE,
    unpack_symbol: Literal["*", "**"] | None = None,
) -> list[T]:
    """Batch evaluator based on pathos.multiprocess.ProcessPool.

    This uses a patched but older version of python multiprocessing that replaces
    pickling with dill and can thus handle decorated functions.

    Args:
        func (Callable): The function that is evaluated.
        arguments (Iterable): Arguments for the functions. Their interperation
            depends on the unpack argument.
        n_cores (int): Number of cores used to evaluate the function in parallel.
            Value below one are interpreted as one. If only one core is used, the
            batch evaluator disables everything that could cause problems, i.e. in that
            case func and arguments are never pickled and func is executed in the main
            process.
        error_handling (str): Can take the values "raise" (raise the error and stop all
            tasks as soon as one task fails) and "continue" (catch exceptions and set
            the traceback of the raised exception.
            KeyboardInterrupt and SystemExit are always raised.
        unpack_symbol (str or None). Can be "**", "*" or None. If None, func just takes
            one argument. If "*", the elements of arguments are positional arguments for
            func. If "**", the elements of arguments are keyword arguments for func.


    Returns:
        list: The function evaluations.

    """
    if not pathos_is_available:
        raise NotImplementedError(
            "To use the pathos_mp_batch_evaluator, install pathos with "
            "conda install -c conda-forge pathos."
        )

    _check_inputs(func, arguments, n_cores, error_handling, unpack_symbol)
    n_cores = int(n_cores)

    reraise = error_handling in [
        "raise",
        ErrorHandling.RAISE,
        ErrorHandling.RAISE_STRICT,
    ]

    @unpack(symbol=unpack_symbol)
    @catch(default="__traceback__", reraise=reraise)
    def internal_func(*args: Any, **kwargs: Any) -> T:
        return func(*args, **kwargs)

    if n_cores <= 1:
        res = [internal_func(arg) for arg in arguments]
    else:
        p = ProcessPool(nodes=n_cores)
        try:
            res = p.map(internal_func, arguments)
        except Exception as e:
            p.terminate()
            raise e

    return res


def joblib_batch_evaluator(
    func: Callable[..., T],
    arguments: list[Any],
    *,
    n_cores: int = N_CORES,
    error_handling: ErrorHandling
    | Literal["raise", "continue"] = ErrorHandling.CONTINUE,
    unpack_symbol: Literal["*", "**"] | None = None,
) -> list[T]:
    """Batch evaluator based on joblib's Parallel.

    Args:
        func (Callable): The function that is evaluated.
        arguments (Iterable): Arguments for the functions. Their interperation
            depends on the unpack argument.
        n_cores (int): Number of cores used to evaluate the function in parallel.
            Value below one are interpreted as one. If only one core is used, the
            batch evaluator disables everything that could cause problems, i.e. in that
            case func and arguments are never pickled and func is executed in the main
            process.
        error_handling (str): Can take the values "raise" (raise the error and stop all
            tasks as soon as one task fails) and "continue" (catch exceptions and set
            the output of failed tasks to the traceback of the raised exception.
            KeyboardInterrupt and SystemExit are always raised.
        unpack_symbol (str or None). Can be "**", "*" or None. If None, func just takes
            one argument. If "*", the elements of arguments are positional arguments for
            func. If "**", the elements of arguments are keyword arguments for func.


    Returns:
        list: The function evaluations.

    """
    _check_inputs(func, arguments, n_cores, error_handling, unpack_symbol)
    n_cores = int(n_cores) if int(n_cores) >= 2 else 1

    reraise = error_handling in [
        "raise",
        ErrorHandling.RAISE,
        ErrorHandling.RAISE_STRICT,
    ]

    @unpack(symbol=unpack_symbol)
    @catch(default="__traceback__", reraise=reraise)
    def internal_func(*args: Any, **kwargs: Any) -> T:
        return func(*args, **kwargs)

    if n_cores == 1:
        res = [internal_func(arg) for arg in arguments]
    else:
        res = Parallel(n_jobs=n_cores)(delayed(internal_func)(arg) for arg in arguments)

    return res


def threading_batch_evaluator(
    func: Callable[..., T],
    arguments: list[Any],
    *,
    n_cores: int = N_CORES,
    error_handling: ErrorHandling
    | Literal["raise", "continue"] = ErrorHandling.CONTINUE,
    unpack_symbol: Literal["*", "**"] | None = None,
) -> list[T]:
    """Batch evaluator based on Python's threading.

    Args:
        func (Callable): The function that is evaluated.
        arguments (Iterable): Arguments for the functions. Their interperation
            depends on the unpack argument.
        n_cores (int): Number of threads used to evaluate the function in parallel.
            Value below one are interpreted as one.
        error_handling (str): Can take the values "raise" (raise the error and stop all
            tasks as soon as one task fails) and "continue" (catch exceptions and set
            the output of failed tasks to the traceback of the raised exception.
            KeyboardInterrupt and SystemExit are always raised.
        unpack_symbol (str or None). Can be "**", "*" or None. If None, func just takes
            one argument. If "*", the elements of arguments are positional arguments for
            func. If "**", the elements of arguments are keyword arguments for func.

    Returns:
        list: The function evaluations.

    """
    _check_inputs(func, arguments, n_cores, error_handling, unpack_symbol)
    n_cores = int(n_cores) if int(n_cores) >= 2 else 1

    reraise = error_handling in [
        "raise",
        ErrorHandling.RAISE,
        ErrorHandling.RAISE_STRICT,
    ]

    @unpack(symbol=unpack_symbol)
    @catch(default="__traceback__", reraise=reraise)
    def internal_func(*args: Any, **kwargs: Any) -> T:
        return func(*args, **kwargs)

    if n_cores == 1:
        res = [internal_func(arg) for arg in arguments]
    else:
        results = [None] * len(arguments)
        threads = []
        errors = []
        error_lock = threading.Lock()

        def thread_func(index: int, arg: Any) -> None:
            try:
                results[index] = internal_func(arg)
            except Exception as e:
                with error_lock:
                    errors.append(e)

        for i, arg in enumerate(arguments):
            thread = threading.Thread(target=thread_func, args=(i, arg))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        if errors:
            raise errors[0]

        res = cast(list[T], results)
    return res


class _CloudpickleTask:
    """A picklable wrapper that transports any callable via cloudpickle.

    Plain pickle cannot transport closures, lambdas, or locally defined functions,
    so an executor that pickles its task (any process-based executor under the
    "spawn" start method, and ``mpi4py``) would reject such a criterion. This
    wrapper serializes the wrapped callable with cloudpickle eagerly and stores
    only the resulting bytes, which plain pickle moves without trouble. The bytes
    are deserialized on first call in the worker.

    """

    def __init__(self, func: Callable[..., Any]) -> None:
        self._payload = cloudpickle.dumps(func)

    def __call__(self, arg: Any) -> Any:
        return cloudpickle.loads(self._payload)(arg)


def executor_batch_evaluator(
    executor: Executor,
) -> BatchEvaluator:
    """Build a batch evaluator backed by a ``concurrent.futures.Executor``.

    This adapts any executor (``ProcessPoolExecutor``, ``ThreadPoolExecutor``,
    ``mpi4py.futures.MPIPoolExecutor``, …) to optimagic's batch-evaluator
    interface. The criterion is wrapped so that closures and locally defined
    functions survive the executor's pickle-based transport, and results are
    returned in the order of ``arguments``.

    Args:
        executor: The executor that runs the function evaluations. The executor
            governs concurrency, so the ``n_cores`` argument of the returned
            batch evaluator is ignored.

    Returns:
        A batch evaluator with the standard optimagic interface.

    """

    def batch_evaluator(
        func: Callable[..., T],
        arguments: list[Any],
        *,
        n_cores: int = N_CORES,  # noqa: ARG001
        error_handling: ErrorHandling
        | Literal["raise", "continue"] = ErrorHandling.CONTINUE,
        unpack_symbol: Literal["*", "**"] | None = None,
    ) -> list[T]:
        """Evaluate ``func`` on every element of ``arguments`` via the executor.

        Concurrency is governed by the executor, so ``n_cores`` is ignored.

        """
        _check_inputs(func, arguments, n_cores, error_handling, unpack_symbol)

        reraise = error_handling in [
            "raise",
            ErrorHandling.RAISE,
            ErrorHandling.RAISE_STRICT,
        ]

        @unpack(symbol=unpack_symbol)
        @catch(default="__traceback__", reraise=reraise)
        def internal_func(*args: Any, **kwargs: Any) -> T:
            return func(*args, **kwargs)

        task = _CloudpickleTask(internal_func)
        return list(executor.map(task, arguments))

    return cast(BatchEvaluator, batch_evaluator)


_MPI_EXECUTOR: Executor | None = None


def mpi_batch_evaluator(
    func: Callable[..., T],
    arguments: list[Any],
    *,
    n_cores: int = N_CORES,
    error_handling: ErrorHandling
    | Literal["raise", "continue"] = ErrorHandling.CONTINUE,
    unpack_symbol: Literal["*", "**"] | None = None,
) -> list[T]:
    """Batch evaluator based on ``mpi4py.futures.MPIPoolExecutor``.

    This is the correct way to run optimagic under MPI: a single optimizer runs
    on the driver rank while the worker ranks are parked by the
    ``python -m mpi4py.futures`` launcher and pick up batched criterion
    evaluations. The launcher is a precondition — start the program with, e.g.,
    ``srun python -m mpi4py.futures your_script.py`` so that worker ranks exist.

    The ``MPIPoolExecutor`` is created on the first call and cached at module
    level, so every batch reuses the same pool of worker ranks. ``n_cores`` is
    ignored; the MPI launch governs how many workers are available.

    Args:
        func (Callable): The function that is evaluated.
        arguments (Iterable): Arguments for the functions. Their interpretation
            depends on the ``unpack_symbol`` argument.
        n_cores (int): Ignored. MPI worker ranks govern concurrency.
        error_handling (str): Can take the values "raise" (raise the error and
            stop all tasks as soon as one task fails) and "continue" (catch
            exceptions and set the output of failed tasks to the traceback of the
            raised exception). KeyboardInterrupt and SystemExit are always raised.
        unpack_symbol (str or None): Can be "**", "*" or None. If None, func just
            takes one argument. If "*", the elements of arguments are positional
            arguments for func. If "**", the elements of arguments are keyword
            arguments for func.

    Returns:
        list: The function evaluations.

    """
    try:
        from mpi4py import MPI
        from mpi4py.futures import MPIPoolExecutor
    except ImportError as e:
        raise ImportError(
            "The mpi batch evaluator requires mpi4py, which is an optional "
            "dependency. Install it with the `optimagic[mpi]` extra, e.g. "
            "`pip install optimagic[mpi]`."
        ) from e

    global _MPI_EXECUTOR  # noqa: PLW0603
    if _MPI_EXECUTOR is None:
        # Use cloudpickle for MPI serialization so closures and locally defined
        # criteria survive transport to the worker ranks.
        MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads)
        executor = MPIPoolExecutor()
        # ``num_workers`` bootstraps the pool and reports how many worker ranks
        # it found. Zero means the program was started without the
        # mpi4py.futures launcher and no worker ranks could be reached.
        if executor.num_workers == 0:
            executor.shutdown(wait=False)
            raise RuntimeError(
                "No MPI worker ranks are available. Launch the program with the "
                "mpi4py.futures launcher so that worker ranks exist, e.g. "
                "`mpiexec -n <N> python -m mpi4py.futures your_script.py` or "
                "`srun python -m mpi4py.futures your_script.py`."
            )
        _MPI_EXECUTOR = executor

    return executor_batch_evaluator(_MPI_EXECUTOR)(
        func,
        arguments,
        n_cores=n_cores,
        error_handling=error_handling,
        unpack_symbol=unpack_symbol,
    )


def _check_inputs(
    func: Callable[..., T],
    arguments: list[Any],
    n_cores: int,
    error_handling: ErrorHandling | Literal["raise", "continue"],
    unpack_symbol: Literal["*", "**"] | None,
) -> None:
    if not callable(func):
        raise TypeError("func must be callable.")

    try:
        arguments = list(arguments)
    except Exception as e:
        raise ValueError("arguments must be list like.") from e

    try:
        int(n_cores)
    except Exception as e:
        raise ValueError("n_cores must be an integer.") from e

    if unpack_symbol not in (None, "*", "**"):
        raise ValueError(
            f"unpack_symbol must be None, '*' or '**', not {unpack_symbol}"
        )

    if error_handling not in [
        "raise",
        "continue",
        ErrorHandling.RAISE,
        ErrorHandling.CONTINUE,
        ErrorHandling.RAISE_STRICT,
    ]:
        raise ValueError(
            "error_handling must be 'raise' or 'continue' or ErrorHandling not "
            f"{error_handling}"
        )


def process_batch_evaluator(
    batch_evaluator: BatchEvaluatorLiteral | BatchEvaluator = "joblib",
) -> BatchEvaluator:
    if batch_evaluator is None:
        deprecations.throw_none_valued_batch_evaluator_warning()
        batch_evaluator = "joblib"

    if callable(batch_evaluator):
        out = batch_evaluator
    elif isinstance(batch_evaluator, str):
        if batch_evaluator == "joblib":
            out = cast(BatchEvaluator, joblib_batch_evaluator)
        elif batch_evaluator == "pathos":
            out = cast(BatchEvaluator, pathos_mp_batch_evaluator)
        elif batch_evaluator == "threading":
            out = cast(BatchEvaluator, threading_batch_evaluator)
        elif batch_evaluator == "mpi":
            out = cast(BatchEvaluator, mpi_batch_evaluator)
        else:
            raise ValueError(
                "Invalid batch evaluator requested. Currently only 'pathos', "
                "'joblib', 'threading', and 'mpi' are supported."
            )
    else:
        raise TypeError("batch_evaluator must be a callable or string.")

    return out
