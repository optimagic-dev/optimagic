from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic, Type, TypeVar, cast

import numpy as np
import pandas as pd
import sqlalchemy as sql
from sqlalchemy.engine import Engine

from optimagic.logging.base import (
    NonUpdatableKeyValueStore,
    UpdatableKeyValueStore,
)
from optimagic.logging.sqlalchemy import (
    IterationStore,
    ProblemStore,
    SQLAlchemyConfig,
    StepStore,
)
from optimagic.logging.types import (
    ExistenceStrategy,
    ExistenceStrategyLiteral,
    IterationState,
    IterationStateWithId,
    ProblemInitialization,
    ProblemInitializationWithId,
    StepResult,
    StepResultWithId,
    StepType,
)
from optimagic.typing import (
    Direction,
    DirectionLiteral,
    IterationHistory,
    MultiStartIterationHistory,
    PyTree,
)


class LogOptions:
    """Base class for defining different log options.

    Serves as a registry for implemented option classes for better discoverability.

    """

    _subclass_registry: list[Type[LogOptions]] = []

    def __init_subclass__(
        cls: Type[LogOptions], abstract: bool = False, **kwargs: dict[Any, Any]
    ):
        if not abstract:
            LogOptions._subclass_registry.append(cls)
        super().__init_subclass__(**kwargs)

    @classmethod
    def available_option_types(cls) -> list[Type[LogOptions]]:
        return cls._subclass_registry


_LogOptionsType = TypeVar("_LogOptionsType", bound=LogOptions)


class LogReader(Generic[_LogOptionsType], ABC):
    """A class that manages the retrieving of optimization and exploration data.

    This class exposes methods to retrieve optimization logging data from stores.

    """

    _step_store: UpdatableKeyValueStore[StepResult, StepResultWithId]
    _iteration_store: NonUpdatableKeyValueStore[IterationState, IterationStateWithId]
    _problem_store: UpdatableKeyValueStore[
        ProblemInitialization, ProblemInitializationWithId
    ]

    @property
    def problem_df(self) -> pd.DataFrame:
        return self._problem_store.to_df()

    @classmethod
    def from_options(cls, log_options: LogOptions) -> LogReader[_LogOptionsType]:
        log_reader_class = _LOG_OPTION_LOG_READER_REGISTRY.get(type(log_options), None)

        if log_reader_class is None:
            raise ValueError(
                f"No LogReader implementation found for type "
                f"{type(log_options)}. Available option types: "
                f"\n {list(_LOG_OPTION_LOG_READER_REGISTRY.keys())}"
            )

        return log_reader_class._create(log_options)

    @classmethod
    @abstractmethod
    def _create(cls, log_options: _LogOptionsType) -> LogReader[_LogOptionsType]:
        pass

    def read_iteration(self, iteration: int) -> IterationStateWithId:
        """Read a specific iteration from the iteration store.

        Args:
            iteration: The iteration number to read. Negative values read from the end.

        Returns:
            A `CriterionEvaluationWithId` object containing the iteration data.

        Raises:
            IndexError: If the iteration is invalid or the store is empty.

        """
        if iteration >= 0:
            rowid = iteration + 1
        else:
            try:
                last_row = self._iteration_store.select_last_rows(1)
                highest_rowid = last_row[0].rowid
            except IndexError as e:
                raise IndexError(
                    "Invalid iteration request, iteration store is empty"
                ) from e

            # iteration is negative here!
            assert highest_rowid is not None
            rowid = highest_rowid + iteration + 1

        row_list = self._iteration_store.select(rowid)

        if len(row_list) == 0:
            raise IndexError(f"Invalid iteration requested: {iteration}")
        else:
            data = row_list[0]

        return data

    def read_history(self) -> IterationHistory:
        """Read the entire iteration history from the iteration store.

        Returns:
            An `IterationHistory` object containing the parameters,
                criterion values, and runtimes.

        """
        raw_res = self._iteration_store.select()
        params_list = []
        criterion_list = []
        runtime_list = []
        for data in raw_res:
            if data.scalar_fun is not None:
                params_list.append(data.params)
                criterion_list.append(data.scalar_fun)
                runtime_list.append(data.timestamp)

        times = np.array(runtime_list)
        times -= times[0]

        return IterationHistory(params_list, criterion_list, times)

    @staticmethod
    def _normalize_direction(
        direction: Direction | DirectionLiteral,
    ) -> Direction:
        if isinstance(direction, str):
            direction = Direction(direction)
        return direction

    def _build_history_dataframe(self) -> pd.DataFrame:
        steps = self._step_store.to_df()
        raw_res = self._iteration_store.select()

        history: dict[str, list[Any]] = {
            "params": [],
            "fun": [],
            "time": [],
            "step": [],
        }

        for data in raw_res:
            if data.scalar_fun is not None:
                history["params"].append(data.params)
                history["fun"].append(data.scalar_fun)
                history["time"].append(data.timestamp)
                history["step"].append(data.step)

        times = np.array(history["time"])
        times -= times[0]
        history["time"] = times.tolist()

        df = pd.DataFrame(history)
        df = df.merge(
            steps[[f"{self._step_store.primary_key}", "type"]],
            left_on="step",
            right_on=f"{self._step_store.primary_key}",
        )
        return df.drop(columns=f"{self._step_store.primary_key}")

    @staticmethod
    def _split_exploration_and_optimization(
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame | None, pd.DataFrame]:
        exploration = df.query(f"type == '{StepType.EXPLORATION.value}'").drop(
            columns=["step", "type"]
        )
        histories = df.query(f"type == '{StepType.OPTIMIZATION.value}'")
        histories = histories.drop(columns="type").set_index("step", append=True)

        return None if exploration.empty else exploration, histories

    @staticmethod
    def _sort_exploration(
        exploration: pd.DataFrame | None, optimization_type: Direction
    ) -> IterationHistory | None:
        if exploration is not None:
            is_minimization = optimization_type is Direction.MINIMIZE
            exploration = exploration.sort_values(by="fun", ascending=is_minimization)
            exploration_dict = cast(dict[str, Any], exploration.to_dict(orient="list"))
            return IterationHistory(**exploration_dict)
        return exploration

    @staticmethod
    def _extract_best_history(
        histories: pd.DataFrame, optimization_type: Direction
    ) -> tuple[IterationHistory, list[IterationHistory] | None]:
        groupby_step_criterion = histories["fun"].groupby(level="step")

        if optimization_type is Direction.MINIMIZE:
            best_idx = groupby_step_criterion.min().idxmin()
        else:
            best_idx = groupby_step_criterion.max().idxmax()

        remaining_indices = (
            histories.index.get_level_values("step").unique().difference([best_idx])
        )

        best_history: pd.DataFrame | pd.Series[Any] = histories.xs(
            best_idx, level="step"
        )

        def _to_dict(pandas_obj: pd.DataFrame | pd.Series) -> dict[str, Any]:  # type:ignore
            if isinstance(pandas_obj, pd.DataFrame):
                result = pandas_obj.to_dict(orient="list")
            else:
                result = best_history.to_dict()
            return cast(dict[str, Any], result)

        best_history_dict = _to_dict(best_history)
        local_histories = [
            _to_dict(histories.xs(idx, level="step")) for idx in remaining_indices
        ]
        if len(local_histories) == 0:
            remaining_histories = None
        else:
            remaining_histories = [
                IterationHistory(**history_dict) for history_dict in local_histories
            ]

        return IterationHistory(**best_history_dict), remaining_histories

    def read_multistart_history(
        self, direction: Direction | DirectionLiteral
    ) -> MultiStartIterationHistory:
        """Read and the multistart optimization history.

        Args:
            direction: The optimization direction, either as an enum or string.

        Returns:
            A `MultiStartIterationHistory` object containing the best history,
                local histories, and exploration history.

        """
        optimization_type = self._normalize_direction(direction)
        history_df = self._build_history_dataframe()
        exploration, optimization_history = self._split_exploration_and_optimization(
            history_df
        )
        exploration_history = self._sort_exploration(exploration, optimization_type)
        best_history, remaining_histories = self._extract_best_history(
            optimization_history, optimization_type
        )

        return MultiStartIterationHistory(
            best_history,
            local_histories=remaining_histories,
            exploration=exploration_history,
        )

    def read_start_params(self) -> PyTree:
        """Read the start parameters form the problem store.

        Returns:
            A pytree object representing the start parameter.

        """
        return self._problem_store.select(1)[0].params


_LogReaderType = TypeVar("_LogReaderType", bound=LogReader[Any])


class LogStore(Generic[_LogOptionsType, _LogReaderType], ABC):
    """A class that manages the logging of optimization and exploration data.

    This class handles storing iterations, steps, and problem
    initialization data using various stores.

    Args:
        iteration_store: A non-updatable store for iteration data.
        step_store: An updatable store for step data.
        problem_store: An updatable store for problem initialization data.

    """

    def __init__(
        self,
        iteration_store: NonUpdatableKeyValueStore[
            IterationState, IterationStateWithId
        ],
        step_store: UpdatableKeyValueStore[StepResult, StepResultWithId],
        problem_store: UpdatableKeyValueStore[
            ProblemInitialization, ProblemInitializationWithId
        ],
    ):
        self.step_store = step_store
        self.iteration_store = iteration_store
        self.problem_store = problem_store

    @classmethod
    def from_options(
        cls, log_options: LogOptions
    ) -> LogStore[_LogOptionsType, _LogReaderType]:
        logger_class = _LOG_OPTION_LOGGER_REGISTRY.get(type(log_options), None)

        if logger_class is None:
            raise ValueError(
                f"No Logger implementation found for type "
                f"{type(log_options)}. Available option types: "
                f"\n {list(_LOG_OPTION_LOGGER_REGISTRY.keys())}"
            )

        return logger_class.create(log_options)

    @classmethod
    @abstractmethod
    def create(
        cls, log_options: _LogOptionsType
    ) -> LogStore[_LogOptionsType, _LogReaderType]:
        pass


class SQLiteLogOptions(SQLAlchemyConfig, LogOptions):
    """Configuration class for setting up an SQLite database with SQLAlchemy.

    This class extends the `SQLAlchemyConfig` class to configure an SQLite database.
    It handles the creation of the database engine, manages database files,
    and applies various optimizations for logging performance.

    Args:
        path (str | Path): The file path to the SQLite database.
        fast_logging (bool): A boolean that determines if “unsafe” settings are used to
            speed up write processes to the database. This should only be used for very
            short running criterion functions where the main purpose of the log
            is a real-time dashboard, and it would not be catastrophic to get
            a corrupted database in case of a sudden system shutdown.
            If one evaluation of the criterion function (and gradient if applicable)
            takes more than 100 ms, the logging overhead is negligible.
        if_database_exists (ExistenceStrategy): Strategy for handling an existing
            database file. One of “extend”, “replace”, “raise”.

    """

    def __init__(
        self,
        path: str | Path,
        fast_logging: bool = True,
        if_database_exists: ExistenceStrategy
        | ExistenceStrategyLiteral = ExistenceStrategy.RAISE,
    ):
        url = f"sqlite:///{path}"
        self._fast_logging = fast_logging
        self._path = path
        if isinstance(if_database_exists, str):
            if_database_exists = ExistenceStrategy(if_database_exists)
        self.if_database_exists = if_database_exists
        super().__init__(url)

    @property
    def path(self) -> str | Path:
        return self._path

    def create_engine(self) -> Engine:
        engine = sql.create_engine(self.url)
        self._configure_engine(engine)
        return engine

    def _configure_engine(self, engine: Engine) -> None:
        """Configure the sqlite engine.

        The two functions that configure the emission of the `begin` statement are taken
        from the sqlalchemy documentation the documentation:
        https://tinyurl.com/u9xea5z
        and are
        the recommended way of working around a bug in the pysqlite driver.

        The other function speeds up the write process. If fast_logging is False, it
        does so using only completely safe optimizations. Of fast_logging is True,
        it also uses unsafe optimizations.

        """

        @sql.event.listens_for(engine, "connect")
        def do_connect(dbapi_connection: Any, connection_record: Any) -> None:  # noqa: ARG001
            # disable pysqlite's emitting of the BEGIN statement entirely.
            # also stops it from emitting COMMIT before absolutely necessary.
            dbapi_connection.isolation_level = None

        @sql.event.listens_for(engine, "begin")
        def do_begin(conn: Any) -> None:
            # emit our own BEGIN
            conn.exec_driver_sql("BEGIN DEFERRED")

        @sql.event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_connection: Any, connection_record: Any) -> None:  # noqa: ARG001
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode = WAL")
            if self._fast_logging:
                cursor.execute("PRAGMA synchronous = OFF")
            else:
                cursor.execute("PRAGMA synchronous = NORMAL")
            cursor.close()


class SQLiteLogReader(LogReader[SQLiteLogOptions]):
    """A class that manages the retrieving of optimization and exploration data from a
    SQLite database.

    This class exposes methods to retrieve optimization logging data from stores.

    Args:
            path (str | Path): The path to the SQLite database file.

    """

    def __init__(self, path: str | Path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"No file found at {path=}")

        log_options = SQLiteLogOptions(
            path, fast_logging=True, if_database_exists=ExistenceStrategy.EXTEND
        )
        self._iteration_store = IterationStore(log_options)
        self._step_store = StepStore(log_options)
        self._problem_store = ProblemStore(log_options)

    @classmethod
    def _create(cls, log_options: SQLiteLogOptions) -> SQLiteLogReader:
        """Create an instance of SQLiteLogReader using the provided log options.

        Args:
            log_options (SQLiteLogOptions): Configuration options for the SQLite log.

        Returns:
            SQLiteLogReader: An instance of SQLiteLogReader initialized with the
            provided log options.

        """
        return cls(log_options.path)


class _SQLiteLogStore(LogStore[SQLiteLogOptions, SQLiteLogReader]):
    """A logger class that stores and manages optimization and exploration data using
    SQLite.

    It supports different strategies for handling existing databases, such as extending,
    replacing, or raising an error.

    """

    @staticmethod
    def _handle_existing_database(
        path: str | Path,
        if_database_exists: ExistenceStrategy | ExistenceStrategyLiteral,
    ) -> None:
        if isinstance(if_database_exists, str):
            if_database_exists = ExistenceStrategy(if_database_exists)
        database_exists = os.path.exists(path)
        if database_exists:
            if if_database_exists is ExistenceStrategy.RAISE:
                raise FileExistsError(
                    f"The database at {path} already exists. To reuse and extend "
                    f"the existing database, set if_database_exists to "
                    f"ExistenceStrategy.EXTEND."
                )
            elif if_database_exists is ExistenceStrategy.REPLACE:
                try:
                    os.remove(path)
                except PermissionError as e:
                    msg = (
                        f"Failed to remove file {path}. "
                        f"In particular, this can happen on Windows "
                        f"machines, when a different process is accessing the file, "
                        f"which results in a PermissionError. In this case, delete"
                        f"the file manually."
                    )
                    raise RuntimeError(msg) from e

    @classmethod
    def create(cls, log_options: SQLiteLogOptions) -> _SQLiteLogStore:
        cls._handle_existing_database(log_options.path, log_options.if_database_exists)

        iteration_store = IterationStore(log_options)
        step_store = StepStore(log_options)
        problem_store = ProblemStore(log_options)
        return cls(iteration_store, step_store, problem_store)


_LOG_OPTION_LOGGER_REGISTRY: dict[Type[LogOptions], Type[LogStore[Any, Any]]] = {
    SQLiteLogOptions: _SQLiteLogStore
}
_LOG_OPTION_LOG_READER_REGISTRY: dict[Type[LogOptions], Type[LogReader[Any]]] = {
    SQLiteLogOptions: SQLiteLogReader
}
