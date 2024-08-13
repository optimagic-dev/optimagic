import os
import warnings
from dataclasses import replace
from functools import cached_property
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import sqlalchemy as sql
from pybaum import tree_flatten, tree_unflatten
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
    CriterionEvaluationResult,
    CriterionEvaluationWithId,
    ExistenceStrategy,
    ExistenceStrategyLiteral,
    IterationHistory,
    MultiStartIterationHistory,
    ProblemInitialization,
    ProblemInitializationWithId,
    StepResult,
    StepResultWithId,
    StepType,
)
from optimagic.parameters.tree_registry import get_registry
from optimagic.typing import OptimizationType, OptimizationTypeLiteral, PyTree


class Logger:
    """A logger class that manages and retrieves optimization and exploration data.

    This class handles storing and retrieving iterations, steps, and problem
    initialization data using various stores.
    It provides methods to read iteration history, retrieve
    specific iterations, and handle multistart optimization history.

    Args:
        iteration_store: A non-updatable store for iteration data.
        step_store: An updatable store for step data.
        problem_store: An updatable store for problem initialization data.

    """

    def __init__(
        self,
        iteration_store: NonUpdatableKeyValueStore[
            CriterionEvaluationResult, CriterionEvaluationWithId
        ],
        step_store: UpdatableKeyValueStore[StepResult, StepResultWithId],
        problem_store: UpdatableKeyValueStore[
            ProblemInitialization, ProblemInitializationWithId
        ],
    ):
        self.step_store = step_store
        self.iteration_store = iteration_store
        self.problem_store = problem_store
        self._pytree_registry = get_registry(extended=True)

    @cached_property
    def _treedef(self) -> tuple[list[Any], Any]:
        return tree_flatten(self.read_start_params(), registry=self._pytree_registry)[
            -1
        ]

    def read_iteration(self, iteration: int) -> CriterionEvaluationWithId:
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
                last_row = self.iteration_store.select_last_rows(1)
                highest_rowid = last_row[0].rowid
            except IndexError as e:
                raise IndexError(
                    "Invalid iteration request, iteration store is empty"
                ) from e

            # iteration is negative here!
            assert highest_rowid is not None
            rowid = highest_rowid + iteration + 1

        row_list = self.iteration_store.select(rowid)

        if len(row_list) == 0:
            raise IndexError(f"Invalid iteration requested: {iteration}")
        else:
            data = row_list[0]

        params = tree_unflatten(
            self._treedef, data.params, registry=self._pytree_registry
        )

        return replace(data, params=params)

    def read_history(self) -> IterationHistory:
        """Read the entire iteration history from the iteration store.

        Returns:
            An `IterationHistory` object containing the parameters,
                criterion values, and runtimes.

        """
        raw_res = self.iteration_store.select()
        params_list = []
        criterion_list = []
        runtime_list = []
        for data in raw_res:
            if data.value is not None:
                params = tree_unflatten(
                    self._treedef, data.params, registry=self._pytree_registry
                )
                params_list.append(params)
                criterion_list.append(data.value)
                runtime_list.append(data.timestamp)

        times = np.array(runtime_list)
        times -= times[0]

        return IterationHistory(params_list, criterion_list, times)

    @staticmethod
    def _normalize_direction(
        direction: OptimizationType | OptimizationTypeLiteral,
    ) -> OptimizationType:
        if isinstance(direction, str):
            direction = OptimizationType(direction)
        return direction

    def _build_history_dataframe(self) -> pd.DataFrame:
        steps = self.step_store.to_df()
        raw_res = self.iteration_store.select()

        history: dict[str, list[Any]] = {
            "params": [],
            "criterion": [],
            "runtime": [],
            "step": [],
        }

        for data in raw_res:
            if data.value is not None:
                params = tree_unflatten(
                    self._treedef, data.params, registry=self._pytree_registry
                )
                history["params"].append(params)
                history["criterion"].append(data.value)
                history["runtime"].append(data.timestamp)
                history["step"].append(data.step)

        times = np.array(history["runtime"])
        times -= times[0]
        history["runtime"] = times.tolist()

        df = pd.DataFrame(history)
        df = df.merge(
            steps[[f"{self.step_store.primary_key}", "type"]],
            left_on="step",
            right_on=f"{self.step_store.primary_key}",
        )
        return df.drop(columns=f"{self.step_store.primary_key}")

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
        exploration: pd.DataFrame | None, optimization_type: OptimizationType
    ) -> IterationHistory | None:
        if exploration is not None:
            is_minimization = optimization_type is OptimizationType.MINIMIZE
            exploration = exploration.sort_values(
                by="criterion", ascending=is_minimization
            )
            exploration_dict = cast(dict[str, Any], exploration.to_dict(orient="list"))
            return IterationHistory(**exploration_dict)
        return exploration

    @staticmethod
    def _extract_best_history(
        histories: pd.DataFrame, optimization_type: OptimizationType
    ) -> tuple[IterationHistory, list[IterationHistory] | None]:
        groupby_step_criterion = histories["criterion"].groupby(level="step")

        if optimization_type is OptimizationType.MINIMIZE:
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
        self, direction: OptimizationType | OptimizationTypeLiteral
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
        return self.problem_store.select_last_rows(1)[0].params


class SQLiteConfig(SQLAlchemyConfig):
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
        self._handle_existing_database(path, if_database_exists)
        url = f"sqlite:///{path}"
        self._fast_logging = fast_logging
        super().__init__(url)

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
                warnings.warn(
                    f"Existing database file at {path} will be removed due to "
                    f"if_database_exists=ExistenceStrategy.REPLACE."
                )
                os.remove(path)

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


class SQLiteLogger(Logger):
    """A logger class that stores and manages optimization and exploration data using
    SQLite. It supports different strategies for handling existing tables and databases,
    such as extending, replacing, or raising an error.

    Args:
        path: The file path to the SQLite database.
        fast_logging: A boolean indicating whether to use fast logging mode.
        if_table_exists: Strategy for handling existing tables.
            Can be 'extend', 'replace', or 'raise'.
        if_database_exists: Strategy for handling the existing database file.
            Can be 'extend', 'replace', or 'raise'.

    """

    def __init__(
        self,
        path: str | Path,
        fast_logging: bool = False,
        if_table_exists: ExistenceStrategy
        | ExistenceStrategyLiteral = ExistenceStrategy.EXTEND,
        if_database_exists: ExistenceStrategy
        | ExistenceStrategyLiteral = ExistenceStrategy.EXTEND,
    ):
        if isinstance(if_table_exists, str):
            if_table_exists = ExistenceStrategy(if_table_exists)

        if isinstance(if_database_exists, str):
            if_database_exists = ExistenceStrategy(if_database_exists)

        db_config = SQLiteConfig(
            path, fast_logging=fast_logging, if_database_exists=if_database_exists
        )
        iteration_store = IterationStore(db_config, if_table_exists=if_table_exists)
        step_store = StepStore(db_config, if_table_exists=if_table_exists)
        problem_store = ProblemStore(db_config, if_table_exists=if_table_exists)
        super().__init__(iteration_store, step_store, problem_store)
