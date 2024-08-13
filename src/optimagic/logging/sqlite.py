import logging
import os.path
import warnings
from pathlib import Path
from typing import Any, cast

import sqlalchemy as sql
from sqlalchemy import (
    Column,
    Integer,
    PickleType,
    String,
)
from sqlalchemy.engine.base import Engine

from optimagic.logging.base import RobustPickler
from optimagic.logging.sqlalchemy import (
    SQLAlchemyConfig,
    SQLAlchemySimpleStore,
    SQLAlchemyTableStore,
    TableConfig,
)
from optimagic.logging.types import (
    CriterionEvaluationResult,
    CriterionEvaluationWithId,
    ExistenceStrategy,
    ExistenceStrategyLiteral,
    ProblemInitialization,
    ProblemInitializationWithId,
    StepResult,
    StepResultWithId,
)

logger = logging.getLogger(__name__)


class SQLiteConfig(SQLAlchemyConfig):
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
        if database_exists and if_database_exists is ExistenceStrategy.RAISE:
            raise FileExistsError(
                "If you want to reuse and extend the existing "
                "database, provide "
                "if_database_exists=ExistenceStrategy.EXTEND"
            )
        elif if_database_exists is ExistenceStrategy.REPLACE:
            warnings.warn(
                f"Due to if_database_exists=ExistenceStrategy.EXTEND, will"
                f"remove existing database file at {path}"
            )
            os.remove(path)

    def _create_engine(self) -> Engine:
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


class IterationStore(
    SQLAlchemySimpleStore[CriterionEvaluationResult, CriterionEvaluationWithId]
):
    _TABLE_NAME = "optimization_iterations"
    _PRIMARY_KEY = "rowid"

    def __init__(
        self,
        db_config: SQLiteConfig,
        if_table_exists: ExistenceStrategy = ExistenceStrategy.EXTEND,
    ):
        super().__init__(
            self._TABLE_NAME,
            self._PRIMARY_KEY,
            db_config,
            CriterionEvaluationResult,
            CriterionEvaluationWithId,
            if_table_exists=if_table_exists,
        )


class StepStore(SQLAlchemyTableStore[StepResult, StepResultWithId]):
    _TABLE_NAME = "steps"
    _PRIMARY_KEY = "rowid"

    def __init__(
        self,
        db_config: SQLiteConfig,
        existence_strategy: ExistenceStrategy = ExistenceStrategy.EXTEND,
    ):
        columns = [
            Column(self._PRIMARY_KEY, Integer, primary_key=True, autoincrement=True),
            Column("type", String),  # e.g. optimization
            Column("status", String),  # e.g. running
            Column("n_iterations", Integer),  # optional
            Column("name", String),  # e.g. "optimization-1", "exploration", not unique
        ]

        table_config = TableConfig(
            self._TABLE_NAME,
            cast(list[Column[Any]], columns),
            self._PRIMARY_KEY,
            existence_strategy,
        )

        super().__init__(
            table_config,
            db_config,
            StepResult,
            StepResultWithId,
        )


class ProblemStore(
    SQLAlchemyTableStore[ProblemInitialization, ProblemInitializationWithId]
):
    _TABLE_NAME = "optimization_problem"
    _PRIMARY_KEY = "rowid"

    def __init__(
        self,
        db_config: SQLiteConfig,
        existence_strategy: ExistenceStrategy = ExistenceStrategy.EXTEND,
    ):
        columns = [
            Column(self._PRIMARY_KEY, Integer, primary_key=True, autoincrement=True),
            Column("direction", String),
            Column("params", PickleType(pickler=RobustPickler)),  # type:ignore
        ]

        table_config = TableConfig(
            self._TABLE_NAME,
            cast(list[Column[Any]], columns),
            self._PRIMARY_KEY,
            existence_strategy,
        )

        super().__init__(
            table_config,
            db_config,
            ProblemInitialization,
            ProblemInitializationWithId,
        )
