from __future__ import annotations

import traceback
import warnings
from dataclasses import asdict, dataclass
from functools import cached_property
from typing import Any, Sequence, Type, cast

import sqlalchemy as sql
from sqlalchemy import Column, Integer, PickleType, String
from sqlalchemy.engine.base import Engine
from sqlalchemy.sql.base import Executable
from sqlalchemy.sql.schema import MetaData

from optimagic.logging.base import (
    InputType,
    NonUpdatableKeyValueStore,
    OutputType,
    RobustPickler,
    UpdatableKeyValueStore,
)
from optimagic.logging.types import (
    IterationState,
    IterationStateWithId,
    ProblemInitialization,
    ProblemInitializationWithId,
    StepResult,
    StepResultWithId,
)


class SQLAlchemyConfig:
    """Configuration class for setting up an SQLAlchemy engine and metadata.

    This class manages the connection URL, engine creation, and metadata reflection
    for an SQLAlchemy database connection.

    Args:
        url: The database URL to connect to.

    """

    def __init__(
        self,
        url: str,
    ):
        self.url = url

    @cached_property
    def metadata(self) -> MetaData:
        """Get the metadata object.

        Returns:
            The SQLAlchemy MetaData object reflecting the database schema.

        """
        engine = self.create_engine()
        metadata = MetaData()
        self._configure_reflect()
        metadata.reflect(engine)
        return metadata

    def create_engine(self) -> Engine:
        """Create and return an SQLAlchemy engine.

        Returns:
            An SQLAlchemy Engine object.

        """
        return sql.create_engine(self.url)

    @staticmethod
    def _configure_reflect() -> None:
        """Mark all BLOB dtypes as PickleType with our custom pickle reader.

        Code ist taken from the documentation: https://tinyurl.com/y7q287jr

        """

        @sql.event.listens_for(sql.Table, "column_reflect")
        def _setup_pickletype(
            inspector: Any, table: sql.Table, column_info: dict[str, Any]
        ) -> None:  # noqa: ARG001
            if isinstance(column_info["type"], sql.BLOB):
                column_info["type"] = sql.PickleType(pickler=RobustPickler)  # type:ignore


@dataclass
class TableConfig:
    """Configuration for creating and managing SQLAlchemy tables.

    This class defines the schema for an SQLAlchemy table, including its name,
    columns, primary key, and strategy for handling existing tables.

    Args:
        table_name: The name of the table.
        columns: A list of SQLAlchemy Column objects defining the table schema.
        primary_key: The name of the primary key column.

    """

    table_name: str
    columns: list[sql.Column[Any]]
    primary_key: str

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    def create_table(self, metadata: MetaData, engine: Engine) -> sql.Table:
        """Create or reflect the table in the database.

        Args:
            metadata: The SQLAlchemy MetaData object.
            engine: The SQLAlchemy Engine object.

        Returns:
            The SQLAlchemy Table object representing the created or reflected table.

        """
        metadata.reflect(engine)
        table = sql.Table(
            self.table_name, metadata, *self.columns, extend_existing=True
        )
        metadata.create_all(engine)
        return table


class _SQLAlchemyStoreMixin:
    """Mixin class for common SQLAlchemy store operations.

    This class provides common methods for selecting, inserting, and executing
    SQL statements in an SQLAlchemy-based key-value store.

    Args:
        db_config: The SQLAlchemyConfig object for database configuration.
        table_config: The TableConfig object for table configuration.

    """

    def __init__(self, db_config: SQLAlchemyConfig, table_config: TableConfig):
        self._db_config = db_config
        self._engine = db_config.create_engine()
        self._table_config = table_config
        self._table = table_config.create_table(db_config.metadata, self._engine)

    @property
    def column_names(self) -> list[str]:
        return self._table_config.column_names

    @property
    def table_name(self) -> str:
        return self._table_config.table_name

    @property
    def table(self) -> sql.Table:
        return self._table

    @property
    def engine(self) -> Engine:
        return self._engine

    def _select_row_by_key(self, key: int) -> list[Any]:
        stmt = self._table.select().where(
            getattr(self._table.c, self._table_config.primary_key) == key
        )
        return self._execute_read_statement(stmt)

    def _select_all_rows(self) -> list[Any]:
        stmt = self._table.select()
        return self._execute_read_statement(stmt)

    def _select_last_rows(self, n_rows: int) -> list[Any]:
        stmt = (
            self._table.select()
            .order_by(getattr(self._table.c, self._table_config.primary_key).desc())
            .limit(n_rows)
        )
        result = self._execute_read_statement(stmt)
        return result[::-1]

    def _insert(self, insert_values: dict[str, Any]) -> None:
        stmt = self._table.insert().values(**insert_values)
        self._execute_write_statement(stmt)

    def _execute_read_statement(self, statement: Executable) -> list[Any]:
        with self._engine.connect() as connection:
            return connection.execute(statement).fetchall()

    def _execute_write_statement(self, statement: Executable) -> None:
        try:
            with self._engine.begin() as connection:
                connection.execute(statement)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            exception_info = traceback.format_exc()
            warnings.warn(
                f"Unable to write to database. The traceback was:\n\n{exception_info}"
            )


class SQLAlchemySimpleStore(
    NonUpdatableKeyValueStore[InputType, OutputType],
    _SQLAlchemyStoreMixin,
):
    """A simple SQLAlchemy-based key-value store that does not support updates.

    This class provides basic key-value storage functionality using SQLAlchemy,
    where values are serialized and stored as BLOBs. The store does not support
    updating existing entries.

    Args:
            table_name: The name of the table.
            primary_key: The primary key column name.
            db_config: The SQLAlchemyConfig object for database configuration.

    """

    _value_column: str = "serialized_value"

    def __init__(
        self,
        table_name: str,
        primary_key: str,
        db_config: SQLAlchemyConfig,
        input_type: Type[InputType],
        output_type: Type[OutputType],
    ):
        super().__init__(input_type, output_type, primary_key)
        columns = [
            sql.Column(primary_key, sql.Integer, primary_key=True, autoincrement=True),
            sql.Column(self._value_column, sql.PickleType(pickler=RobustPickler)),  # type:ignore
        ]
        table_config = TableConfig(table_name, columns, self.primary_key)

        _SQLAlchemyStoreMixin.__init__(self, db_config, table_config)

    def __reduce__(
        self,
    ) -> tuple[
        Type[SQLAlchemySimpleStore[Any, Any]],
        tuple[str, str, SQLAlchemyConfig, Type[Any], Type[Any]],
    ]:
        return SQLAlchemySimpleStore, (
            self.table_name,
            self.primary_key,
            self._db_config,
            self._input_type,
            self._output_type,
        )

    def insert(self, value: InputType) -> None:
        """Insert a new value into the store.

        Args:
            value: The value to insert into the store.

        """
        self._insert({self._value_column: value})

    def _select_by_key(self, key: int) -> list[OutputType]:
        result = self._select_row_by_key(key)
        return self._post_process(result)

    def _select_all(self) -> list[OutputType]:
        result = self._select_all_rows()
        return self._post_process(result)

    def select_last_rows(self, n_rows: int) -> list[OutputType]:
        """Select the last `n_rows` values from the store.

        Args:
            n_rows: The number of rows to select.

        Returns:
            A list of the last `n_rows` output values.

        """
        result = self._select_last_rows(n_rows)
        return self._post_process(result)

    def _post_process(self, results: Sequence[sql.Row]) -> list[OutputType]:  # type:ignore
        output_list = []
        for row in results:
            row_dict = {self.primary_key: row[0]}
            row_dict.update(asdict(row[-1]))
            output_list.append(self._output_type(**row_dict))
        return output_list


class SQLAlchemyTableStore(
    UpdatableKeyValueStore[InputType, OutputType], _SQLAlchemyStoreMixin
):
    """An SQLAlchemy-based key-value store that supports updates.

    This class provides key-value storage functionality using SQLAlchemy,
    allowing for insertion, updating, and selection of data.

    Args:
        table_config: The TableConfig object defining the table schema.
        db_config: The SQLAlchemyConfig object for database configuration.
        input_type: The type of input data.
        output_type: The type of output data.

    """

    def __init__(
        self,
        table_config: TableConfig,
        db_config: SQLAlchemyConfig,
        input_type: Type[InputType],
        output_type: Type[OutputType],
    ):
        _SQLAlchemyStoreMixin.__init__(self, db_config, table_config)
        super().__init__(input_type, output_type, self._table_config.primary_key)

    def __reduce__(
        self,
    ) -> tuple[
        Type[SQLAlchemyTableStore[Any, Any]],
        tuple[TableConfig, SQLAlchemyConfig, Type[Any], Type[Any]],
    ]:
        return SQLAlchemyTableStore, (
            self._table_config,
            self._db_config,
            self._input_type,
            self._output_type,
        )

    def insert(self, value: InputType) -> None:
        """Insert a new value into the store.

        Args:
            value: The value to insert into the store.

        """
        self._insert(asdict(value))

    def _update(self, key: int, value: InputType | dict[str, Any]) -> None:
        if not isinstance(value, dict):
            update_values = asdict(value)
        else:
            update_values = value
        stmt = (
            self._table.update()
            .where(getattr(self._table.c, self.primary_key) == key)
            .values(**update_values)
        )
        self._execute_write_statement(stmt)

    def _select_by_key(self, key: int) -> list[OutputType]:
        result = self._select_row_by_key(key)
        return self._post_process(result)

    def _select_all(self) -> list[OutputType]:
        result = self._select_all_rows()
        return self._post_process(result)

    def select_last_rows(self, n_rows: int) -> list[OutputType]:
        """Select the last `n_rows` values from the store.

        Args:
            n_rows: The number of rows to select.

        Returns:
            A list of the last `n_rows` output values.

        """
        result = self._select_last_rows(n_rows)
        return self._post_process(result)

    def _post_process(self, results: Sequence[sql.Row]) -> list[OutputType]:  # type:ignore
        return [
            self._output_type(**dict(zip(self.column_names, row, strict=False)))
            for row in results
        ]


class IterationStore(SQLAlchemySimpleStore[IterationState, IterationStateWithId]):
    """Store for managing iteration data in an SQLite database.

    Args:
        db_config (SQLiteConfig): The SQLiteConfig object for database configuration.

    """

    _TABLE_NAME = "optimization_iterations"
    _PRIMARY_KEY = "rowid"

    def __init__(
        self,
        db_config: SQLAlchemyConfig,
    ):
        super().__init__(
            self._TABLE_NAME,
            self._PRIMARY_KEY,
            db_config,
            IterationState,
            IterationStateWithId,
        )


class StepStore(SQLAlchemyTableStore[StepResult, StepResultWithId]):
    """Store for managing step data in an SQLite database.

    Args:
        db_config (SQLiteConfig): The SQLiteConfig object for database configuration.

    """

    _TABLE_NAME = "steps"
    _PRIMARY_KEY = "rowid"

    def __init__(
        self,
        db_config: SQLAlchemyConfig,
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
    """Store for managing optimization problem initialization data in an SQLite
    database.

    Args:
        db_config (SQLiteConfig): The SQLiteConfig object for database configuration.

    """

    _TABLE_NAME = "optimization_problem"
    _PRIMARY_KEY = "rowid"

    def __init__(
        self,
        db_config: SQLAlchemyConfig,
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
        )

        super().__init__(
            table_config,
            db_config,
            ProblemInitialization,
            ProblemInitializationWithId,
        )
