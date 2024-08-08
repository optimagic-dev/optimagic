from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import is_dataclass, fields, asdict

from typing import TypeVar, Generic, Type, Any

import pandas as pd

InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")


class AbstractKeyValueStore(Generic[InputType, OutputType], ABC):
    def __init__(self, input_type: Type[InputType], output_type: Type[OutputType]):
        if not (is_dataclass(input_type) and is_dataclass(output_type)):
            raise ValueError("Arguments input_type and output_type must by dataclasses")

        self._output_type = output_type
        self._input_type = input_type
        self._supported_fields = {f.name for f in fields(input_type)}

    def insert(self, value: InputType | dict[str, Any]) -> None:
        self._check_fields(value)
        self._insert(value)

    @abstractmethod
    def _insert(self, value: InputType | dict[str, Any]) -> None:
        pass

    def update(self, key: int, value: InputType | dict[str, Any]) -> None:
        self._check_fields(value)
        self._update(key, value)

    def _check_fields(self, value: InputType | dict[str, Any]) -> None:
        if isinstance(value, dict):
            not_supported_fields = set(value.keys()).difference(self._supported_fields)
            if not_supported_fields:
                raise ValueError(
                    f"Not supported fields {not_supported_fields}. "
                    f"Only supports fields {self._supported_fields}"
                )

    @abstractmethod
    def _update(self, key: int, value: InputType | dict[str, Any]) -> None:
        pass

    @abstractmethod
    def _select_by_key(self, key: int) -> list[OutputType]:
        pass

    @abstractmethod
    def _select_all(self) -> list[OutputType]:
        pass

    def select(self, key: int | None = None) -> list[OutputType]:
        if key is None:
            return self._select_all()

        return self._select_by_key(key)

    @abstractmethod
    def select_last_rows(self, n_rows: int) -> list[OutputType]:
        pass

    def to_df(self) -> pd.DataFrame:
        items = self._select_all()
        return pd.DataFrame([asdict(item) for item in items])  # type:ignore
