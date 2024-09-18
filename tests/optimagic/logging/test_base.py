from dataclasses import dataclass

import pytest

from optimagic.logging.base import InputType, NonUpdatableKeyValueStore, OutputType
from optimagic.typing import DictLikeAccess


def test_key_value_store_raise_errors():
    class NoDataClass(NonUpdatableKeyValueStore):
        def __init__(self):
            super().__init__({1}, [], "key")

        def insert(self, value: InputType) -> None:
            pass

        def _select_by_key(self, key: int) -> list[OutputType]:
            pass

        def _select_all(self) -> list[OutputType]:
            pass

        def select_last_rows(self, n_rows: int) -> list[OutputType]:
            pass

    class WrongPrimaryKey(NonUpdatableKeyValueStore):
        @dataclass(frozen=True)
        class InputDummy(DictLikeAccess):
            value: str

        @dataclass(frozen=True)
        class OutputDummy(DictLikeAccess):
            id: int
            value: str

        def __init__(self):
            super().__init__(
                WrongPrimaryKey.InputDummy, WrongPrimaryKey.OutputDummy, "ID"
            )

        def insert(self, value: InputType) -> None:
            pass

        def _select_by_key(self, key: int) -> list[OutputType]:
            pass

        def _select_all(self) -> list[OutputType]:
            pass

        def select_last_rows(self, n_rows: int) -> list[OutputType]:
            pass

    with pytest.raises(ValueError):
        NoDataClass()

    with pytest.raises(ValueError):
        WrongPrimaryKey()
