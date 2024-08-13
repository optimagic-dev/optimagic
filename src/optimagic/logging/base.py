import io
import warnings
from abc import ABC, abstractmethod
from dataclasses import asdict, fields, is_dataclass
from typing import Any, Generic, Type, TypeVar

import cloudpickle
import pandas as pd

from optimagic.exceptions import get_traceback
from optimagic.typing import DictLikeAccess

InputType = TypeVar("InputType", bound=DictLikeAccess)
OutputType = TypeVar("OutputType", bound=DictLikeAccess)


class KeyValueStore(Generic[InputType, OutputType], ABC):
    def __init__(
        self,
        input_type: Type[InputType],
        output_type: Type[OutputType],
        primary_key: str,
    ):
        if not (is_dataclass(input_type) and is_dataclass(output_type)):
            raise ValueError("Arguments input_type and output_type must by dataclasses")

        output_fields = {f.name for f in fields(output_type)}
        if primary_key not in output_fields:
            raise ValueError(
                f"Primary key {primary_key} not found in output_type fields "
                f"{fields(output_type)}"
            )

        self._output_type = output_type
        self._input_type = input_type
        self._primary_key = primary_key
        self._supported_fields = {f.name for f in fields(input_type)}

    @property
    def primary_key(self) -> str:
        return self._primary_key

    @abstractmethod
    def insert(self, value: InputType) -> None:
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
        return pd.DataFrame([asdict(item) for item in items])


class UpdatableKeyValueStore(KeyValueStore[InputType, OutputType], ABC):
    def update(self, key: int, value: InputType | dict[str, Any]) -> None:
        self._check_fields(value)
        self._update(key, value)

    @abstractmethod
    def _update(self, key: int, value: InputType | dict[str, Any]) -> None:
        pass

    def _check_fields(self, value: InputType | dict[str, Any]) -> None:
        if isinstance(value, dict):
            not_supported_fields = set(value.keys()).difference(self._supported_fields)
            if not_supported_fields:
                raise ValueError(
                    f"Not supported fields {not_supported_fields}. "
                    f"Only supports fields {self._supported_fields}"
                )


class NonUpdatableKeyValueStore(KeyValueStore[InputType, OutputType], ABC):
    def __getattr__(self, name: str) -> Any:
        if name == "update":
            msg = (
                f"'{self.__class__.__name__}' object does not allow to update items in"
                f"the store"
            )
        else:
            msg = f"'{self.__class__.__name__}' object has no attribute '{name}'"
        raise AttributeError(msg)


class RobustPickler:
    @staticmethod
    def loads(
        data: Any,
        fix_imports: bool = True,  # noqa: ARG004
        encoding: str = "ASCII",  # noqa: ARG004
        errors: str = "strict",  # noqa: ARG004
        buffers: Any = None,  # noqa: ARG004
    ) -> Any:
        """Robust pickle loading.

        We first try to unpickle the object with pd.read_pickle. This makes no
        difference for non-pandas objects but makes the de-serialization
        of pandas objects more robust across pandas versions. If that fails, we use
        cloudpickle. If that fails, we return None but do not raise an error.

        See: https://github.com/pandas-dev/pandas/issues/16474

        """
        try:
            res = pd.read_pickle(io.BytesIO(data), compression=None)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            try:
                res = cloudpickle.loads(data)
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception:
                res = None
                tb = get_traceback()
                warnings.warn(
                    f"Unable to read PickleType column from database:\n{tb}\n "
                    "The entry was replaced by None."
                )

        return res

    @staticmethod
    def dumps(
        obj: Any,
        protocol: str | None = None,
        *,
        fix_imports: bool = True,  # noqa: ARG001
        buffer_callback: Any = None,  # noqa: ARG004
    ) -> Any:
        return cloudpickle.dumps(obj, protocol=protocol)
