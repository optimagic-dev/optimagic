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


class _KeyValueStore(Generic[InputType, OutputType], ABC):
    """Generic abstract base class for a key-value store.

    This class defines the basic interface for key-value stores that support
    insertion and selection of items based on a primary key.

    Args:
        input_type: The type of input data that can be stored.
        output_type: The type of output data that can be retrieved.
        primary_key: The primary key used to uniquely identify items in the store.

    Raises:
        ValueError: If input_type or output_type is not a dataclass, or if
                the primary key is not found in output_type fields.

    """

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
        """Get the primary key of the store.

        Returns:
            The primary key field name.

        """
        return self._primary_key

    @abstractmethod
    def insert(self, value: InputType) -> None:
        """Implement this method to insert a new value into the key-value store.

        Make sure an auto-increment logic is implemented for the insertion.

        """

    @abstractmethod
    def _select_by_key(self, key: int) -> list[OutputType]:
        """Implement this method to select a value from the store by its primary key."""

    @abstractmethod
    def _select_all(self) -> list[OutputType]:
        """Implement this method to select all values from the store."""

    def select(self, key: int | None = None) -> list[OutputType]:
        """Select items from the store.

        Args:
            key: Optional; the primary key of the item to be selected. If not provided,
                 all items will be selected.

        Returns:
            A list of output items.

        """
        if key is None:
            return self._select_all()

        return self._select_by_key(key)

    @abstractmethod
    def select_last_rows(self, n_rows: int) -> list[OutputType]:
        """Implement this to select the last `n_rows` from the store.

        Args:
            n_rows: The number of rows to select.

        Returns:
            A list of the last `n_rows` output items.

        """

    def to_df(self) -> pd.DataFrame:
        """Convert the store's data to a Pandas DataFrame.

        Returns:
            A DataFrame containing all items in the store.

        """
        items = self._select_all()
        return pd.DataFrame([asdict(item) for item in items])


class UpdatableKeyValueStore(_KeyValueStore[InputType, OutputType], ABC):
    """Generic abstract base class for an updatable key-value store.

    This class extends `KeyValueStore` to add support for updating existing
    items in the store.

    """

    def update(self, key: int, value: InputType | dict[str, Any]) -> None:
        """Update an existing item in the store.

        Args:
            key: The primary key of the item to be updated.
            value: The updated item, or a dictionary of fields to update.

        Raises:
            ValueError: If any fields in `value` are not supported by the store.

        """
        self._check_fields(value)
        self._update(key, value)

    @abstractmethod
    def _update(self, key: int, value: InputType | dict[str, Any]) -> None:
        """Implement the internal method to update an existing item in the store."""

    def _check_fields(self, value: InputType | dict[str, Any]) -> None:
        if isinstance(value, dict):
            not_supported_fields = set(value.keys()).difference(self._supported_fields)
            if not_supported_fields:
                raise ValueError(
                    f"Not supported fields {not_supported_fields}. "
                    f"Only supports fields {self._supported_fields}"
                )


class NonUpdatableKeyValueStore(_KeyValueStore[InputType, OutputType], ABC):
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
