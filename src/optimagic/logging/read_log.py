"""
Deprecated module:

Functions to read data from the database used for logging.

The functions in the module are meant for end users of optimagic. They do not require
any knowledge of databases.

When using them internally, make sure to supply a database to path_or_database.
Otherwise, the functions may be very slow.

"""

import warnings
from dataclasses import dataclass

from optimagic.logging.logger import SQLiteLogOptions, SQLiteLogReader


@dataclass
class OptimizeLogReader:
    def __new__(cls, *args, **kwargs):
        warnings.warn(
            "OptimizeLogReader is deprecated and will be removed in a future "
            "version. Please use optimagic.logging.SQLiteLogReader instead.",
            FutureWarning,
        )
        sqlite_options = SQLiteLogOptions(*args, **kwargs)
        return SQLiteLogReader.from_options(sqlite_options)
