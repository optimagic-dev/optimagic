import sys
from traceback import format_exception


class EstimagicError(Exception):
    """Base exception for estimagic which should be inherited by all exceptions."""


class TableExistsError(EstimagicError):
    """Exception for database tables that should not exist but do."""


class InvalidFunctionError(EstimagicError):
    """Exception for invalid user provided functions.

    This includes user functions that do not comply with interfaces, raise errors or
    produce NaNs.

    """


class UserFunctionRuntimeError(EstimagicError):
    """Exception that is raised when user provided functions raise errors."""


class InvalidKwargsError(EstimagicError):
    """Exception for invalid user provided keyword arguments."""


class InvalidParamsError(EstimagicError):
    """Exception for invalid user provided parameters."""


class InvalidConstraintError(EstimagicError):
    """Exception for invalid user provided constraints."""


class NotInstalledError(EstimagicError):
    """Exception when optional dependencies are needed but not installed."""


class NotAvailableError(EstimagicError):
    """Exception when something is not available, e.g. because a calculation failed."""


class StopOptimizationError(EstimagicError):
    def __init__(self, message, current_status):
        super().__init__(message)
        self.message = message
        self.current_status = current_status

    def __reduce__(self):
        """Taken from here: https://tinyurl.com/y6eeys2f"""
        return (StopOptimizationError, (self.message, self.current_status))


def get_traceback():
    tb = format_exception(*sys.exc_info())
    if isinstance(tb, list):
        tb = "".join(tb)
    return tb


INVALID_INFERENCE_MSG = (
    "Taking the inverse of the information matrix failed. Only ever use this "
    "covariance matrix or standard errors based on it for diagnostic purposes, not for "
    "drawing conclusions."
)


INVALID_SENSITIVITY_MSG = (
    "Taking inverse failed during the calculation of sensitvity measures. Interpret "
    "them with caution."
)
