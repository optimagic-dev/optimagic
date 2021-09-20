try:
    from better_exceptions import format_exception
except ImportError:
    from traceback import format_exception
import sys


class TableExistsError(Exception):
    pass


class StopOptimizationError(Exception):
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
