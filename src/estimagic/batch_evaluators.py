from optimagic.batch_evaluators import joblib_batch_evaluator as _joblib_batch_evaluator
from optimagic.batch_evaluators import (
    pathos_mp_batch_evaluator as _pathos_mp_batch_evaluator,
)
from optimagic.batch_evaluators import (
    process_batch_evaluator as _process_batch_evaluator,
)
from optimagic.decorators import deprecated

MSG = (
    "estimagic.batch_evaluators.{name} has been deprecated in version 0.5.0. Use "
    "optimagic.batch_evaluators.{name} instead. This function will be removed in "
    "version 0.6.0."
)


pathos_mp_batch_evaluator = deprecated(
    _pathos_mp_batch_evaluator, MSG.format(name="pathos_mp_batch_evaluator")
)

joblib_batch_evaluator = deprecated(
    _joblib_batch_evaluator, MSG.format(name="joblib_batch_evaluator")
)

process_batch_evaluator = deprecated(
    _process_batch_evaluator, MSG.format(name="process_batch_evaluator")
)
