RELATIVE_CRITERION_TOLERANCE = 2e-9
"""float: Inspired by scipy L-BFGS-B defaults, but rounded."""

ABSOLUTE_CRITERION_TOLERANCE = 0
"""float: Disabled by default because it is very problem specific."""

GRADIENT_TOLERANCE = 1e-5
"""float: Same as scipy."""

RELATIVE_PARAMS_TOLERANCE = 1e-5
"""float: Same as scipy."""

ABSOLUTE_PARAMS_TOLERANCE = 0
"""float: Disabled by default because it is very problem specific."""

MAX_CRITERION_EVALUATIONS = 1_000_000
MAX_ITERATIONS = 1_000_000

MAX_LINE_SEARCH_STEPS = 20
"""int: Inspired by scipy L-BFGS-B."""

INITIAL_TRUST_RADIUS = 1
MAX_TRUST_RADIUS = 100

LIMITED_MEMORY_STORAGE_LENGTH = 10
"""int: Taken from scipy L-BFGS-B."""

SECOND_BEST_ABSOLUTE_CRITERION_TOLERANCE = 1e-08
"""float: absolute criterion tolerance estimagic requires if no other stopping
criterion apart from max iterations etc. is available
this is taken from scipy (SLSQP's value, smaller than Nelder-Mead)"""

SECOND_BEST_ABSOLUTE_PARAMS_TOLERANCE = 0.0001
"""float: The absolute parameter tolerance estimagic requires if no other stopping
criterion apart from max iterations etc. is available. This is taken from Nelder-Mead.
"""

INITIAL_TRUST_RADIUS = 1
"""float: recommended for scipy_trust_constr in :cite:`Conn2000`, p. 19.
It is also scipy's default for COBYLA's the start Rho, which behaves similar to an
initial trust radius."""
