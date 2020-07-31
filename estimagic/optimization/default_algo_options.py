# Inspired by scipy L-BFGS-B defaults, but rounded
RELATIVE_CRITERION_TOLERANCE = 2e-9

# Disabled by default because it is very problem specific
ABSOLUTE_CRITERION_TOLERANCE = 0

# Same as scipy
GRADIENT_TOLERANCE = 1e-5

# Same as scipy
RELATIVE_PARAMS_TOLERANCE = 1e-5

# Disabled by default because it is very problem specific
ABSOLUTE_PARAMS_TOLERANCE = 0

MAX_CRITERION_EVALUATIONS = 1_000_000
MAX_ITERATIONS = 1_000_000

# Inspired by scipy L-BFGS-B
MAX_LINE_SEARCH_STEPS = 20

INITIAL_TRUST_RADIUS = 1
MAX_TRUST_RADIUS = 100


# Taken from scipy L-BFGS-B
LIMITED_MEMORY_STORAGE_LENGTH = 10
