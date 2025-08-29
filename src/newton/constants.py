# Use symbolic substitution in the solver.
CONFIG_USE_SYMB_SUB = True

# Use sparse solver for the 2D solver.
CONFIG_USE_SPARSE_SOLVE = True
CONFIG_USE_NEWTON_FAER = True
CONFIG_USE_REGULARIZATION = True

# For Tikhonov regularization
# TODO: Explore reasonable values for this.
# TODO: We should do this for dense solve too, if we go down that route.
# Ref: https://en.wikipedia.org/wiki/Ridge_regression
REGULARIZATION_LAMBDA = 1e-9

# Generally used to avoid numerical issues in constraint definitions.
EPS = 1e-9

# Tolerance for rank checks in the solver.
NONZERO_RANK_TOLERANCE = 1e-6
