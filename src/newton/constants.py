# Attempt system decomposition.
DECOMPOSE_SYSTEM = False


# Generally used to avoid numerical issues in constraint definitions.
EPS = 1e-9

# Tolerance for rank checks in the solver.
NONZERO_RANK_TOLERANCE = 1e-6

# For Tikhonov regularization
# TODO: Explore reasonable values for this.
# TODO: We should do this for dense solve too, if we go down that route.
# Ref: https://en.wikipedia.org/wiki/Ridge_regression
REGULARIZATION_LAMBDA = 1e-9
