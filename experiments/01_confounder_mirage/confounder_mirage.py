"""
Confounder or mediator: the same two regressions, opposite truths.
Fork (Z -> F, Z -> R): F has zero effect; adjusting for Z removes the false beta.
Chain (F -> Z -> R): F's effect is 0.8, all through Z; adjusting for Z hides it.
Prints naive (R on F) and adjusted (R on F, Z) betas. Expected: fork 0.482 / ~0, chain 0.790 / ~0.
"""

import numpy as np
import statsmodels.api as sm

N = 5000
SEED = 0


def fork(rng):
    """Z (e.g. a market regime) drives both F and R; F has no effect on R."""
    z = rng.normal(size=N)
    f = 0.8 * z + rng.normal(size=N)
    r = 1.0 * z + rng.normal(size=N)
    return f, z, r


def chain(rng):
    """F drives Z, which drives R: F's total effect on R is 0.8 * 1.0, entirely through Z."""
    f = rng.normal(size=N)
    z = 0.8 * f + rng.normal(size=N)
    r = 1.0 * z + rng.normal(size=N)
    return f, z, r


def beta_f(r, regressors):
    fit = sm.OLS(r, sm.add_constant(np.column_stack(regressors))).fit()
    return fit.params[1], fit.pvalues[1]


for name, world, truth in [('Fork ', fork, 0.0), ('Chain', chain, 0.8)]:
    f, z, r = world(np.random.default_rng(SEED))
    for label, regressors in [('naive   ', [f]), ('adjusted', [f, z])]:
        beta, p = beta_f(r, regressors)
        print(f"{name} {label} beta_F: {beta: .3f}  p={p:.1e}  (true effect {truth})")

print("Same regression output, opposite truth: adjust for Z in the fork, never in the chain.")
