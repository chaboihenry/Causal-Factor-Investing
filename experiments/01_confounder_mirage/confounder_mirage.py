"""Confounder bias: a factor with zero causal effect looks significant.

Reproduces the core mechanism behind de Prado's 'factor mirage':
an associational regression reports a premium that does not causally exist.
"""

import numpy as np
import statsmodels.api as sm

rng = np.random.default_rng(0)
n = 5000

# Z is a hidden confounder, e.g. a macro/market regime.
z = rng.normal(size=n)

# Candidate factor F is driven by Z, but has NO direct effect on returns.
f = 0.8 * z + rng.normal(size=n)

# True returns depend ONLY on Z; the causal effect of F is exactly zero.
r = 1.0 * z + rng.normal(size=n)

# Traditional factor study: regress returns on the factor alone.
naive = sm.OLS(r, sm.add_constant(f)).fit()

# Causal version: adjust for the confounder (back-door adjustment).
causal = sm.OLS(r, sm.add_constant(np.column_stack([f, z]))).fit()

print(f"Naive  beta_F (truth ~0): {naive.params[1]: .3f}  p={naive.pvalues[1]:.1e}")
print(f"Causal beta_F (truth  0): {causal.params[1]: .3f}  p={causal.pvalues[1]:.1e}")