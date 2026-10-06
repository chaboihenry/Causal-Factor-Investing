"""
Collider bias: controlling for a common effect can flip a true factor's sign.

Simulates the second mechanism behind de Prado's 'factor mirage':
adding a control caused by both the factor and returns raises adjusted R²
while biasing, and even reversing, the factor's estimated effect.
"""

import numpy as np
import statsmodels.api as sm

rng = np.random.default_rng(0)
n = 5000
beta = 0.5   # true causal effect of x on y

x = rng.normal(size=n)
y = beta * x + rng.normal(size=n)
noise_c = rng.normal(size=n)

# Correct model: y ~ x
fit_naive = sm.OLS(y, sm.add_constant(x)).fit()
print(f"Correct model         beta_x={fit_naive.params[1]: .3f}  "
      f"p={fit_naive.pvalues[1]:.1e}  adjR2={fit_naive.rsquared_adj:.3f}\n")

# Settings: (a, b) = strength of x -> c and y -> c
settings = [(1, 1), (0, 1), (1, 0), (2, 2)]

for a, b in settings:
    c = a * x + b * y + noise_c

    # Collider model: y ~ x + c  (columns: const, x, c -> x is index 1)
    design = sm.add_constant(np.column_stack([x, c]))
    fit = sm.OLS(y, design).fit()

    print(f"a={a} b={b}  collider beta_x={fit.params[1]: .3f}  "
          f"p={fit.pvalues[1]:.1e}  adjR2={fit.rsquared_adj:.3f}")