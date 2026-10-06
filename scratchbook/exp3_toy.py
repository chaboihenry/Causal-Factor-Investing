import numpy as np
from causallearn.search.ConstraintBased.PC import pc

rng = np.random.default_rng(1)
n = 2000
# a = rng.normal(size=n)
# b = 0.8 * a + rng.normal(size=n)
# c = 0.6 * b + rng.normal(size=n) # true chain: a -> b -> c

a = rng.normal(size=n)
c = rng.normal(size=n)
b = 0.8 * a + 0.6 * c + rng.normal(size=n)   # a -> b <- c

data = np.column_stack([a, b, c])
cg = pc(data, alpha=0.05, indep_test='fisherz', node_names=['a', 'b', 'c'])
print(cg.G) # discovered Graph


