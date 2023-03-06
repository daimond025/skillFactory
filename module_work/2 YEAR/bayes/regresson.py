import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


n = np.ones(4)*5
y = np.array([0, 1, 3, 5])
dose = np.array([-.86,-.3,-.05,.73])

N = len(10)

with pm.Model() as baseball_model:

    phi = pm.Uniform("phi", lower=0.0, upper=1.0)

    kappa_log = pm.Exponential("kappa_log", lam=1.5)
    kappa = pm.Deterministic("kappa", tt.exp(kappa_log))

    thetas = pm.Beta("thetas", alpha=phi * kappa, beta=(1.0 - phi) * kappa, shape=N)
    y = pm.Binomial("y", n=at_bats, p=thetas, observed=hits)

with baseball_model:

    theta_new = pm.Beta("theta_new", alpha=phi * kappa, beta=(1.0 - phi) * kappa)
    y_new = pm.Binomial("y_new", n=4, p=theta_new, observed=0)

print(dose)
exit()