import numpy as np

def projection_simplex_sort(v, z=1):
	# Courtesy: EdwardRaff/projection_simplex.py
    if v.sum() == z and np.alltrue(v >= 0):
        return v
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w



def logistic(w, kappa, C, N):
    """
    Calculates the logistic function of x.
    """
    return 1 / (1 + np.exp(-kappa * (w/N - C)))

def bernoulli(p):
    """
    Samples a Bernoulli random variable with probability of success p.
    """
    if np.random.rand() < p:
        return 1
    else:
        return 0