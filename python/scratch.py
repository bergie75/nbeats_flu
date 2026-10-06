import numpy as np
from scipy.stats import dirichlet
from particles import distributions as dists


class Dirichlet(dists.ProbDist):
    def __init__(self, alpha):
        self.alpha = np.asarray(alpha, dtype=float)
        self.dim = len(self.alpha)
        self.dtype = np.dtype((float, (self.dim,)))

    def rvs(self, size=None):
        return np.random.dirichlet(self.alpha, size=size)

    def logpdf(self, x):
        try:
            return dirichlet.logpdf(np.asarray(x).T, self.alpha)
        except:
            return -np.inf

if __name__ == "__main__":
    test = Dirichlet([5,2,1])
    print(test.logpdf([0.8,0.1,0.1]))