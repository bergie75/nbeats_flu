import numpy as np
from scipy.special import gammaln

from particles import distributions as dists


class DirichletWithFixedTail(dists.ProbDist):
    def __init__(self, alpha, fixed):
        self.alpha = np.asarray(alpha, dtype=float)
        self.fixed = np.asarray(fixed, dtype=float)

        if self.alpha.shape != (3,) or np.any(self.alpha <= 0):
            raise ValueError("alpha must contain 3 positive values")
        if self.fixed.shape != (3,):
            raise ValueError("fixed must contain 3 values")

        self.dim = 6
        self.dtype = np.dtype(float)

        # Normalizing constant for the 3-dimensional Dirichlet density
        self._log_norm = (
            gammaln(self.alpha.sum()) - gammaln(self.alpha).sum()
        )

    def rvs(self, size=None):
        x = np.random.dirichlet(self.alpha, size=size)
        if size is None:
            return np.concatenate((x, self.fixed))
        tail = np.broadcast_to(self.fixed, (size, 3))
        return np.concatenate((x, tail), axis=-1)

    def logpdf(self, x):
        x = np.asarray(x)
        first = x[..., :3]
        tail = x[..., 3:]

        # Density is zero outside the Dirichlet simplex or when the
        # final three values do not equal the fixed values.
        valid = (
            np.all(first > 0, axis=-1)
            & np.isclose(first.sum(axis=-1), 1.0)
            & np.all(np.isclose(tail, self.fixed), axis=-1)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            lp = self._log_norm + np.sum(
                (self.alpha - 1) * np.log(first), axis=-1
            )
        return np.where(valid, lp, -np.inf)

if __name__ == "__main__":
    test = DirichletWithFixedTail([5,2,1],[0.0,0.0,1e-9])
    print(test.logpdf([0.8,0.1,0.1,0.1,0,1e-9]))