import numpy as np
import jax
import jax.numpy as jnp
from scipy.special import gammaln

from particles import distributions as dists
from particles import state_space_models as ssm  # where state-space models are defined
key = jax.random.key(np.random.randint(1,10**6))

class BetaFirst(dists.ProbDist):
    def __init__(self, beta_0, beta_1, fixed):
        self.beta_0 = beta_0
        self.beta_1 = beta_1
        self.alpha = jnp.array([beta_0, beta_1])
        self.fixed = fixed
        self._log_norm = gammaln(self.alpha.sum()) - gammaln(self.alpha).sum()
        self.remainder = 1.0-fixed[0]

    def rvs(self, key=key, size=None):
        fixed = jnp.asarray(self.fixed)
        scale = 1.0 - fixed[0]

        shape = () if size is None else (size,)
        x = jax.random.beta(key, self.beta_0, self.beta_1, shape=shape)
        s = scale * x
        v = scale * (1.0 - x)

        if size is None:
            return jnp.concatenate((jnp.stack((s, v)), fixed))

        tail = jnp.broadcast_to(fixed, (size, fixed.shape[0]))
        return jnp.concatenate((jnp.stack((s, v), axis=-1), tail), axis=-1)
    
    def logpdf(self, x):
        x = np.asarray(x)
        first = x[..., :2]
        tail = x[..., 2:]

        # Density is zero outside the Dirichlet simplex or when the
        # final three values do not equal the fixed values.
        valid = (
            jnp.all(first > 0, axis=-1)
            & jnp.isclose(first.sum(axis=-1), self.remainder)
            & jnp.all(np.isclose(tail, self.fixed), axis=-1)
        )
        lp = -jnp.log(self.remainder) + self._log_norm + jnp.sum((self.alpha - 1) * jnp.log(first/self.remainder), axis=-1)
        return jnp.where(valid, lp, -np.inf)

thingy = BetaFirst(2,2,[0.01,0,0,1e-9])
x=thingy.rvs(size=5)
print(thingy.logpdf(x))