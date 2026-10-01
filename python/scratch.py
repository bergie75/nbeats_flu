import numpy as np
from particles import distributions as dists
from scipy import stats

# return stats.lognorm.logpdf(x, self.sigma, scale=np.exp(self.mu))
print(stats.lognorm.logpdf(1e-9, 0.1, scale=np.exp(1e-9)))
# # if we put x=mu into pdf, the exponential is zero and the only thing left is the prefactor
# f=1/(np.sqrt(2*np.pi*0.1*0.1)*1e-9)
# print(np.log(f))
print(dists.LogNormal(mu=1e-9, sigma=0.1).logpdf(1e-9))