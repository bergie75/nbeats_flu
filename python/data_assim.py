import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import vmap, jit

import numpy as onp
#from scipy.integrate import odeint
from scipy.special import gammaln
import pandas as pd
from matplotlib import pyplot as plt
import os

# modules from particles
import particles  # core module
from particles import distributions as dists  # where probability distributions are defined
from particles import state_space_models as ssm  # where state-space models are defined
from particles.collectors import Moments
from particles import mcmc  # where the MCMC algorithms (PMMH, Particle Gibbs, etc) live

# Sasha's more sophisticated ODE package
from diffrax import diffeqsolve, ODETerm, Tsit5, SaveAt, PIDController

us_population = 320*10**6  # this is not right, use as starting point
#us_population = 3*10**8

# we will carry out the simulations on proportions, and convert to per 100K rates
@jit
def competing_flu(t, x, p):
    # unpack vectors
    beta_A, beta_B, gamma_A, gamma_B, epsilon, theta_A, theta_B, mu = p
    S, S_A, S_B, V, I_A, I_B, H_A, H_B, R, H_A_cum, H_B_cum = x

    # Force of infection
    lambda_A = beta_A * I_A
    lambda_B = beta_B * I_B

    dS_dt = -(lambda_A + lambda_B)*S
    dS_A_dt = -(lambda_A + epsilon*lambda_B)*S_A
    dS_B_dt = -(epsilon*lambda_A + lambda_B)*S_B
    dV_dt = -epsilon*(lambda_A + lambda_B)*V
    dI_A_dt = lambda_A*(S + S_A + epsilon*(S_B + V)) - gamma_A*I_A
    dI_B_dt = lambda_B*(S + S_B + epsilon*(S_A + V)) - gamma_B*I_B
    dH_A_dt = gamma_A*theta_A*I_A - mu*H_A
    dH_B_dt = gamma_B*theta_B*I_B - mu*H_B
    dR_dt = gamma_A*(1 - theta_A)*I_A + gamma_B*(1 - theta_B)*I_B + mu*(H_A+H_B)
    dH_A_cum_dt = gamma_A*theta_A*I_A
    dH_B_cum_dt = gamma_B*theta_B*I_B

    return jnp.array([dS_dt,dS_A_dt,dS_B_dt,dV_dt,dI_A_dt,dI_B_dt,dH_A_dt,dH_B_dt,dR_dt,dH_A_cum_dt,dH_B_cum_dt])

## create objects for ode solver
odeterm = ODETerm(competing_flu)
odesolver = Tsit5()
odestepsize = PIDController(rtol=1e-6, atol=1e-6)

## used in PX() of state space model:
@jit
def onestep(prev, p):
    deltaT = 7.0  # case data are separated by week
    ## leaving out "saveat" saves only t1
    next = diffeqsolve(odeterm, odesolver, t0=0., t1=deltaT, dt0=None, y0=prev, args=p, stepsize_controller=odestepsize)
    return next.ys[0]

## vectorizes and jax-compiles onestep() for calling on a batch of particles
## the particles are a matrix with dims [# particles, # compartments]
## returns a matrix with same shape
batch_onestep = jit(vmap(onestep, in_axes=(0,None)))

if __name__ == "__main__":
    # dS_dt,dS_A_dt,dS_B_dt,dV_dt,dI_A_dt,dI_B_dt,dH_A_dt,dH_B_dt,dR_dt,dH_A_cum_dt,dH_B_cum_dt
    # prop_guess = jnp.array([0,0.05,0.05,0.47,0.01,0.01,eps_log,eps_log])
    # prop_guess[0] = 1-jnp.sum(prop_guess)
    # alpha_tot = 1
    # alpha = alpha_tot*prop_guess
    # comp_A = prop_guess[0] + prop_guess[1]
    # comp_B = prop_guess[0] + prop_guess[2]

    #construct a prior for our model
    # prior_dict = {"beta_A": dists.Beta(a=1, b=9),
    #               "beta_B": dists.Beta(a=1, b=9),
    #               "gamma_A": dists.Beta(a=1, b=16),
    #               "gamma_B": dists.Beta(a=1, b=16),
    #               "epsilon": dists.Beta(a=5, b=95),
    #               "theta_A": dists.Dirac(loc=0.01),
    #               "theta_B": dists.Dirac(loc=0.01),
    #               "mu": dists.Uniform(a=0.05, b=0.4),
    #               "sigma_meas": dists.Dirac(loc=0.5)}

    # prior_dict = {"beta_A": dists.Beta(a=1, b=9), 
    #             "beta_B": dists.Beta(a=1, b=9),
    #             "gamma_A": dists.Beta(a=1, b=16),
    #             "gamma_B": dists.Beta(a=1, b=16),
    #             "epsilon": dists.Beta(a=5, b=95),
    #             "sigma_meas": dists.Dirac(loc=0.1)}

    prior_dict = {
            "beta_A": dists.Gamma(4.0,6.0),  # likely using alpha-beta parametrization, mean = a/b
            "beta_B": dists.Gamma(4.0,6.0),
            "theta_A": dists.Beta(21.0,40.0),
            "theta_B": dists.Beta(21.0,40.0),
            "sigma_meas": dists.Gamma(10.0,100.0)
            }

    # class for determining logarithm of case counts
    # Note to self: vector-valued inputs to parameters for univariate distributions yields one distribution for each value
    # which is used when you evaluate for each particle
    class FluPart(ssm.StateSpaceModel):

        default_params = {'beta_A':0.8,'beta_B':0.7,'gamma_A':0.2,'gamma_B':0.2,'epsilon':0.05,
                        'theta_A':0.35,'theta_B':0.25,'mu':0.2,'sigma_meas':0.1}

        def PX0(self):  # Distribution of X_0
            nonS = jnp.array([0.05,0.10,0.47,0.001,0.001,0,0,0,1e-9,1e-9])
            y0 = jnp.array([1.0 - jnp.sum(nonS), *nonS])
            component_distributions = [dists.Dirac(loc=y0[j]) for j in range(y0.shape[0])]
            return dists.IndepProd(*component_distributions)

        def PX(self, t, xp):  # Distribution of X_t given X_{t-1}=xp (p=past)
            p = [self.beta_A, self.beta_B, self.gamma_A, self.gamma_B, self.epsilon, self.theta_A, self.theta_B, self.mu]
            totals = batch_onestep(xp, p)
            component_distributions = [dists.Dirac(loc=totals[:, j]) for j in range(totals.shape[1])]
            return dists.IndepProd(*component_distributions)
            
        def PY(self, t, xp, x):  # Distribution of Y_t given X_t=x (and possibly X_{t-1}=xp)
            ## xp is unavailable at t0
            y_A = x[:,9]*us_population/10**5 if xp is None else (x[:,9] - xp[:,9])*us_population/10**5
            y_B = x[:,10]*us_population/10**5 if xp is None else (x[:,10] - xp[:,10])*us_population/10**5
            lognorms = [dists.LogNormal(mu=jnp.log(y_A+1e-9), sigma=self.sigma_meas), dists.LogNormal(mu=jnp.log(y_B+1e-9), sigma=self.sigma_meas)]
            return dists.IndepProd(*lognorms)

    # definition of model and prior
    my_prior = dists.StructDist(prior_dict)

    # acquire data
    cwd = os.getcwd()
    datafile = os.path.join(cwd, "nbeats_flu", "data", "FluSurveillance_Custom_Download_Data.csv")

    # cumulative rate hospitalized is per one hundred thousand
    usecols=["YEAR", "WEEK", "AGE CATEGORY", "RACE CATEGORY", "SEX CATEGORY", "VIRUS TYPE CATEGORY", "CUMULATIVE RATE"]
    df = pd.read_csv(datafile, usecols=usecols)
    overall_data = df[df["AGE CATEGORY"]=="Overall"]
    overall_data = overall_data[overall_data["RACE CATEGORY"]=="Overall"]
    overall_data = overall_data[overall_data["SEX CATEGORY"]=="Overall"].reset_index(drop=True)
    flu_A = overall_data[overall_data["VIRUS TYPE CATEGORY"]=="Influenza A"].reset_index(drop=True)
    flu_B = overall_data[overall_data["VIRUS TYPE CATEGORY"]=="Influenza B"].reset_index(drop=True)
    both = overall_data[overall_data["VIRUS TYPE CATEGORY"]=="Overall"].reset_index(drop=True)

    flu_A_cases = jnp.array(flu_A["CUMULATIVE RATE"])
    flu_A_incident = jnp.diff(flu_A_cases[30:60], prepend=0.0) + 1e-9 # 30:60 is 2010 flu season
    flu_B_cases = jnp.array(flu_B["CUMULATIVE RATE"])
    flu_B_incident = jnp.diff(flu_B_cases[30:60], prepend=0.0) + 1e-9
    both_cases = jnp.array(both["CUMULATIVE RATE"])
    time_series = both_cases[30:60]
    incident_cases = jnp.diff(time_series, prepend=0.0)
    both_strain_incident = jnp.stack([flu_A_incident, flu_B_incident])

    # perform filtering
    pmmh = mcmc.PMMH(ssm_cls=FluPart, prior=my_prior, data=both_strain_incident, Nx=50, niter = 5000)
    pmmh.run()
    plt.plot(pmmh.chain.lpost)
    plt.show()

    for p in prior_dict.keys():
        plt.figure()
        plt.plot(pmmh.chain.theta[p])
        plt.xlabel('iter')
        plt.ylabel(p)
        plt.show()