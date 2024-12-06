import inspect
import sys
import blackjax
import discovery as ds
import jax
from jax import numpy as jnp
from discovery.deterministic import CW_Signal
from matplotlib import pyplot as plt
import time
import glob
import matplotlib.pyplot as plt

import optax
import matplotlib.pyplot as plt

def load_pulsars(psr_files_pattern):
    """
    Load pulsar data from files matching the given pattern.

    :param psr_files_pattern: Glob pattern to match pulsar data files
    :return: List of Pulsar objects
    """
    psr_files = glob.glob(psr_files_pattern)
    psrs = [ds.Pulsar.read_feather(psr) for psr in psr_files]
    return psrs

def get_timespan(psrs):
    """
    Get the timespan for the given pulsars.

    :param psrs: List of Pulsar objects
    :return: Timespan
    """
    return ds.getspan(psrs)

def mydelay(psr, prior, common=[], name='delay'):
    """
    Create a delay function for a given pulsar and prior.

    :param psr: Pulsar object
    :param prior: Prior function
    :param common: List of common parameters
    :param name: Name prefix for parameters
    :return: Delay function
    """
    psr._pos = jnp.asarray(psr.pos)
    delay = CW_Signal(psr)._get_delay

    argspec = inspect.getfullargspec(prior)
    argmap = [(arg if arg in common else f'{name}_{arg}' if f'{name}_{arg}' in common else f'{psr.name}_{name}_{arg}')
              for arg in argspec.args]

    def priorfunc(params):
        return prior(*[params[arg] for arg in argmap])
    priorfunc.params = argmap

    def delayfunc(params):
        inp = jnp.asarray([params[arg] for arg in argmap])
        return delay(*inp)
    delayfunc.params = argmap

    return delayfunc

def prior(cw_costheta, cw_phi, cw_cosinc, cw_log10_Mc, cw_log10_fgw, cw_log10_h, cw_phase0, cw_psi, **kwargs):
    """
    Prior of the Continuous Wave (CW).

    :param cw_costheta: Cosine of the GW source polar angle in celestial coordinates [radians]
    :param cw_phi: GW source azimuthal angle in celestial coordinates [radians]
    :param cw_cosinc: Cosine of the inclination of the GW source [radians]
    :param cw_log10_Mc: log10 of the SMBHB chirp mass [solar masses]
    :param cw_log10_fgw: log10 of the GW frequency [Hz]
    :param cw_log10_h: log10 of the GW strain
    :param cw_phase0: Initial phase of the GW source [radians]
    :param cw_psi: Polarization angle of the GW source [radians]
    :return: GW induced residuals from continuous wave source
    """
    return 0

def create_likelihood(psrs, tspan):
    """
    Create the likelihood function for the given pulsars and timespan.

    :param psrs: List of Pulsar objects
    :param tspan: Timespan
    :return: ArrayLikelihood object
    """
    com = ['cw_costheta', 'cw_phi', 'cw_cosinc', 'cw_log10_Mc', 'cw_log10_fgw', 'cw_log10_h', 'cw_phase0', 'cw_psi']
    gl = ds.ArrayLikelihood([ds.PulsarLikelihood([
        psr.residuals,
        ds.makenoise_measurement(psr, psr.noisedict),
        ds.makegp_ecorr(psr, psr.noisedict),
        ds.makegp_timing(psr),
        ds.makegp_fourier(psr, ds.powerlaw, 30, T=tspan, name='red_noise'),
        ds.makegp_fourier(psr, ds.powerlaw, 14, T=tspan, common=['gw_log10_A', 'gw_gamma'], name='gw'),
        mydelay(psr, prior, common=com, name=f'cw{0}')
    ]) for psr in psrs])
    return gl

def create_prior(params):
    """
    Create the prior function for the given parameters.

    :param params: List of parameter names
    :return: Log prior function
    """
    return ds.prior.makelogprior_uniform(params, ds.prior.priordict_standard)

def logdensity(x, jlogp, jlogl):
    """
    Compute the log density (log prior + log likelihood).

    :param x: Parameter values
    :param jlogp: JIT compiled log prior function
    :param jlogl: JIT compiled log likelihood function
    :return: Log density
    """
    return jlogp(x) + jlogl(x)

# Example usage
psr_files_pattern = "../data/v1p1_de440_pint_bipm2019-B*.feather"
psrs = load_pulsars(psr_files_pattern)
tspan = get_timespan(psrs)

gl = create_likelihood(psrs, tspan)
initial_position = ds.prior.sample_uniform(gl.logL.params)
logprior = create_prior(gl.logL.params)
loglike = gl.logL

# JIT compile the log likelihood and log prior
jlogl = jax.jit(loglike)
jlogp = jax.jit(logprior)

# Measure execution time of logdensity
logdensity(initial_position, jlogp, jlogl)
initial_position = ds.prior.sample_uniform(gl.logL.params)

start_time = time.time()
logdensity(initial_position, jlogp, jlogl)
end_time = time.time()
print(f"logdensity execution time: {end_time - start_time} seconds")
print(f"Initial position: {initial_position}")
# Test log likelihood and log prior for different parameters
for i in range(10):
    x = ds.prior.sample_uniform(gl.logL.params)
    print(f"Log density : {logdensity(x, jlogp, jlogl)}")
    print(f"Log likelihood : {jlogl(x)}")
    print(f"Log prior : {jlogp(x)}")

# # obtain the gradient of the log density
# jgrad_logdensity = jax.jit(jax.grad(logdensity))
# # maximize the log density using the gradient
# # max_position, max_logdensity = blackjax.inference.max_likelihood(initial_position, jlogl, jgrad_logdensity)
# # print(f"Max position: {max_position}")
# # print(f"Max log density: {max_logdensity}")
# # perform maximization with jax.scipy.optimize.minimize
# max_position = jax.scipy.optimize.minimize(lambda x: -logdensity(x, jlogp, jlogl), initial_position)

# def eval_logdensity(params):
#     return logdensity(params, jlogp, jlogl)

# grads = jax.grad(eval_logdensity)
# print(f"Gradient: {grads(initial_position)}")

# create function that from parameter to dictionary
list_name = gl.logL.params
def to_dict(xs):
    return dict(zip(list_name, xs))

# density function without dictionary but from array
def log_density(xs, jlogl, list_name):
    map_x = {list_name[ii]: xs[ii] for ii in range(len(list_name))}#dict(zip(list_name, xs))
    res_ll = jlogl(map_x)
    if jnp.isnan(res_ll) or jnp.isinf(res_ll):
        return -1e20
    else:
        return res_ll

# gradient of the log density
grads = jax.grad(log_density)
# test the gradient witn initial position
# print(grads(params))
# Optimization with differential evolution
from scipy.optimize import differential_evolution
import re

# Define parameter boundaries based on priors
bounds = []
for par in initial_position:
    for parname, ranges in ds.prior.priordict_standard.items():
        if re.match(parname, par):
            bounds.append(ranges)

# print bounds and respective names
for i, bound in enumerate(bounds):
    print(f"{gl.logL.params[i]}: {bound}")
###########################################################################
# Eryn sample code
from eryn.ensemble import EnsembleSampler
from eryn.state import State
from eryn.prior import ProbDistContainer, uniform_dist
from eryn.utils import TransformContainer
from eryn.moves import GaussianMove, StretchMove, CombineMove
from eryn.utils.utility import groups_from_inds

import corner

ndim = len(gl.logL.params)
priors_in = {i: uniform_dist(bound[0], bound[1]) for i, bound in enumerate(bounds)}
priors = ProbDistContainer(priors_in)
nwalkers = 32
coords = priors.rvs(size=(nwalkers,))
print(priors.logpdf(coords))
# check speed coords[0]
print(log_density(coords[0], jlogl, list_name))

time1 = time.time()
print(log_density(coords[1], jlogl, list_name))
time2 = time.time()
print(f"Time taken: {time2 - time1} seconds")

ensemble = EnsembleSampler(
    nwalkers,
    ndim,
    log_density,
    priors,
    args=[jlogl, list_name],
)

nsteps = 1000
# burn for 1000 steps
burn = 0
# thin by 5
thin_by = 1
out = ensemble.run_mcmc(coords, nsteps, burn=burn, progress=True, thin_by=thin_by)

samples = ensemble.get_chain()['model_0'].reshape(-1, ndim)
corner.corner(samples, labels=gl.logL.params)
plt.savefig('corner_plot.png')
# # Initialize lists to store log-density values and parameter values
# logdensity_values = []
# param_values = []

# # Define learning rate and initialize optimizer
# learning_rate = 1e-2
# optimizer = optax.optimistic_gradient_descent(learning_rate)

# # Initialize parameters as a numeric JAX array
# opt_state = optimizer.init(params)

# # Optimization loop
# num_steps = 1000  # Number of optimization steps
# list_params = []
# # breakpoint()
# for step in range(num_steps):
#     # Compute the logdensity and gradient
#     logdensity_value, grads = jax.value_and_grad(log_density)(params)
#     # Store the log-density value
#     logdensity_values.append(logdensity_value)
#     # Store the parameter values
#     param_values.append(to_dict(params))

#     # Compute updates and apply to parameters
#     updates, opt_state = optimizer.update(grads, opt_state)
#     params = optax.apply_updates(params, updates)
#     # check if params are in bounds, if not do not update
#     for i, bound in enumerate(bounds):
#         if params[i] < bound[0]:
#             params[i] = bound[0]
#         elif params[i] > bound[1]:
#             params[i] = bound[1]

#     # Optionally print progress
#     if step % 100 == 0:
#         print("-------------------------------")
#         print(f"Step {step}: log-density = {logdensity_value}")
#         # print(f"Gradient: {grads}")

#         # Print dict
#         for key, value in to_dict(params).items():
#             if "cw" or "gw" in key:
#                 print(f"{key}: {value}")

# # Create diagnostic plots
# # Plot log-density values
# plt.figure(figsize=(12, 6))
# plt.plot(logdensity_values, label='Log-Density')
# plt.xlabel('Step')
# plt.ylabel('Log-Density')
# plt.title('Log-Density over Optimization Steps')
# plt.legend()
# plt.savefig('log_density.png')

# # Create separate plots for each parameter
# for key in param_values[0].keys():
#     plt.figure(figsize=(12, 6))
#     plt.plot([params[key] for params in param_values], label=key)
#     plt.xlabel('Step')
#     plt.ylabel(f'{key} Value')
#     plt.title(f'{key} Value over Optimization Steps')
#     plt.legend()
#     plt.savefig(f'Optimization_of_{key}.png')