import time
import jax
import re
import numpy as np
from mcmc_utils import *
from eryn.ensemble import EnsembleSampler
from eryn.state import State
from eryn.prior import ProbDistContainer, uniform_dist
from eryn.moves import StretchMove
from eryn.utils import TransformContainer
import corner
import matplotlib.pyplot as plt

# Load data
psr_files_pattern = "../data/v1p1_de440_pint_bipm2019-B*.feather"
psrs = load_pulsars(psr_files_pattern)
tspan = get_timespan(psrs)

# Create likelihood and prior
gl = create_likelihood(psrs, tspan)
initial_position = ds.prior.sample_uniform(gl.logL.params)
logprior = create_prior(gl.logL.params)
loglike = gl.logL

# JIT compile log prior and likelihood
jlogl = jax.jit(loglike)
jlogp = jax.jit(logprior)

# Create log density function
def log_density(xs, jlogl, params_dict):
    map_x = {el: xs[ii] for ii, el in enumerate(params_dict.keys())}
    # time1 = time.time()
    out = jlogl(map_x)
    # time2 = time.time()
    # print(f"Time to compute log likelihood: {time2 - time1}")
    # if time2 - time1 > 0.01:
    #     print(f"val cw: {map_x['cw_costheta']}, {map_x['cw_phi']}, {map_x['cw_cosinc']}, {map_x['cw_log10_Mc']}, {map_x['cw_log10_fgw']}, {map_x['cw_log10_h']}, {map_x['cw_phase0']}, {map_x['cw_psi']}")
    #     print(f"val gw: {map_x['gw_log10_A']}, {map_x['gw_gamma']}")
    return out

# Gradient of the log density
grads = jax.grad(log_density)

# Define parameter bounds
bounds = []
for par in initial_position:
    for parname, ranges in ds.prior.priordict_standard.items():
        if re.match(parname, par):
            bounds.append(ranges)

# Initialize Eryn sampler
ndim = len(gl.logL.params)
priors_in = {i: uniform_dist(bound[0], bound[1]) for i, bound in enumerate(bounds)}
priors = ProbDistContainer(priors_in)
nwalkers = 32
ntemps = 6

# fill kwargs dictionary
tempering_kwargs=dict(ntemps=ntemps)

coords = priors.rvs(size=(ntemps, nwalkers))
dict_name = {param: val for param, val in zip(gl.logL.params, initial_position)}

# test speed of log_density
log_density(coords[0][0], jlogl, dict_name)
for nw in range(nwalkers):
    for nt in range(ntemps):

        start_time = time.time()
        log_density(coords[nt][nw], jlogl, dict_name)
        end_time = time.time()
        print(f"log_density execution time: {end_time - start_time} seconds, {(end_time - start_time) * nwalkers * ntemps}")

# breakpoint()
# Run MCMC
ensemble = EnsembleSampler(
    nwalkers,
    ndim,
    log_density,
    priors,
    args=[jlogl, dict_name],
    moves=[StretchMove(live_dangerously=True)],
    tempering_kwargs=tempering_kwargs,
)

nsteps = 1000
thin_by = 1
out = ensemble.run_mcmc(coords, nsteps, burn=0, progress=True, thin_by=thin_by)

# Extract samples and create a corner plot
for temp in range(ntemps):
    print(temp + 1)
    samples = ensemble.get_chain()['model_0'][:, temp].reshape(-1, ndim)
    corner.corner(samples, labels=gl.logL.params)
    plt.savefig(f'corner_plot_{temp}.png')
