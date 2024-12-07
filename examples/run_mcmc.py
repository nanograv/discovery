""" 
Example of running MCMC with Eryn on NG data
nohup python run_mcmc.py > out_mcmc.log &
"""
# set CUDA device
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
psr_files_pattern = "../data/v1p1_de440_pint_bipm2019-*.feather"
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
jlogl(initial_position)
jlogp(initial_position)

# Create log density function
def log_density(xs, jlogl, params_dict, *args, **kwargs):
    map_x = {el: xs[ii] for ii, el in enumerate(params_dict.keys())}
    # time1 = time.time()
    out = jlogl(map_x)
    # make sure that the log likelihood is finite
    out = jnp.where(jnp.isnan(out), -1e50, out)
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
nwalkers = 64
ntemps = 4

# fill kwargs dictionary
tempering_kwargs=dict(ntemps=ntemps)

coords = priors.rvs(size=(ntemps, nwalkers))
dict_name = {param: val for param, val in zip(gl.logL.params, initial_position)}

# test speed of log_density
log_density(coords[0][0], jlogl, dict_name)
for nw in range(2):
    for nt in range(2):
        start_time = time.time()
        log_density(coords[nt][nw], jlogl, dict_name)
        end_time = time.time()
        print(f"log_density execution time: {end_time - start_time} seconds, {1/((end_time - start_time) * nwalkers * ntemps)} it/s expected")

vmap_logd = jax.vmap(log_density, in_axes=(0, None, None), out_axes=0)
# consider using jax.pmap

# test speed of vectorized log_density
temp_coords = priors.rvs(size=(ntemps * nwalkers))

def vectorized_log_density(xs, jlogl, params_dict, *args, **kwargs):
    # make sure the dimensions of input are always the same otherwise it will need to compile every time for different input shapes
    temp_coords[:xs.shape[0]] = xs
    out = vmap_logd(temp_coords, jlogl, params_dict)
    out = np.asarray(out[:xs.shape[0]])
    return out

vectorized_log_density(priors.rvs(size=(ntemps*nwalkers)), jlogl, dict_name)

# compilation speed for different number of entries
start_time = time.time()
vectorized_log_density(priors.rvs(size=(nwalkers*ntemps)), jlogl, dict_name)
end_time = time.time()
print(f"vmap_logd execution time: {end_time - start_time} seconds, {1/((end_time - start_time))} it/s expected")

# breakpoint()
# Run MCMC
ensemble = EnsembleSampler(
    nwalkers,
    ndim,
    vectorized_log_density,
    priors,
    args=[jlogl, dict_name],
    moves=[StretchMove(live_dangerously=True, use_gpu=True)],
    tempering_kwargs=tempering_kwargs,
    backend=f'output_nwalkers{nwalkers}_ntemps{ntemps}.h5',
    vectorize=True
)

nsteps = 1000
thin_by = 1
out = ensemble.run_mcmc(coords, nsteps, burn=0, progress=True, thin_by=thin_by)

# Extract samples and create a corner plot
for temp in range(1):
    print(temp + 1)
    ind = np.asarray([ii for ii, val in enumerate(gl.logL.params) if ('cw' in val)or('gw' in val)])
    samples = [ensemble.get_chain()['model_0'][:, temp, :, :, ii].flatten() for ii in ind]
    labels = np.array([str(gl.logL.params[ii]) for ii in ind])
    # check shape samples and labels
    plt.figure()
    corner.corner(samples, labels=labels)
    plt.savefig(f'corner_plot_{temp}.png')
