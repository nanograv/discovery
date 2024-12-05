import inspect
import sys
import pickle, json
import discovery as ds
import jax
from jax import numpy as jnp
sys.path.append('./etudes/')
from etudes.deterministic import CW_Signal

from matplotlib import pyplot as plt

# Choose two pulsars for reproducibility
psr_files = [
    "./data/v1p1_de440_pint_bipm2019-B1855+09.feather",
    "./data/v1p1_de440_pint_bipm2019-B1953+29.feather",
]

# Construct a list of Pulsar objects
psrs = [ds.Pulsar.read_feather(psr) for psr in psr_files]

# Get the timespan
tspan = ds.getspan(psrs)



num_psrs = 2
psrs = psrs[:num_psrs]


tspan = ds.getspan(psrs)


# It is twice as fast per iteration to create this object with generator input
def mydelay(psr, prior, common=[], name='delay'):
    delay = CW_Signal(psr).get_delay
    argspec = inspect.getfullargspec(delay)
    argmap = [(arg if arg in common else f'{name}_{arg}' if f'{name}_{arg}' in common else f'{psr.name}_{name}_{arg}')
              for arg in argspec.args]

    def priorfunc(params):
        return prior(f, df, *[params[arg] for arg in argmap])
    priorfunc.params = argmap

    def delayfunc(params):
        return psr.toas - delay(*[params[arg] for arg in argmap])

    delayfunc.params = argmap

    return delayfunc


gl = ds.GlobalLikelihood((ds.PulsarLikelihood([psrs[ii].residuals,
                                    ds.makenoise_measurement(psrs[ii], psrs[ii].noisedict),
                                    ds.makegp_ecorr(psrs[ii], psrs[ii].noisedict),
                                    ds.makegp_timing(psrs[ii]),
                                    ds.makegp_fourier(psrs[ii], ds.powerlaw, 30, T=tspan, name='red_noise'),
                                    ds.makegp_fourier(psrs[ii], ds.powerlaw, 14, T=tspan, common=['gw_log10_A', 'gw_gamma'], name='gw'),
                                    mydelay(psrs[ii], common=[], name='delay')
                                    ])
                                    for ii in range(len(psrs))))


x0 = ds.prior.sample_uniform(gl.logL.params)
logprior = ds.prior.makelogprior_uniform(gl.logL.params, ds.prior.priordict_standard)
loglike = gl.logL

jlogl = jax.jit(loglike)
jlogp = jax.jit(logprior)

@jax.jit
def logdensity(x):
    # x = map_params(loglike.params, x)
    return jlogp(x) + jlogl(x)

initial_position = ds.prior.sample_uniform(gl.logL.params)
logdensity(initial_position)

logdensity(initial_position)  # ~16 ms per loop
