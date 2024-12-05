import inspect
import sys
import pickle, json
import discovery as ds
import jax
from jax import numpy as jnp
from discovery.deterministic import CW_Signal

from matplotlib import pyplot as plt
import time

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
        return psr.toas - delay(*inp)
    delayfunc.params = argmap

    return delayfunc

def prior(cw_costheta,
          cw_phi,
          cw_cosinc,
          cw_log10_Mc,
          cw_log10_fgw,
          cw_log10_h,
          cw_phase0,
          cw_psi,
          **kwargs):
    """prior of the CW.

    :param cw_costheta: Cosine of the GW source polar angle in
        celestial coordinates [radians]
    :type cw_costheta: float, optional
    :param cw_phi: GW source azimuthal angle in celestial
        coordinates [radians]
    :type cw_phi: float, optional
    :param cw_cosinc: Cosine of the inclination of the GW source [radians]
    :type cw_cosinc: float, optional
    :param cw_log10_Mc: log10 of the SMBHB chirp mass [solar masses]
    :type cw_log10_Mc: float, optional
    :param cw_log10_fgw: log10 of the GW frequency [Hz]
    :type cw_log10_fgw: float, optional
    :param cw_log10_h: log10 of the GW strain
    :type cw_log10_h: float, optional
    :param cw_phase0: Initial phase of the GW source [radians]
    :type cw_phase0: float, optional
    :param cw_psi: Polarization angle of the GW source [radians]
    :type cw_psi: float, optional

    :return: GW induced residuals from continuous wave source
    :rtype: array-like
    """

    return 0

com = ['cw_costheta', 'cw_phi', 'cw_cosinc', 'cw_log10_Mc', 'cw_log10_fgw', 'cw_log10_h', 'cw_phase0', 'cw_psi']
## Priors (`prior.py`)
# - `makelogprior_uniform(params, [priordict])`: returns a function `logprior(params)` that evaluates the total log prior according to `priordict` (given, e.g., as `{'FourierGP_log10_A': [-18, -11]'}`). Some standard [`enterprise_extensions`](https://github.com/nanograv/enterprise_extensions) priors are included by default (e.g., `crn_log10_A, crn_gamma, gw_log10_A, gw_gamma, ...`). Parameters that are in the list `params` but are not in `priordict` and have no default are not included in the computation.
# - `sample_uniform(params, [priordict])`: creates a dictionary of random values for the parameters in the list `params`, using the uniform priors in `priordict` or in the system default. Fails if a parameter in `params` has no specification.
# ds.prior
gl = ds.GlobalLikelihood((ds.PulsarLikelihood([psrs[ii].residuals,
                                    ds.makenoise_measurement(psrs[ii], psrs[ii].noisedict),
                                    ds.makegp_ecorr(psrs[ii], psrs[ii].noisedict),
                                    ds.makegp_timing(psrs[ii]),
                                    ds.makegp_fourier(psrs[ii], ds.powerlaw, 30, T=tspan, name='red_noise'),
                                    ds.makegp_fourier(psrs[ii], ds.powerlaw, 14, T=tspan, common=['gw_log10_A', 'gw_gamma'], name='gw'),
                                    mydelay(psrs[ii], prior, common=com, name='delay')
                                    ])
                                    for ii in range(len(psrs))))


print("params", gl.logL.params)
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

start_time = time.time()
logdensity(initial_position)  # ~16 ms per loop
end_time = time.time()

print(f"logdensity execution time: {end_time - start_time} seconds")
