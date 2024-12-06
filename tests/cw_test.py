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
    tref = 5.2e9  # reference time
    delay = CW_Signal(psr, tref=tref, evolve=True).get_delay

    argspec = inspect.getfullargspec(prior)
    argmap = [(arg if arg in common else f'{name}_{arg}' if f'{name}_{arg}' in common else f'{psr.name}_{name}_{arg}')
              for arg in argspec.args]

    def priorfunc(params):
        return prior(*[params[arg] for arg in argmap])
    priorfunc.params = argmap

    def delayfunc(params):
        inp = jnp.asarray([params[arg] for arg in argmap])
        # breakpoint()
        return delay(inp)
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
        mydelay(psr, prior, common=com, name=f'cw')
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
