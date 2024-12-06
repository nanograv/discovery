import inspect
import glob
import jax
from jax import numpy as jnp
import discovery as ds
from discovery.deterministic import CW_Signal

def load_pulsars(psr_files_pattern):
    """Load pulsar data from files matching the given pattern."""
    psr_files = glob.glob(psr_files_pattern)
    psrs = [ds.Pulsar.read_feather(psr) for psr in psr_files]
    return psrs

def get_timespan(psrs):
    """Get the timespan for the given pulsars."""
    return ds.getspan(psrs)

def mydelay(psr, prior, common=[], name='delay'):
    """Create a delay function for a given pulsar and prior."""
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
        return delay(inp)
    delayfunc.params = argmap

    return delayfunc

def prior(cw_costheta, cw_phi, cw_cosinc, cw_log10_Mc, cw_log10_fgw, cw_log10_h, cw_phase0, cw_psi, **kwargs):
    """Define the prior for the Continuous Wave (CW)."""
    return 0  # Uniform prior

def create_likelihood(psrs, tspan):
    """Create the likelihood function for the given pulsars and timespan."""
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

    # hd = ds.ArrayLikelihood([ds.PulsarLikelihood([psr.residuals,
    #                                             ds.makenoise_measurement(psr, psr.noisedict),
    #                                             ds.makegp_ecorr(psr, psr.noisedict),
    #                                             ds.makegp_timing(psr),
    #                                             mydelay(psr, prior, common=com, name=f'cw')]) for psr in psrs],
    #                         ds.makecommongp_fourier(psrs, ds.powerlaw, 30, T=tspan, name='red_noise'),
    #                         ds.makegp_fourier_global(psrs, ds.powerlaw, ds.hd_orf, 14, T=tspan, name='gw'))
    # breakpoint()
    #print params
    print("sampling parameters",gl.logL.params)
    return gl

def create_prior(params):
    """Create the prior function for the given parameters."""
    return ds.prior.makelogprior_uniform(params, ds.prior.priordict_standard)

def logdensity(x, jlogp, jlogl):
    """Compute the log density (log prior + log likelihood)."""
    return jlogp(x) + jlogl(x)
