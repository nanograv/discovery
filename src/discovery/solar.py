import numpy as np
import functools
import inspect
import jax.numpy as jnp

from . import const
from . import matrix
from . import fourierbasis
from . import quantize

AU_light_sec = const.AU / const.c  # 1 AU in light seconds
AU_pc = const.AU / const.pc        # 1 AU in parsecs (for DM normalization)

def theta_impact(psr):
    """From enterprise_extensions: use the attributes of an Enterprise
    Pulsar object to calculate the solar impact angle.
    Returns solar impact angle (rad), distance to Earth (R_earth),
    impact distance (b), perpendicular distance (z_earth)."""

    earth = psr.planetssb[:, 2, :3]
    sun = psr.sunssb[:, :3]
    earthsun = earth - sun

    R_earth = np.sqrt(np.einsum('ij,ij->i', earthsun, earthsun))
    Re_cos_theta_impact = np.einsum('ij,ij->i', earthsun, psr.pos_t)

    theta_impact = np.arccos(-Re_cos_theta_impact / R_earth)
    b = np.sqrt(R_earth**2 - Re_cos_theta_impact**2)

    return theta_impact, R_earth, b, -Re_cos_theta_impact

def make_solardm(psr):
    """From enterprise_extensions: calculate DM
    due to 1/r^2 solar wind density model."""

    theta, r_earth, _, _ = theta_impact(psr)
    shape = matrix.jnparray(AU_light_sec * AU_pc / r_earth / np.sinc(1 - theta/np.pi) * 4.148808e3 / psr.freqs**2)

    def solardm(n_earth):
        return n_earth * shape

    return solardm

def fourierbasis_solar_dm(psr,
                        components,
                        T=None):
    """
    From enterprise_extions: construct DM-Solar Model Fourier design matrix.

    :param psr: Pulsar object
    :param components: Number of Fourier components in the model
    :param T: Total timespan of the data

    :return: F: SW DM-variation fourier design matrix
    :return: f: Sampling frequencies
    """

    # get base Fourier design matrix and frequencies
    f, df, fmat = fourierbasis(psr, components, T)

    dt_DM = make_solardm(psr)

    return f, df, fmat * dt_DM[:, None]

def makegp_timedomain_solar_dm(psr, covariance, dt=1.0, common=[], name='timedomain_sw_gp'):
     argspec = inspect.getfullargspec(covariance)
     argmap = [(arg if arg in common else f'{name}_{arg}' if f'{name}_{arg}' in common else f'{psr.name}_{name}_{arg}')
               for arg in argspec.args if arg not in ['tau']]

     dt_DM = make_solardm(psr) * 4.148808e3 / (psr.freqs**2)
 
     bins = quantize(psr.toas, dt)
     Umat = np.vstack([bins == i for i in range(bins.max() + 1)]).T.astype('d')
     Umat = Umat * dt_DM[:, None] 
     toas = psr.toas @ Umat / Umat.sum(axis=0)
 
     get_tmat = covariance
     tau = jnp.abs(toas[:, jnp.newaxis] - toas[jnp.newaxis, :])
 
     def getphi(params):
         return get_tmat(tau, *[params[arg] for arg in argmap])
     getphi.params = argmap
 
     return matrix.VariableGP(matrix.NoiseMatrix2D_var(getphi), Umat)

def chromaticdelay(toas, freqs, t0, log10_Amp, log10_tau, idx):
    toadays, invnormfreqs = toas / const.day, 1400.0 / freqs
    dt = toadays - t0

    return matrix.jnp.where(dt > 0.0, -1.0 * (10**log10_Amp) * matrix.jnp.exp(-dt / (10**log10_tau)) * invnormfreqs**idx, 0.0)

def make_chromaticdelay(psr, idx=None):
    """From enterprise_extensions: pre-calculate chromatic exponential-dip delay."""

    toadays, invnormfreqs = matrix.jnparray(psr.toas / const.day), matrix.jnparray(1400.0 / psr.freqs)

    def decay(t0, log10_Amp, log10_tau, idx):
        dt = toadays - t0
        return matrix.jnp.where(dt > 0.0, -1.0 * (10**log10_Amp) * matrix.jnp.exp(-dt / (10**log10_tau)) * invnormfreqs**idx, 0.0)

    if idx is not None:
        decay = functools.partial(decay, idx=idx)

    return decay
