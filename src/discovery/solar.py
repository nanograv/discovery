import numpy as np
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

    def solardm(params):
        return params['n_earth'] * shape
    solardm.params = ['n_earth']
    return solardm

def make_chromaticdecay(psr):
    """From enterprise_extensions: calculate chromatic exponential-dip delay."""

    toadays, normfreqs = matrix.jnparray(psr.toas / const.day), matrix.jnparray(1400.0 / psr.freqs)

    def decay(t0, log10_Amp, log10_tau, idx):
        dt = toadays - t0
        return matrix.jnp.where(dt > 0.0, -1.0 * (10**log10_Amp) * matrix.jnp.exp(-dt / (10**log10_tau)) * normfreqs**idx, 0.0)

    return decay

def _dm_solar_close(n_earth, r_earth):
    return (n_earth * AU_light_sec * AU_pc / r_earth)


def _dm_solar(n_earth, theta, r_earth):
    return ((np.pi - theta) *
            (n_earth * AU_light_sec * AU_pc
             / (r_earth * np.sin(theta))))


def dm_solar(n_earth, theta, r_earth):
    """
    Calculates Dispersion measure due to 1/r^2 solar wind density model.
    ::param :n_earth Solar wind proton/electron density at Earth (1/cm^3)
    ::param :theta: angle between sun and line-of-sight to pulsar (rad)
    ::param :r_earth :distance from Earth to Sun in (light seconds).
    See You et al. 2007 for more details.
    """
    return matrix.jnp.where(np.pi - theta >= 1e-5,
                    _dm_solar(n_earth, theta, r_earth),
                    _dm_solar_close(n_earth, r_earth))

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
    theta, R_earth, _, _ = theta_impact(psr)
    dm_sol_wind = dm_solar(1.0, theta, R_earth)
    dt_DM = dm_sol_wind * 4.148808e3 / (psr.freqs**2)

    return f, df, fmat * dt_DM[:, None]

def makegp_timedomain_solar_dm(psr, covariance, dt=1.0, common=[], name='timedomain_sw_gp'):
    argspec = inspect.getfullargspec(covariance)
    argmap = [(arg if arg in common else f'{name}_{arg}' if f'{name}_{arg}' in common else f'{psr.name}_{name}_{arg}')
              for arg in argspec.args if arg not in ['tau']]
    
    # get solar wind ingredients
    theta, R_earth, _, _ = theta_impact(psr)
    dm_sol_wind = dm_solar(1.0, theta, R_earth)
    dt_DM = dm_sol_wind * 4.148808e3 / (psr.freqs**2)

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