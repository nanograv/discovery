import numpy as np
import functools

from discovery import const
from discovery import matrix

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
