import numpy as np
import functools

from . import const
from . import matrix

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
    """Create a closure function for calculating chromatic exponential-dip delay.

    This function acts as a factory. It pre-calculates TOA
    and frequency-dependent terms from a pulsar object and returns a new
    function (`decay`). This returned function computes the time delay
    induced by a chromatic exponential event for specific event parameters.

    The closure mechanism works as follows: The inner function `decay` retains
    access to the variables `toadays`, `invnormfreqs`, `ln_10`, and
    `ln_invnormfreqs` calculated in the scope of `make_chromaticdelay`, even
    after `make_chromaticdelay` has finished executing. If `idx` is provided
    to `make_chromaticdelay`, it is also fixed for the returned `decay`
    function using `functools.partial`.

    Parameters
    ----------
    psr : discovery.Pulsar
        Pulsar object.
    idx : float, optional
        The chromatic index defining the delay's radio-frequency dependence.
        If `None` (default), the returned `decay` function will require `idx`
        as an argument. If a float is provided, this value is fixed for the
        returned `decay` function.

    Returns
    -------
    Callable
        A function `decay` that calculates the chromatic delay.
        Its signature depends on whether `idx` was provided to
        `make_chromaticdelay`:

        - If `idx` was provided: `decay(t0, log10_Amp, log10_tau)`
        - If `idx` was `None`: `decay(t0, log10_Amp, log10_tau, idx)`

        The arguments for the returned `decay` function are:
            - `t0` (float): The MJD of the exponential dip event.
            - `log10_Amp` (float): Log10 of the amplitude of the signal (unitless).
              The actual amplitude `A` is `10**log10_Amp`. The amplitude is
              defined at the reference frequency (1400 MHz) and time `t0`.
            - `log10_tau` (float): Log10 of the decay timescale `tau` (in days).
              The actual timescale is `tau = 10**log10_tau`.
            - `idx` (float, required only if `idx` was `None` in the outer function):
              The chromatic index.

        The `decay` function returns an array of delays (in seconds) corresponding
        to each TOA/frequency pair in the `psr` object.

    Notes
    -----
    Originally, this function was not written in log-space. However, it was
    found that large negative numbers in the exponent (`-(t - t_0)/tau`) lead
    to `NaN` values in the gradients of likelihoods that incorporate this delay.

    Now, most of the computation is done in log-space with the intent of stabilizing
    gradients. Also for frequencies less than the normalizing
    frequency (1400 MHz), the exponent is pushed towards more positive values.
    However, this is often not enough to handle all of the possible `NaN`
    values, so we mask out any that remain.

    """
    toadays, invnormfreqs = matrix.jnparray(psr.toas / const.day), matrix.jnparray(1400.0 / psr.freqs)

    # Put quantities into log-space
    ln_10 = matrix.jnp.log(10)
    ln_invnormfreqs = matrix.jnp.log(invnormfreqs)

    def decay(t0, log10_Amp, log10_tau, idx):

        # Note that the usage of `jax.numpy.where` allows for the TOAs to be unordered.
        # We only need to store the differences here.
        dt = toadays - t0

        # Store this mask because we'll use it twice.
        dt_mask = dt > 0.0

        # Get the exponent for the exponential dip. Return 0 if the TOA is before the
        # start time `t0`.
        vals = matrix.jnp.where(
            dt_mask,
            ln_10 * log10_Amp - dt / (10**log10_tau) + idx * ln_invnormfreqs,
            0.0,
        )

        # Filter out any `NaN`s. A few things were tried here (e.g. clipping), but
        # they didn't handle the `NaN`s in the gradient. This, while brute force,
        # does the trick.
        vals = matrix.jnp.where(matrix.jnp.isnan(vals), 0.0, vals)

        return matrix.jnp.where(dt_mask, -1.0 * matrix.jnp.exp(vals), vals)

    if idx is not None:
        decay = functools.partial(decay, idx=idx)

    return decay
