import functools

import numpy as np
import jax
import jax.numpy as jnp

from . import matrix
from . import const


def fpc_fast(pos, gwtheta, gwphi):
    x, y, z = pos

    sin_phi = jnp.sin(gwphi)
    cos_phi = jnp.cos(gwphi)
    sin_theta = jnp.sin(gwtheta)
    cos_theta = jnp.cos(gwtheta)

    m_dot_pos = sin_phi * x - cos_phi * y
    n_dot_pos = -cos_theta * cos_phi * x - cos_theta * sin_phi * y + sin_theta * z
    omhat_dot_pos = -sin_theta * cos_phi * x - sin_theta * sin_phi * y - cos_theta * z

    denom = 1.0 + omhat_dot_pos

    fplus = 0.5 * (m_dot_pos**2 - n_dot_pos**2) / denom
    fcross = (m_dot_pos * n_dot_pos) / denom

    return fplus, fcross


def makedelay_binary(pulsarterm=True):
    def delay_binary(toas, pos, log10_h0, log10_f0, ra, sindec, cosinc, psi, phi_earth, phi_psr):
        """BBH residuals from Ellis et. al 2012, 2013"""

        h0 = 10**log10_h0
        f0 = 10**log10_f0

        dec, inc = jnp.arcsin(sindec), jnp.arccos(cosinc)

        # calculate antenna pattern (note: pos is pulsar sky position unit vector)
        fplus, fcross = fpc_fast(pos, 0.5 * jnp.pi - dec, ra)  # careful with dec -> gwtheta conversion

        if pulsarterm:
            phi_avg = 0.5 * (phi_earth + phi_psr)
        else:
            phi_avg = phi_earth

        tref = 86400.0 * 51544.5  # MJD J2000 in seconds

        phase = phi_avg + 2.0 * jnp.pi * f0 * (toas - tref)
        cphase, sphase = jnp.cos(phase), jnp.sin(phase)

        # fix this for no pulsarterm

        if pulsarterm:
            phi_diff = 0.5 * (phi_earth - phi_psr)
            sin_diff = jnp.sin(phi_diff)

            delta_sin =  2.0 * cphase * sin_diff
            delta_cos = -2.0 * sphase * sin_diff
        else:
            delta_sin = sphase
            delta_cos = cphase

        At = -1.0 * (1.0 + jnp.cos(inc)**2) * delta_sin
        Bt =  2.0 * jnp.cos(inc) * delta_cos

        alpha = h0 / (2 * jnp.pi * f0)

        # calculate rplus and rcross
        rplus  = alpha * (-At * jnp.cos(2 * psi) + Bt * jnp.sin(2 * psi))
        rcross = alpha * ( At * jnp.sin(2 * psi) + Bt * jnp.cos(2 * psi))

        # calculate residuals
        res = -fplus * rplus - fcross * rcross

        return res

    if not pulsarterm:
        delay_binary = functools.partial(delay_binary, phi_psr=jnp.nan)

    return delay_binary


def makedelay_binary_phases(pulsarterm=True):
    """Factory for computing cphase and sphase vectors from binary parameters."""
    def delay_binary_phases(toas, log10_f0):
        """Compute cosine and sine phase vectors.

        Returns:
            cphase: cosine phase vector (toas,)
            sphase: sine phase vector (toas,)
        """
        f0 = 10**log10_f0
        tref = 86400.0 * 51544.5  # MJD J2000 in seconds

        # Compute base phase vectors from frequency evolution only
        phase_base = 2.0 * jnp.pi * f0 * (toas - tref)
        cphase = jnp.cos(phase_base)
        sphase = jnp.sin(phase_base)

        return jnp.array([cphase, sphase])

    return delay_binary_phases


def makedelay_binary_coefficients(pulsarterm=True):
    """Factory for computing coefficients to multiply phase vectors."""
    def delay_binary_coefficients(pos, log10_h0, log10_f0, ra, sindec, cosinc, psi, phi_earth, phi_psr):
        """Compute antenna pattern factors and phase coefficients.

        Returns:
            coeffs: dictionary with keys:
                - 'fplus', 'fcross': antenna pattern factors
                - 'rplus_coeff_c', 'rplus_coeff_s': coefficients for cphase/sphase in rplus
                - 'rcross_coeff_c', 'rcross_coeff_s': coefficients for cphase/sphase in rcross

            Full response is reconstructed as:
            res = -fplus * (rplus_coeff_c * cphase + rplus_coeff_s * sphase)
                  -fcross * (rcross_coeff_c * cphase + rcross_coeff_s * sphase)
        """
        h0 = 10**log10_h0
        f0 = 10**log10_f0

        dec, inc = jnp.arcsin(sindec), jnp.arccos(cosinc)

        # calculate antenna pattern (note: pos is pulsar sky position unit vector)
        fplus, fcross = fpc_fast(pos, 0.5 * jnp.pi - dec, ra)  # careful with dec -> gwtheta conversion

        # Calculate coefficients that multiply cphase and sphase
        # Apply addition theorem: cos(phi_avg + phase_base) = cos(phi_avg)*cos(phase_base) - sin(phi_avg)*sin(phase_base)
        # Original cphase_orig = cos(phi_avg + phase_base)
        # Original sphase_orig = sin(phi_avg + phase_base)
        if pulsarterm:
            phi_avg = 0.5 * (phi_earth + phi_psr)
            phi_diff = 0.5 * (phi_earth - phi_psr)

            cos_avg = jnp.cos(phi_avg)
            sin_avg = jnp.sin(phi_avg)
            sin_diff = jnp.sin(phi_diff)

            # cphase_orig = cos_avg * cphase - sin_avg * sphase
            # sphase_orig = sin_avg * cphase + cos_avg * sphase
            # delta_sin =  2.0 * cphase_orig * sin_diff
            # delta_cos = -2.0 * sphase_orig * sin_diff

            c_coeff_sin = 2.0 * cos_avg * sin_diff    # coefficient for cphase in delta_sin
            s_coeff_sin = -2.0 * sin_avg * sin_diff   # coefficient for sphase in delta_sin
            c_coeff_cos = -2.0 * sin_avg * sin_diff   # coefficient for cphase in delta_cos
            s_coeff_cos = -2.0 * cos_avg * sin_diff   # coefficient for sphase in delta_cos
        else:
            # cphase_orig = cos(phi_earth + phase_base) = cos(phi_earth)*cphase - sin(phi_earth)*sphase
            # sphase_orig = sin(phi_earth + phase_base) = sin(phi_earth)*cphase + cos(phi_earth)*sphase
            # delta_sin = sphase_orig
            # delta_cos = cphase_orig
            cos_earth = jnp.cos(phi_earth)
            sin_earth = jnp.sin(phi_earth)

            c_coeff_sin = sin_earth    # coefficient for cphase in delta_sin
            s_coeff_sin = cos_earth    # coefficient for sphase in delta_sin
            c_coeff_cos = cos_earth    # coefficient for cphase in delta_cos
            s_coeff_cos = -sin_earth   # coefficient for sphase in delta_cos

        # At = -1.0 * (1.0 + cos(inc)^2) * delta_sin
        # Bt = 2.0 * cos(inc) * delta_cos
        cos_inc = jnp.cos(inc)
        At_coeff_c = -1.0 * (1.0 + cos_inc**2) * c_coeff_sin
        At_coeff_s = -1.0 * (1.0 + cos_inc**2) * s_coeff_sin
        Bt_coeff_c = 2.0 * cos_inc * c_coeff_cos
        Bt_coeff_s = 2.0 * cos_inc * s_coeff_cos

        alpha = h0 / (2 * jnp.pi * f0)
        cos_2psi = jnp.cos(2 * psi)
        sin_2psi = jnp.sin(2 * psi)

        # rplus = alpha * (-At * cos(2*psi) + Bt * sin(2*psi))
        # rcross = alpha * (At * sin(2*psi) + Bt * cos(2*psi))

        # Coefficient for cphase in rplus: alpha * (-At_coeff_c * cos(2*psi) + Bt_coeff_c * sin(2*psi))
        rplus_coeff_c = alpha * (-At_coeff_c * cos_2psi + Bt_coeff_c * sin_2psi)
        # Coefficient for sphase in rplus: alpha * (-At_coeff_s * cos(2*psi) + Bt_coeff_s * sin(2*psi))
        rplus_coeff_s = alpha * (-At_coeff_s * cos_2psi + Bt_coeff_s * sin_2psi)

        # Coefficient for cphase in rcross: alpha * (At_coeff_c * sin(2*psi) + Bt_coeff_c * cos(2*psi))
        rcross_coeff_c = alpha * (At_coeff_c * sin_2psi + Bt_coeff_c * cos_2psi)
        # Coefficient for sphase in rcross: alpha * (At_coeff_s * sin(2*psi) + Bt_coeff_s * cos(2*psi))
        rcross_coeff_s = alpha * (At_coeff_s * sin_2psi + Bt_coeff_s * cos_2psi)

        Ac = -fplus * rplus_coeff_c - fcross * rcross_coeff_c
        As = -fplus * rplus_coeff_s - fcross * rcross_coeff_s

        return jnp.array([Ac, As])

    if not pulsarterm:
        delay_binary_coefficients = functools.partial(delay_binary_coefficients, phi_psr=jnp.nan)

    return delay_binary_coefficients



def cos2comp(f, df, A, f0, phi, t0):
    """Project signal A * cos(2pi f t + phi) onto Fourier basis
    cos(2pi k t/T), sin(2pi k t/T) for t in [t0, t0+T]."""

    T = 1.0 / df[0]

    Delta_omega = 2.0 * jnp.pi * (f0 - f[::2])
    Sigma_omega = 2.0 * jnp.pi * (f0 + f[::2])

    phase_Delta_start = phi + Delta_omega * t0
    phase_Delta_end   = phi + Delta_omega * (t0 + T)

    phase_Sigma_start = phi + Sigma_omega * t0
    phase_Sigma_end   = phi + Sigma_omega * (t0 + T)

    ck = (A / T) * (
        (jnp.sin(phase_Delta_end) - jnp.sin(phase_Delta_start)) / Delta_omega +
        (jnp.sin(phase_Sigma_end) - jnp.sin(phase_Sigma_start)) / Sigma_omega
    )

    sk = (A / T) * (
        (jnp.cos(phase_Delta_end) - jnp.cos(phase_Delta_start)) / Delta_omega -
        (jnp.cos(phase_Sigma_end) - jnp.cos(phase_Sigma_start)) / Sigma_omega
    )

    return jnp.stack((sk, ck), axis=1).reshape(-1)


def makefourier_binary(pulsarterm=True):
    def fourier_binary(f, df, mintoa, pos, log10_h0, log10_f0, ra, sindec, cosinc, psi, phi_earth, phi_psr):
        """BBH residuals from Ellis et. al 2012, 2013"""

        h0 = 10**log10_h0
        f0 = 10**log10_f0

        dec, inc = jnp.arcsin(sindec), jnp.arccos(cosinc)

        # calculate antenna pattern (note: pos is pulsar sky position unit vector)
        fplus, fcross = fpc_fast(pos, 0.5 * jnp.pi - dec, ra)  # careful with dec -> gwtheta conversion

        if pulsarterm:
            phi_avg  = 0.5 * (phi_earth + phi_psr)
        else:
            phi_avg = phi_earth

        tref = 86400.0 * 51544.5  # MJD J2000 in seconds

        cphase = cos2comp(f, df, 1.0, f0, phi_avg - 2.0 * jnp.pi * f0 * tref, mintoa)
        sphase = cos2comp(f, df, 1.0, f0, phi_avg - 2.0 * jnp.pi * f0 * tref - 0.5*jnp.pi, mintoa)

        # fix this for no pulsarterm

        if pulsarterm:
            phi_diff = 0.5 * (phi_earth - phi_psr)
            sin_diff = jnp.sin(phi_diff)

            delta_sin =  2.0 * cphase * sin_diff
            delta_cos = -2.0 * sphase * sin_diff
        else:
            delta_sin = sphase
            delta_cos = cphase

        At = -1.0 * (1.0 + jnp.cos(inc)**2) * delta_sin
        Bt =  2.0 * jnp.cos(inc) * delta_cos

        alpha = h0 / (2 * jnp.pi * f0)

        # calculate rplus and rcross
        rplus  = alpha * (-At * jnp.cos(2 * psi) + Bt * jnp.sin(2 * psi))
        rcross = alpha * ( At * jnp.sin(2 * psi) + Bt * jnp.cos(2 * psi))

        # calculate residuals
        res = -fplus * rplus - fcross * rcross

        return res

    if not pulsarterm:
        fourier_binary = functools.partial(fourier_binary, phi_psr=jnp.nan)

    return fourier_binary


def chromatic_exponential(psr, fref=1400.0):
    r"""
    Factory function for chromatic exponential delay model.

    Creates a delay function that models chromatic exponential events (e.g., glitches,
    state changes) with frequency-dependent amplitude scaling.

    Parameters
    ----------
    psr : Pulsar
        Pulsar object containing toas and freqs attributes
    fref : float, optional
        Reference frequency in MHz for normalization (default: 1400.0)

    Returns
    -------
    delay : callable
        Function with signature (t0, log10_Amp, log10_tau, sign_param, alpha) -> ndarray
        Computes chromatic exponential delay:

        .. math::

            \Delta(t) = \pm A_0 \exp\left(-\frac{t - t_0}{\tau}\right) \left(\frac{f_{\text{ref}}}{f}\right)^\alpha H(t - t_0)

        where :math:`H(t - t_0)` is the Heaviside step function.
    """
    toas, fnorm = matrix.jnparray(psr.toas / const.day), matrix.jnparray(fref / psr.freqs)

    def delay(t0, log10_Amp, log10_tau, sign_param, alpha):
        r"""
        Compute chromatic exponential delay.

        .. math::

            \Delta(t) = \pm A_0 \exp\left(-\frac{t - t_0}{\tau}\right) \left(\frac{f_{\text{ref}}}{f}\right)^\alpha H(t - t_0)

        Parameters
        ----------
        t0 : float
            Event epoch :math:`t_0` in days (MJD)
        log10_Amp : float
            Log10 of amplitude :math:`A_0` in seconds
        log10_tau : float
            Log10 of exponential decay timescale :math:`\tau` in days
        sign_param : float
            Sign of the delay (positive or negative)
        alpha : float
            Chromatic index :math:`\alpha` (spectral index for frequency dependence)

        Returns
        -------
        delay : ndarray
            Array of timing residuals :math:`\Delta(t)` in seconds with shape matching psr.toas
        """
        return jnp.sign(sign_param) * 10**log10_Amp * jnp.exp(- (toas - t0) / 10**log10_tau) * fnorm**alpha * jnp.heaviside(toas - t0, 1.0)

    delay.__name__ = "chromatic_exponential_delay"
    return delay


def chromatic_annual(psr, fref=1400.0):
    r"""
    Factory function for chromatic annual delay model.

    Creates a delay function that models chromatic annual sinusoidal variations
    (e.g., annual DM variations) with frequency-dependent amplitude scaling.

    Parameters
    ----------
    psr : Pulsar
        Pulsar object containing toas and freqs attributes
    fref : float, optional
        Reference frequency in MHz for normalization (default: 1400.0)

    Returns
    -------
    delay : callable
        Function with signature (log10_Amp, phase, alpha) -> ndarray
        Computes chromatic annual delay:

        .. math::

            \Delta(t) = A_0 \sin(2\pi f_{\text{yr}} t + \phi) \left(\frac{f_{\text{ref}}}{f}\right)^\alpha

        where :math:`f_{\text{yr}}` is the annual frequency (1/year).
    """
    toas, fnorm = matrix.jnparray(psr.toas), matrix.jnparray(fref / psr.freqs)

    def delay(log10_Amp, phase, alpha):
        r"""
        Compute chromatic annual delay.

        .. math::

            \Delta(t) = A_0 \sin(2\pi f_{\text{yr}} t + \phi) \left(\frac{f_{\text{ref}}}{f}\right)^\alpha

        Parameters
        ----------
        log10_Amp : float
            Log10 of amplitude :math:`A_0` in seconds
        phase : float
            Phase offset :math:`\phi` in radians
        alpha : float
            Chromatic index :math:`\alpha` (spectral index for frequency dependence)

        Returns
        -------
        delay : ndarray
            Array of timing residuals :math:`\Delta(t)` in seconds with shape matching psr.toas
        """
        return 10**log10_Amp * jnp.sin(2*jnp.pi * const.fyr * toas + phase) * fnorm**alpha

    delay.__name__ = "chromatic_annual_delay"
    return delay


def chromatic_gaussian(psr, fref=1400.0):
    r"""
    Factory function for chromatic Gaussian delay model.

    Creates a delay function that models chromatic Gaussian events (e.g., transient
    DM variations, localized events) with frequency-dependent amplitude scaling.

    Parameters
    ----------
    psr : Pulsar
        Pulsar object containing toas and freqs attributes
    fref : float, optional
        Reference frequency in MHz for normalization (default: 1400.0)

    Returns
    -------
    delay : callable
        Function with signature (t0, log10_Amp, log10_sigma, sign_param, alpha) -> ndarray
        Computes chromatic Gaussian delay:

        .. math::

            \Delta(t) = \pm A_0 \exp\left(-\frac{(t - t_0)^2}{2\sigma^2}\right) \left(\frac{f_{\text{ref}}}{f}\right)^\alpha
    """
    toas, fnorm = matrix.jnparray(psr.toas / const.day), matrix.jnparray(fref / psr.freqs)

    def delay(t0, log10_Amp, log10_sigma, sign_param, alpha):
        r"""
        Compute chromatic Gaussian delay.

        .. math::

            \Delta(t) = \pm A_0 \exp\left(-\frac{(t - t_0)^2}{2\sigma^2}\right) \left(\frac{f_{\text{ref}}}{f}\right)^\alpha

        Parameters
        ----------
        t0 : float
            Event epoch :math:`t_0` in days (MJD)
        log10_Amp : float
            Log10 of amplitude :math:`A_0` in seconds
        log10_sigma : float
            Log10 of Gaussian width :math:`\sigma` in days
        sign_param : float
            Sign of the delay (positive or negative)
        alpha : float
            Chromatic index :math:`\alpha` (spectral index for frequency dependence)

        Returns
        -------
        delay : ndarray
            Array of timing residuals :math:`\Delta(t)` in seconds with shape matching psr.toas
        """
        return jnp.sign(sign_param) * 10**log10_Amp * jnp.exp(-(toas - t0)**2 / (2 * (10**log10_sigma)**2)) * fnorm**alpha

    delay.__name__ = "chromatic_gaussian_delay"
    return delay


def orthometric_shapiro(psr, binphase):
    r"""
    Factory function for orthometric Shapiro delay model.

    Creates a delay function that models Shapiro delay in binary pulsars using
    the orthometric parameterization from Freire & Wex (2010).

    Parameters
    ----------
    psr : Pulsar
        Pulsar object containing toas attribute
    binphase : array-like
        Binary orbital phase :math:`\Phi` at each TOA (same shape as psr.toas)

    Returns
    -------
    delay : callable
        Function with signature (h3, stig) -> ndarray
        Computes orthometric Shapiro delay (Equation 29 in Freire & Wex 2010):

        .. math::

            \Delta_s = -\frac{2 h_3}{\zeta^3} \log(1 + \zeta^2 - 2 \zeta \sin\Phi)

    Raises
    ------
    ValueError
        If binphase shape does not match psr.toas shape

    References
    ----------
    Freire, P. C. C., & Wex, N. (2010). The orthometric parametrization of the
    Shapiro delay and an improved test of general relativity with binary pulsars.
    MNRAS, 409(1), 199-212.
    """
    toas, binphase = matrix.jnparray(psr.toas / const.day), matrix.jnparray(binphase)
    if not np.shape(binphase) == np.shape(toas):
        raise ValueError("Input binphase must have the same shape as toas")

    def delay(h3, stig):
        r"""
        Compute orthometric Shapiro delay.

        Implements Equation (29) from Freire & Wex (2010):

        .. math::

            \Delta_s = -\frac{2 h_3}{\zeta^3} \log(1 + \zeta^2 - 2 \zeta \sin\Phi)

        Parameters
        ----------
        h3 : float
            Orthometric amplitude parameter :math:`h_3` (related to companion mass and inclination)
        stig : float
            Orthometric shape parameter :math:`\zeta` (related to orbital inclination)

        Returns
        -------
        delay : ndarray
            Shapiro timing delay :math:`\Delta_s` in seconds with shape matching psr.toas
        """
        return -(2.0 * h3 / stig**3) * jnp.log(1 + stig**2 - 2 * stig * jnp.sin(binphase))

    delay.__name__ = "orthometric_shapiro_delay"
    return delay