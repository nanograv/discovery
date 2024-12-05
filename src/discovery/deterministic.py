# -*- coding: utf-8 -*-
"""Contains custom PyTree object for deterministic signal modeling. Continuous wave
delay functions (full frequency evolution vs phase approximation) are defined
at class initialization.
"""

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
from . import const


def create_gw_antenna_pattern(pos, gwtheta, gwphi):
    """Function to create pulsar antenna pattern functions as defined
    in Ellis, Siemens, and Creighton (2012).
    :param pos: Unit vector from Earth to pulsar
    :type pos: array-like
    :param gwtheta: GW polar angle in radians
    :type gwtheta: float
    :param gwphi: GW azimuthal angle in radians
    :type gwphi: float

    :return: (fplus, fcross, cosMu), where fplus and fcross
             are the plus and cross antenna pattern functions
             and cosMu is the cosine of the angle between the
             pulsar and the GW source.
    :rtype: tuple
    """

    # use definition from Sesana et al 2010 and Ellis et al 2012
    m = jnp.array([jnp.sin(gwphi), -jnp.cos(gwphi), 0.0])
    n = jnp.array(
        [
            -jnp.cos(gwtheta) * jnp.cos(gwphi),
            -jnp.cos(gwtheta) * jnp.sin(gwphi),
            jnp.sin(gwtheta),
        ]
    )
    omhat = jnp.array(
        [
            -jnp.sin(gwtheta) * jnp.cos(gwphi),
            -jnp.sin(gwtheta) * jnp.sin(gwphi),
            -jnp.cos(gwtheta),
        ]
    )

    fplus = (
        0.5 * (jnp.dot(m, pos) ** 2 - jnp.dot(n, pos) ** 2) / (1 + jnp.dot(omhat, pos))
    )
    fcross = (jnp.dot(m, pos) * jnp.dot(n, pos)) / (1 + jnp.dot(omhat, pos))
    cosMu = -jnp.dot(omhat, pos)

    return fplus, fcross, cosMu


@register_pytree_node_class
class CW_Signal(object):
    """Class for single-pulsar deterministic continuous-wave signals. Main
    output function is `get_delay`, with different forms of the
    function defined at initialization of the class.

    :param psr: A Pulsar object containing pulsar TOAs and residuals
    :type psr: :class:`enterprise.pulsar.Pulsar`
    :param tref: The reference time for phase and frequency [s]
    :type tref: float, optional
    :param evolve: Whether to use full frequency evolution or phase approximation
    :type evolve: bool, optional
    """

    def __init__(self, psr, tref=0, evolve=True):
        """Constructor method"""
        self.psr = psr
        self.tref = tref
        self.evolve = evolve

        self._init_delay(evolve=self.evolve)

    def _init_delay(self, evolve=True):
        """Set the form of the delay function"""
        if evolve:
            self._freqevol_fn = self._full_evolve
        else:
            self._freqevol_fn = self._phase_approx

    def _full_evolve(self, w0, mc, toas, phase0):
        """Full frequency and phase evolution"""
        omega = w0 * (1 - 256 / 5 * mc ** (5 / 3) * w0 ** (8 / 3) * toas) ** (-3 / 8)
        phase = phase0 + 1 / 32 / mc ** (5 / 3) * (w0 ** (-5 / 3) - omega ** (-5 / 3))
        return omega, phase

    def _phase_approx(self, w0, mc, toas, phase0):
        """Phase approximation across observational timespan"""
        omega = w0
        phase = phase0 + omega * toas
        return omega, phase

    def get_delay(self, pars, **kwargs):
        """Call underlying `_get_delay` function for input parameters"""
        return self._get_delay(**pars, **kwargs)

    @jax.jit
    def _get_delay(
        self,
        cw_costheta=0,
        cw_phi=0,
        cw_cosinc=0,
        cw_log10_Mc=9,
        cw_log10_fgw=-8,
        cw_log10_h=-15,
        cw_phase0=0,
        cw_psi=0,
        **kwargs,
    ):
        """Generalized function to compute GW induced residuals from a SMBHB,
        defined in Ellis et. al 2012, 2013.

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

        # convert all units to time
        mc = 10**cw_log10_Mc * const.Tsun
        fgw = 10**cw_log10_fgw
        gwtheta = jnp.arccos(cw_costheta)
        inc = jnp.arccos(cw_cosinc)

        dist = 2 * mc ** (5 / 3) * (jnp.pi * fgw) ** (2 / 3) / 10**cw_log10_h

        # calculate antenna pattern and cosMu
        fplus, fcross, _ = create_gw_antenna_pattern(self.psr._pos, gwtheta, cw_phi)

        # subtract reference time from TOAs
        toas = self.psr.toas - self.tref

        # orbital frequency and phase
        w0 = jnp.pi * fgw
        phase0 = cw_phase0 / 2

        # calculate frequency and phaes evolution
        omega, phase = self._freqevol_fn(w0, mc, toas, phase0)

        # define time dependent coefficients and amplitudes
        At = -0.5 * jnp.sin(2 * phase) * (3 + jnp.cos(2 * inc))
        Bt = 2 * jnp.cos(2 * phase) * jnp.cos(inc)
        alpha = mc ** (5.0 / 3.0) / (dist * omega ** (1.0 / 3.0))

        # calculate rplus and rcross
        rplus = alpha * (-At * jnp.cos(2 * cw_psi) + Bt * jnp.sin(2 * cw_psi))
        rcross = alpha * (At * jnp.sin(2 * cw_psi) + Bt * jnp.cos(2 * cw_psi))

        # calculate residuals
        res = -fplus * rplus - fcross * rcross

        return res

    # Necessary flatten and unflatten methods to register class
    # as a PyTree
    def tree_flatten(self):
        """Method for flattening custom PyTree"""
        return (), (self.psr, self.tref, self.evolve)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Method for reconstructing custom PyTree"""
        return cls(*children, *aux_data)
