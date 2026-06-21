#!/usr/bin/env python3
"""Tests for external-basis deterministic signals (Route A): ExtSignal,
make_extsignal_fourier, makecw_extsignal, and the cross-terms wired into
VectorWoodburyKernel_varP.make_kernelproduct_gpcomponent.

An ExtSignal carries a deterministic signal on its OWN Fourier basis (e.g. a
continuous wave needing higher frequencies than the GP bases reach). It has no
prior; the likelihood folds it in via GP-CW cross-terms.
"""

from pathlib import Path
import pytest

import numpy as np
import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)

import discovery as ds

DATA = Path(__file__).resolve().parent.parent / "data"
PSR_NAMES = ["B1855+09", "B1937+21", "B1953+29"]


def _read_psrs():
    return [ds.Pulsar.read_feather(DATA / f"v1p1_de440_pint_bipm2019-{n}.feather")
            for n in PSR_NAMES]


def _psls(psrs):
    return [ds.PulsarLikelihood([p.residuals,
            ds.makenoise_measurement(p, noisedict=p.noisedict, ecorr=True),
            ds.makegp_timing(p, svd=True)]) for p in psrs]


def _set_cw(pars, names, rng, log10_h0=-14.0):
    """Fill CW parameters with a sane, non-degenerate point."""
    for k in names:
        if k.endswith('log10_h0'):
            pars[k] = log10_h0
        elif k.endswith('log10_f0'):
            pars[k] = -7.7
        elif k.endswith('sindec') or k.endswith('cosinc'):
            pars[k] = 0.3
        elif k.endswith('phi_psr'):          # distinct per pulsar (avoids the
            pars[k] = rng.uniform(0, 2 * np.pi)   # phi_earth==phi_psr cancellation)
        elif k == 'cw_phi_earth':
            pars[k] = 1.7
        else:
            pars[k] = 0.5
    return pars


@pytest.mark.integration
class TestExtSignal:

    def test_crossterm_bruteforce(self):
        """The extsignal cross-terms reproduce a brute-force residual model.

        clogL(F c + F_cw c_cw) - clogL(F c) must equal the direct
        -0.5 (y-Fc-F_cw c_cw)^T N^-1 (y-Fc-F_cw c_cw) + 0.5 (y-Fc)^T N^-1 (y-Fc).
        """
        psrs = _read_psrs()
        T = ds.getspan(psrs)
        n_rn, n_cw = 20, 40
        rng = np.random.default_rng(0)
        np.random.seed(0)

        cw = ds.makecw_extsignal(psrs, components=n_cw, T=T)
        common = dict(commongp=ds.makecommongp_fourier(
            psrs, ds.powerlaw, components=n_rn, T=T, name='red_noise'))
        m_base = ds.ArrayLikelihood(_psls(psrs), **common)
        m_ext = ds.ArrayLikelihood(_psls(psrs), extsignals=[cw], **common)

        pars = ds.sample_uniform(m_base.logL.params)
        for p in psrs:                       # physical Fourier amplitudes ~1e-6
            pars[f'{p.name}_red_noise_coefficients({n_rn*2})'] = \
                1e-6 * rng.standard_normal(2 * n_rn)
        _set_cw(pars, cw.params, rng, log10_h0=-14.5)

        delta_code = float(m_ext.clogL(pars)) - float(m_base.clogL(pars))

        ccw = cw.coeffs(pars)
        delta_brute = 0.0
        for i, (F, N, y) in enumerate(zip(m_ext.vsm.Fs, m_ext.vsm.Ns, m_ext.ys)):
            c_i = jnp.asarray(pars[f'{psrs[i].name}_red_noise_coefficients({n_rn*2})'])
            r_gp = y - F @ c_i
            r_all = r_gp - cw.Fs[i] @ ccw[i]
            delta_brute += (-0.5 * (r_all @ N.solve_1d(r_all)[0])
                            + 0.5 * (r_gp @ N.solve_1d(r_gp)[0]))
        delta_brute = float(delta_brute)

        assert abs(delta_code - delta_brute) / abs(delta_brute) < 1e-9

    def test_zero_amplitude_is_identity(self):
        """A negligible-amplitude CW leaves clogL bit-identical to no extsignal."""
        psrs = _read_psrs()
        T = ds.getspan(psrs)
        n_rn, n_gw, n_cw = 24, 14, 60
        rng = np.random.default_rng(2)
        np.random.seed(2)

        def build(**kw):
            return ds.ArrayLikelihood(
                _psls(psrs),
                commongp=ds.makecommongp_fourier(psrs, ds.powerlaw,
                                                 components=n_rn, T=T, name='red_noise'),
                globalgp=ds.makeglobalgp_fourier(psrs, ds.powerlaw, ds.hd_orf,
                                                 components=n_gw, T=T, name='gw'),
                decenter=True, **kw)

        cw = ds.makecw_extsignal(psrs, components=n_cw, T=T)
        m_base = build()
        m_ext = build(extsignals=[cw])

        pars = ds.sample_uniform(m_base.logL.params)
        for p in psrs:
            pars[f'{p.name}_gw_coefficients({n_gw*2})'] = rng.standard_normal(2 * n_gw)
            pars[f'{p.name}_red_noise_coefficients({n_rn*2})'] = \
                rng.standard_normal(2 * n_rn)
        _set_cw(pars, cw.params, rng, log10_h0=-30.0)   # negligible strain

        assert float(m_base.clogL(pars)[0]) == float(m_ext.clogL(pars)[0])

    def test_params_gradient_and_prior_isolation(self):
        """CW params appear in clogL.params, gradients flow, a loud CW moves
        clogL, and the CW params never enter the GP prior."""
        psrs = _read_psrs()
        T = ds.getspan(psrs)
        n_rn, n_gw, n_cw = 24, 14, 60
        rng = np.random.default_rng(3)
        np.random.seed(3)

        cw = ds.makecw_extsignal(psrs, components=n_cw, T=T)
        m = ds.ArrayLikelihood(
            _psls(psrs),
            commongp=ds.makecommongp_fourier(psrs, ds.powerlaw, components=n_rn,
                                             T=T, name='red_noise'),
            globalgp=ds.makeglobalgp_fourier(psrs, ds.powerlaw, ds.hd_orf,
                                             components=n_gw, T=T, name='gw'),
            decenter=True, extsignals=[cw])

        clogl = m.clogL
        # CW params are recognized parameters ...
        assert set(cw.params).issubset(set(clogl.params))
        # ... but never enter the GP prior.
        assert set(cw.params).isdisjoint(set(m.vsm.prior.params))

        # logL refuses when extsignals is set; pull hyperparam names from clogL
        hyperparams = [k for k in clogl.params
                       if '_coefficients(' not in k and k not in set(cw.params)]
        pars = ds.sample_uniform(hyperparams)
        for p in psrs:
            pars[f'{p.name}_gw_coefficients({n_gw*2})'] = rng.standard_normal(2 * n_gw)
            pars[f'{p.name}_red_noise_coefficients({n_rn*2})'] = \
                rng.standard_normal(2 * n_rn)

        quiet = _set_cw(dict(pars), cw.params, rng, log10_h0=-30.0)
        loud = _set_cw(dict(pars), cw.params, rng, log10_h0=-13.0)
        assert abs(float(clogl(loud)[0]) - float(clogl(quiet)[0])) > 1.0

        g = jax.grad(lambda pp: clogl(pp)[0])(loud)
        gnorm = float(sum(np.sum(np.asarray(g[k])**2) for k in cw.params)) ** 0.5
        assert np.isfinite(gnorm) and gnorm > 0.0

    def test_higher_frequency_basis(self):
        """make_extsignal_fourier with more components reaches a higher maximum
        frequency (bin spacing fixed at 1/T_obs)."""
        psrs = _read_psrs()
        T = ds.getspan(psrs)

        cw_lo = ds.makecw_extsignal(psrs, components=20, T=T)
        cw_hi = ds.makecw_extsignal(psrs, components=80, T=T)

        # more components -> wider design matrix, same pulsar order
        assert cw_hi.Fs[0].shape[1] == 2 * 80
        assert cw_lo.Fs[0].shape[1] == 2 * 20
        assert cw_hi.Fs[0].shape[0] == cw_lo.Fs[0].shape[0]   # same toas
