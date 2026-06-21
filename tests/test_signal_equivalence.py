#!/usr/bin/env python3
"""Equivalence tests for the deterministic-Fourier-signal knobs:
``means`` (on commongp; enters both ``logL`` via prior-mean shift and ``clogL``
via the centered prior penalty) and ``extsignals`` (on ArrayLikelihood; enters
``clogL`` via cross-terms on its own basis).

When the deterministic signal lives on the GP's basis, both implement the same
mathematical operation: adding ``F a`` to the residual model. We verify this by
showing each path reduces to the same y-shifted reference model
(y -> y - F a, plain GP, no deterministic signal).
"""

from pathlib import Path

import numpy as np
import pytest
import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)

import discovery as ds
from discovery import matrix


DATA = Path(__file__).resolve().parent.parent / "data"
PSR_NAME = "B1855+09"
NCOMP = 5

# fixed deterministic Fourier-coefficient signal (shape 2*NCOMP)
A_VEC = jnp.array([1e-7, 5e-8, -2e-8, 1e-8, -5e-9, 3e-9, 0., 0., 0., 0.])

# fixed (non-degenerate) hyperparameters keep these tests independent of any
# prior sampler; the equivalences hold for arbitrary hyperparameters.
HYPERPARAMS = {
    f'{PSR_NAME}_rn_gamma': 3.5,
    f'{PSR_NAME}_rn_log10_A': -14.5,
}


def _setup():
    """Common ingredients: pulsar, GP basis F, original y, shifted y."""
    psr = ds.Pulsar.read_feather(
        str(DATA / f"v1p1_de440_pint_bipm2019-{PSR_NAME}.feather"))
    T = ds.getspan([psr])
    _, _, fmat = ds.fourierbasis(psr, NCOMP, T=T)
    F_rn = np.asarray(fmat)                          # (ntoa, 2*NCOMP)
    y_orig = np.asarray(psr.residuals)
    y_shifted = y_orig - F_rn @ np.asarray(A_VEC)    # y - F a
    return psr, T, F_rn, y_orig, y_shifted


def _psrl(psr, y):
    """PulsarLikelihood with timing only; red noise lives on the commongp."""
    return ds.PulsarLikelihood([
        y,
        ds.makenoise_measurement(psr, noisedict=psr.noisedict, ecorr=True),
        ds.makegp_timing(psr, svd=True),
    ])


def _commongp(psr, T, with_means=False):
    """The shared commongp -- optionally with ``means`` set to A_VEC."""
    def mean_fn(f, df):
        return A_VEC
    kw = dict(means=mean_fn) if with_means else {}
    return ds.makecommongp_fourier([psr], ds.powerlaw, components=NCOMP,
                                   T=T, name='rn', **kw)


def _zero_coefficients(m, hyperparams):
    """clogL parameter dict: hyperparams + every ``*_coefficients(N)`` zeroed."""
    pars = {**hyperparams}
    for k in m.clogL.params:
        if '_coefficients(' in k:
            n = int(k.split('(')[1].rstrip(')'))
            pars[k] = jnp.zeros(n)
    return pars


def _clogl(m, pars):
    """Unwrap clogL: some paths return ``(value, c)`` when staged."""
    r = m.clogL(pars)
    return float(r[0]) if isinstance(r, tuple) else float(r)


def test_means_reduces_to_y_shifted_reference():
    """``logL`` with ``means=a`` matches the marginalized ``logL`` of the same
    model with y replaced by y - F a and no mean. Verifies the ``means`` path
    in ``make_kernelproduct_varN``."""
    psr, T, _, y_orig, y_shifted = _setup()

    m_means = ds.ArrayLikelihood([_psrl(psr, y_orig)],
                                 commongp=_commongp(psr, T, with_means=True))
    m_shift = ds.ArrayLikelihood([_psrl(psr, y_shifted)],
                                 commongp=_commongp(psr, T, with_means=False))

    v_means = float(m_means.logL(HYPERPARAMS))
    v_shift = float(m_shift.logL(HYPERPARAMS))

    # sanity: the deterministic signal does shift the likelihood
    v_plain = float(ds.ArrayLikelihood(
        [_psrl(psr, y_orig)], commongp=_commongp(psr, T)).logL(HYPERPARAMS))
    assert abs(v_means - v_plain) > 0.1, "means should change logL"

    assert np.allclose(v_means, v_shift, atol=1e-8, rtol=1e-10), \
        f"means {v_means} != y-shifted {v_shift}"


def test_extsignals_same_basis_reduces_to_y_shifted_reference():
    """``clogL`` at c=0 with ``extsignals=[es]`` (where ``es.Fs`` equals the
    GP basis and ``es.coeffs`` returns ``a``) matches ``clogL`` at c=0 of the
    same model with y replaced by y - F a and no extsignal."""
    psr, T, F_rn, y_orig, y_shifted = _setup()

    def cw_coeffs(params):
        return A_VEC.reshape(1, -1)
    cw_coeffs.params = []

    extsig = matrix.ExtSignal([F_rn], cw_coeffs, name='extsig_match_gp')

    m_ext = ds.ArrayLikelihood([_psrl(psr, y_orig)],
                               commongp=_commongp(psr, T),
                               extsignals=[extsig])
    m_shift = ds.ArrayLikelihood([_psrl(psr, y_shifted)],
                                 commongp=_commongp(psr, T))

    pars_ext = _zero_coefficients(m_ext, HYPERPARAMS)
    pars_shift = _zero_coefficients(m_shift, HYPERPARAMS)

    v_ext = _clogl(m_ext, pars_ext)
    v_shift = _clogl(m_shift, pars_shift)

    m_plain = ds.ArrayLikelihood([_psrl(psr, y_orig)], commongp=_commongp(psr, T))
    v_plain = _clogl(m_plain, _zero_coefficients(m_plain, HYPERPARAMS))
    assert abs(v_ext - v_plain) > 0.1, "extsignal should change clogL"

    assert np.allclose(v_ext, v_shift, atol=1e-8, rtol=1e-10), \
        f"extsignals(F_cw=F_gp) {v_ext} != y-shifted {v_shift}"


def test_means_and_extsignals_same_basis_are_consistent():
    """Direct head-to-head: both ``means`` and ``extsignals(F_cw=F_gp)``
    reduce to the same y-shifted reference, by transitivity implementing the
    same deterministic Fourier signal on the GP's basis."""
    psr, T, F_rn, y_orig, y_shifted = _setup()

    v_means = float(ds.ArrayLikelihood(
        [_psrl(psr, y_orig)],
        commongp=_commongp(psr, T, with_means=True)).logL(HYPERPARAMS))

    def cw_coeffs(params):
        return A_VEC.reshape(1, -1)
    cw_coeffs.params = []
    extsig = matrix.ExtSignal([F_rn], cw_coeffs, name='extsig_match_gp')
    m_ext = ds.ArrayLikelihood([_psrl(psr, y_orig)],
                               commongp=_commongp(psr, T),
                               extsignals=[extsig])
    m_ext_shift = ds.ArrayLikelihood([_psrl(psr, y_shifted)],
                                     commongp=_commongp(psr, T))

    v_ext_at_c0 = _clogl(m_ext, _zero_coefficients(m_ext, HYPERPARAMS))
    v_ref_at_c0 = _clogl(m_ext_shift,
                         _zero_coefficients(m_ext_shift, HYPERPARAMS))
    v_ref_logL = float(m_ext_shift.logL(HYPERPARAMS))

    assert np.allclose(v_means, v_ref_logL, atol=1e-8, rtol=1e-10)
    assert np.allclose(v_ext_at_c0, v_ref_at_c0, atol=1e-8, rtol=1e-10)


def test_means_in_clogL_matches_extsignals_same_basis():
    """Tightly tests the means->clogL wiring: ``clogL_means`` at ``c = a0``
    (prior penalty vanishes, data uses c=a0) matches ``clogL_extsig`` at
    ``c = 0`` (extcontrib supplies the same data shift)."""
    psr, T, F_rn, y_orig, _ = _setup()

    m_means = ds.ArrayLikelihood([_psrl(psr, y_orig)],
                                 commongp=_commongp(psr, T, with_means=True))

    def cw_coeffs(params):
        return A_VEC.reshape(1, -1)
    cw_coeffs.params = []
    extsig = matrix.ExtSignal([F_rn], cw_coeffs, name='extsig_match_gp')
    m_ext = ds.ArrayLikelihood([_psrl(psr, y_orig)],
                               commongp=_commongp(psr, T),
                               extsignals=[extsig])

    pars_means = {**HYPERPARAMS}
    coef_key = f'{PSR_NAME}_rn_coefficients(10)'
    pars_means[coef_key] = A_VEC

    pars_ext = _zero_coefficients(m_ext, HYPERPARAMS)

    v_means = _clogl(m_means, pars_means)
    v_ext = _clogl(m_ext, pars_ext)

    assert np.allclose(v_means, v_ext, atol=1e-8, rtol=1e-10), \
        f"means at c=a0 ({v_means}) != extsignals at c=0 ({v_ext})"


def test_extsignals_with_logL_raises():
    """``extsignals`` is wired into ``clogL`` only -- the marginalized ``logL``
    path silently dropped them before. We now raise NotImplementedError."""
    psr, T, F_rn, y_orig, _ = _setup()

    def cw_coeffs(params):
        return A_VEC.reshape(1, -1)
    cw_coeffs.params = []
    extsig = matrix.ExtSignal([F_rn], cw_coeffs, name='extsig')

    m = ds.ArrayLikelihood([_psrl(psr, y_orig)],
                           commongp=_commongp(psr, T),
                           extsignals=[extsig])

    with pytest.raises(NotImplementedError, match="extsignals"):
        m.logL(HYPERPARAMS)

    # without extsignals, logL works normally (regression guard)
    m_plain = ds.ArrayLikelihood([_psrl(psr, y_orig)],
                                 commongp=_commongp(psr, T))
    assert np.isfinite(float(m_plain.logL(HYPERPARAMS)))
