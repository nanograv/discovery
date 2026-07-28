"""Characterization (golden) tests for the PSD/prior functions slated for the
single-precision log-space + scale/clip migration.

Each production PSD is pinned against an **independent linear-space oracle** — a
direct transcription of today's formula — over a parameter grid chosen so every
Phi stays strictly inside the future clip window (1e-18, 1e-9) s^2, i.e. the
clip is inert. These tests therefore:

  * pass on the current (linear) implementation now, and
  * must still pass after the bodies are rewritten in log-space with a
    clip-in-log and a `scale` closure constant (the rewrite must not move the
    value for clip-inert, scale=1 inputs).

Because the log-space rewrite reorders float64 arithmetic, the comparison is
`rtol=1e-10`, not bit-exact. A meta-check asserts the grid really is clip-inert,
so the window assumption can't silently rot.
"""
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import pytest

import discovery as ds
from discovery import const

# future clip window (s^2); the grid must stay inside it so these tests survive
# the clip-in-log rewrite untouched.
LOW, HIGH = 1e-18, 1e-9
RTOL = 1e-10

KAPPA = 0.1  # broken-powerlaw transition smoothness (matches signals.py)


# --- a realistic Fourier basis (f, df), matching signals.fourierbasis ---------
def _basis(components=30, T_years=15.0):
    T = T_years * 365.25 * 86400.0
    f0 = np.arange(1, components + 1, dtype=np.float64) / T
    df0 = np.diff(np.concatenate(([0.0], f0)))
    return np.repeat(f0, 2), np.repeat(df0, 2)


F, DF = _basis()                  # numpy, for the oracle
JF, JDF = jnp.asarray(F), jnp.asarray(DF)   # jax, for production (jax `.at[]` paths)
NF = F.shape[0]          # 2 * components
NRHO = NF // 2           # one log10_rho per frequency


# --- independent linear-space oracles (transcriptions of current bodies) ------
def ref_powerlaw(f, df, log10_A, gamma):
    return (10.0 ** (2.0 * log10_A)) / 12.0 / np.pi ** 2 * \
        const.fyr ** (gamma - 3.0) * f ** (-gamma) * df


def ref_broken_factor(f, log10_fb, gamma):
    return (1.0 + (f / 10.0 ** log10_fb) ** (1.0 / KAPPA)) ** (KAPPA * gamma)


def ref_brokenpowerlaw(f, df, log10_A, gamma, log10_fb):
    return ref_powerlaw(f, df, log10_A, gamma) * ref_broken_factor(f, log10_fb, gamma)


def ref_freespectrum(log10_rho):
    return np.repeat(10.0 ** (2.0 * np.asarray(log10_rho, dtype=np.float64)), 2)


# --- helpers ------------------------------------------------------------------
def _assert_inert(phi):
    """Meta-check: the oracle output is strictly inside the clip window, so the
    future clip cannot change it (keeps these golden tests valid post-rewrite)."""
    phi = np.asarray(phi)
    assert phi.min() > LOW, f"grid hits floor: min Phi={phi.min():.3e} <= {LOW:.0e}"
    assert phi.max() < HIGH, f"grid hits ceiling: max Phi={phi.max():.3e} >= {HIGH:.0e}"


def _match(prod, oracle):
    _assert_inert(oracle)
    np.testing.assert_allclose(np.asarray(prod), oracle, rtol=RTOL, atol=0.0)


# --- parameter grids (clip-inert) --------------------------------------------
PL_GRID = [(la, g) for la in (-13.5, -13.0) for g in (2.0, 3.0, 4.0)]
FB_ABOVE_BAND = -7.0   # > max basis frequency, so the broken factor ~ 1


# === tests ====================================================================
@pytest.mark.parametrize("log10_A,gamma", PL_GRID)
def test_powerlaw(log10_A, gamma):
    _match(ds.powerlaw(JF, JDF, log10_A, gamma),
           ref_powerlaw(F, DF, log10_A, gamma))


@pytest.mark.parametrize("log10_A,gamma", PL_GRID)
def test_brokenpowerlaw(log10_A, gamma):
    _match(ds.brokenpowerlaw(JF, JDF, log10_A, gamma, FB_ABOVE_BAND),
           ref_brokenpowerlaw(F, DF, log10_A, gamma, FB_ABOVE_BAND))


@pytest.mark.parametrize("log10_rho", [-7.0, -6.0, -5.5])
def test_freespectrum(log10_rho):
    rho = np.full(NRHO, log10_rho)
    _match(ds.freespectrum(JF, JDF, rho), ref_freespectrum(rho))


@pytest.mark.parametrize("log10_A,gamma", PL_GRID)
def test_powerlaw_brokencrn(log10_A, gamma):
    crn_log10_A, crn_gamma, crn_log10_fb = -13.5, 3.0, FB_ABOVE_BAND
    oracle = (ref_powerlaw(F, DF, log10_A, gamma)
              + ref_powerlaw(F, DF, crn_log10_A, crn_gamma)
              * ref_broken_factor(F, crn_log10_fb, crn_gamma))
    _match(ds.powerlaw_brokencrn(JF, JDF, log10_A, gamma,
                                 crn_log10_A, crn_gamma, crn_log10_fb), oracle)


@pytest.mark.parametrize("log10_A,gamma", PL_GRID)
def test_brokenpowerlaw_brokencrn(log10_A, gamma):
    log10_fb, crn_log10_A, crn_gamma, crn_log10_fb = FB_ABOVE_BAND, -13.5, 3.0, FB_ABOVE_BAND
    oracle = (ref_brokenpowerlaw(F, DF, log10_A, gamma, log10_fb)
              + ref_powerlaw(F, DF, crn_log10_A, crn_gamma)
              * ref_broken_factor(F, crn_log10_fb, crn_gamma))
    _match(ds.brokenpowerlaw_brokencrn(JF, JDF, log10_A, gamma, log10_fb,
                                       crn_log10_A, crn_gamma, crn_log10_fb), oracle)


def test_makepowerlaw_crn():
    ccomps = 10
    log10_A, gamma, crn_log10_A, crn_gamma = -13.0, 3.0, -13.5, 4.0
    fn = ds.makepowerlaw_crn(ccomps)            # crn_gamma='variable' (sampled)
    oracle = ref_powerlaw(F, DF, log10_A, gamma).copy()
    oracle[:2 * ccomps] += ref_powerlaw(F[:2 * ccomps], DF[:2 * ccomps],
                                        crn_log10_A, crn_gamma)
    _match(fn(JF, JDF, log10_A, gamma, crn_log10_A, crn_gamma), oracle)


def test_makefreespectrum_crn():
    ccomps = 10
    rho = np.full(NRHO, -6.0)
    crn_rho = np.full(ccomps, -6.0)
    fn = ds.makefreespectrum_crn(ccomps)
    oracle = ref_freespectrum(rho).copy()
    oracle[:2 * ccomps] += ref_freespectrum(crn_rho)
    _match(fn(JF, JDF, rho, crn_rho), oracle)


def test_make_combined_crn():
    ccomps = 10
    log10_A, gamma, crn_log10_A, crn_gamma = -13.0, 3.0, -13.5, 4.0
    combined, crn_params = ds.make_combined_crn(ccomps, ds.powerlaw, ds.powerlaw)
    assert crn_params == ["crn_log10_A", "crn_gamma"]
    oracle = ref_powerlaw(F, DF, log10_A, gamma).copy()
    oracle[:2 * ccomps] += ref_powerlaw(F[:2 * ccomps], DF[:2 * ccomps],
                                        crn_log10_A, crn_gamma)
    _match(combined(JF, JDF, log10_A, gamma, crn_log10_A, crn_gamma), oracle)
