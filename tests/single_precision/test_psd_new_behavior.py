"""New-behavior tests for the PSD factories (single_precision/
new behaviour for the log-space PSD path).

The characterization suite (test_psd_characterization.py) pins the *default*
factory numerics against the linear oracle. This suite locks the *new* surface
the migration adds:

  1. ``scale``       -> Phi multiplied by scale**2.
  2. fixed ``gamma`` -> drops ``gamma`` from the signature, matches the sampled
     form evaluated at that gamma, and yields no ``*_gamma`` sampled param.
  3. clip activation -> pathological inputs stay finite and saturate at the
     [low_clip, high_clip] bounds.
  4. no-leak guard   -> ``scale``/``low_clip``/``high_clip`` (and fixed gamma)
     never appear in a built model's sampled-parameter set.
"""
import inspect
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import pytest

import discovery as ds
from discovery import signals

from .test_psd_characterization import _basis, FB_ABOVE_BAND, NRHO

F, DF = _basis()
JF, JDF = jnp.asarray(F), jnp.asarray(DF)

RTOL = 1e-10

# A clip-inert operating point (matches the characterization grid).
LOG10_A, GAMMA, LOG10_FB = -13.0, 3.0, FB_ABOVE_BAND
# Flat (gamma=0), low-amplitude operating point for the scale tests: a flat
# spectrum has no frequency dynamic range, so even scale=10 (x100 in Phi) and
# scale=0.5 (/4) both stay strictly inside the clip window. The
# Phi = scale**2 * Phi relation only holds where neither side saturates.
LOG10_A_SCALE, GAMMA_SCALE = -14.5, 0.0
LOG10_RHO = np.full(NRHO, -6.0)
JRHO = jnp.asarray(LOG10_RHO)


def _close(a, b):
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=RTOL)


# --- 1. scale multiplies Phi by scale**2 -------------------------------------
@pytest.mark.parametrize("s", [0.5, 2.0, 10.0])
def test_scale_powerlaw(s):
    _close(signals.make_powerlaw(scale=s)(JF, JDF, LOG10_A_SCALE, GAMMA_SCALE),
           s ** 2 * signals.make_powerlaw()(JF, JDF, LOG10_A_SCALE, GAMMA_SCALE))


@pytest.mark.parametrize("s", [0.5, 2.0, 10.0])
def test_scale_brokenpowerlaw(s):
    _close(signals.make_brokenpowerlaw(scale=s)(JF, JDF, LOG10_A_SCALE, GAMMA_SCALE, LOG10_FB),
           s ** 2 * signals.make_brokenpowerlaw()(JF, JDF, LOG10_A_SCALE, GAMMA_SCALE, LOG10_FB))


@pytest.mark.parametrize("s", [0.5, 2.0, 10.0])
def test_scale_freespectrum(s):
    _close(signals.make_freespectrum(scale=s)(JF, JDF, JRHO),
           s ** 2 * signals.make_freespectrum()(JF, JDF, JRHO))


# --- 2. fixed gamma: signature, numerics --------------------------------------
def test_fixed_gamma_powerlaw_signature():
    fn = signals.make_powerlaw(gamma=GAMMA)
    assert list(inspect.signature(fn).parameters) == ["f", "df", "log10_A"]


def test_fixed_gamma_brokenpowerlaw_signature():
    fn = signals.make_brokenpowerlaw(gamma=GAMMA)
    assert list(inspect.signature(fn).parameters) == ["f", "df", "log10_A", "log10_fb"]


@pytest.mark.parametrize("g", [2.0, 3.0, 4.5])
def test_fixed_gamma_powerlaw_matches_sampled(g):
    _close(signals.make_powerlaw(gamma=g)(JF, JDF, LOG10_A),
           signals.make_powerlaw()(JF, JDF, LOG10_A, g))


@pytest.mark.parametrize("g", [2.0, 3.0, 4.5])
def test_fixed_gamma_brokenpowerlaw_matches_sampled(g):
    _close(signals.make_brokenpowerlaw(gamma=g)(JF, JDF, LOG10_A, LOG10_FB),
           signals.make_brokenpowerlaw()(JF, JDF, LOG10_A, g, LOG10_FB))


# --- 3. clip activation at pathological inputs --------------------------------
@pytest.mark.parametrize("log10_A,expect_log", [(50.0, -9.0), (-200.0, -18.0)])
def test_clip_powerlaw(log10_A, expect_log):
    out = np.asarray(signals.make_powerlaw()(JF, JDF, log10_A, GAMMA))
    assert np.all(np.isfinite(out))
    _close(out, np.full_like(out, 10.0 ** expect_log))


def test_clip_freespectrum_ceiling():
    out = np.asarray(signals.make_freespectrum()(JF, JDF, jnp.full(NRHO, 50.0)))
    assert np.all(np.isfinite(out))
    _close(out, np.full_like(out, 10.0 ** -9.0))


def test_clip_brokenpowerlaw_finite_no_overflow():
    # kappa=0.1 -> raw (f/fb)^10 would overflow; logaddexp form must stay finite.
    out = np.asarray(signals.make_brokenpowerlaw()(JF, JDF, LOG10_A, 6.0, -9.0))
    assert np.all(np.isfinite(out))
    assert out.max() <= 10.0 ** -9.0 + 1e-30


def test_custom_clip_window_respected():
    out = np.asarray(signals.make_powerlaw(high_clip=-12.0)(JF, JDF, 50.0, GAMMA))
    _close(out, np.full_like(out, 10.0 ** -12.0))


# --- 4. no-leak guard: closure config never reaches model params --------------
DATA = Path(__file__).resolve().parents[2] / "data"
B1855 = DATA / "v1p1_de440_pint_bipm2019-B1855+09.feather"


@pytest.fixture(scope="module")
def psr():
    return ds.Pulsar.read_feather(B1855)


def _rn_model(psr, psd):
    return ds.PulsarLikelihood([
        psr.residuals,
        ds.makenoise_measurement(psr, psr.noisedict),
        ds.makegp_fourier(psr, psd, components=30, name="rednoise"),
    ])


def test_scale_clips_not_sampled(psr):
    params = _rn_model(psr, signals.make_powerlaw(scale=2.0)).logL.params
    for leaked in ("scale", "low_clip", "high_clip"):
        assert not any(leaked in p for p in params), f"{leaked} leaked into {params}"
    assert any("rednoise_log10_A" in p for p in params)
    assert any("rednoise_gamma" in p for p in params)


def test_fixed_gamma_drops_gamma_param(psr):
    params = _rn_model(psr, signals.make_powerlaw(gamma=4.33)).logL.params
    assert any("rednoise_log10_A" in p for p in params)
    assert not any("rednoise_gamma" in p for p in params), params
