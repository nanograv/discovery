"""Tests for the DM/chromatic Fourier and FFT-covariance signal helpers.

Covers the renamed Fourier bases (``fourierbasis_dm`` / ``fourierbasis_chrom``
and their ``make_*`` factories), the new FFT-covariance GPs
(``makegp_fftcov_dm`` / ``makegp_fftcov_chrom``) and their time-interpolation
bases, the back-compatibility of the deprecated ``dmfourierbasis*`` aliases, and
the extended ``makenoise_measurement_simple`` white-noise model.
"""

import numpy as np
import pytest
from pathlib import Path

try:
    import discovery as ds
    from discovery import signals, matrix, prior
    HAVE_DISCOVERY = True
except ImportError:
    HAVE_DISCOVERY = False


@pytest.fixture
def data_dir():
    """Path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def psr(data_dir):
    """A real multi-backend test pulsar (has .freqs needed for chromatic bases)."""
    if not HAVE_DISCOVERY:
        pytest.skip("discovery package not installed")
    return ds.Pulsar.read_feather(data_dir / "multi_backend_pulsar.feather")


# ---------------------------------------------------------------------------
# Fourier basis correctness
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_fourierbasis_dm_scaling(psr):
    """fourierbasis_dm scales the achromatic basis by (fref/freqs)**2."""
    fref = 1400.0
    _, _, F0 = signals.fourierbasis(psr, 5)
    f, df, Fdm = signals.fourierbasis_dm(psr, 5, fref=fref)

    expected = F0 * ((fref / psr.freqs) ** 2)[:, None]
    assert Fdm.shape == F0.shape
    assert np.allclose(np.asarray(Fdm), expected)


@pytest.mark.unit
@pytest.mark.parametrize("alpha", [1.5, 2.0, 4.0])
def test_fourierbasis_chrom_scaling(psr, alpha):
    """fourierbasis_chrom returns a closure scaling by (fref/freqs)**alpha."""
    fref = 1400.0
    _, _, F0 = signals.fourierbasis(psr, 5)
    f, df, fmatfunc = signals.fourierbasis_chrom(psr, 5, fref=fref)

    assert callable(fmatfunc)
    expected = np.asarray(F0) * ((fref / psr.freqs) ** alpha)[:, None]
    assert np.allclose(np.asarray(fmatfunc(alpha)), expected)


@pytest.mark.unit
def test_make_fourierbasis_dm_matches_direct(psr):
    """make_fourierbasis_dm(alpha=2) reproduces fourierbasis_dm."""
    basis = signals.make_fourierbasis_dm(alpha=2.0, tndm=False)
    _, _, Fmade = basis(psr, 5)
    _, _, Fdirect = signals.fourierbasis_dm(psr, 5)
    assert np.allclose(np.asarray(Fmade), np.asarray(Fdirect))


@pytest.mark.unit
def test_make_fourierbasis_chrom_is_dm_with_alpha(psr):
    """make_fourierbasis_chrom is make_fourierbasis_dm with a chromatic index."""
    fmade = signals.make_fourierbasis_chrom(alpha=4.0, tndm=False)(psr, 5)[2]
    fref = 1400.0
    _, _, F0 = signals.fourierbasis(psr, 5)
    expected = F0 * ((fref / psr.freqs) ** 4.0)[:, None]
    assert np.allclose(np.asarray(fmade), expected)


# ---------------------------------------------------------------------------
# Deprecated aliases: must still work AND warn via print
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_dmfourierbasis_deprecated_alias(psr):
    """dmfourierbasis raises a DeprecationWarning and matches fourierbasis_dm."""
    with pytest.warns(DeprecationWarning, match="fourierbasis_dm"):
        _, _, F_old = signals.dmfourierbasis(psr, 5)

    _, _, F_new = signals.fourierbasis_dm(psr, 5)
    assert np.allclose(np.asarray(F_old), np.asarray(F_new))


@pytest.mark.unit
def test_dmfourierbasis_alpha_deprecated_alias(psr):
    """dmfourierbasis_alpha raises a DeprecationWarning and matches fourierbasis_chrom."""
    with pytest.warns(DeprecationWarning, match="fourierbasis_chrom"):
        _, _, func_old = signals.dmfourierbasis_alpha(psr, 5)

    _, _, func_new = signals.fourierbasis_chrom(psr, 5)
    assert np.allclose(np.asarray(func_old(3.0)), np.asarray(func_new(3.0)))


@pytest.mark.unit
def test_make_dmfourierbasis_deprecated_alias(psr):
    """make_dmfourierbasis raises a DeprecationWarning and matches make_fourierbasis_dm."""
    with pytest.warns(DeprecationWarning, match="make_fourierbasis_dm"):
        basis_old = signals.make_dmfourierbasis(alpha=2.0)

    F_old = basis_old(psr, 5)[2]
    F_new = signals.make_fourierbasis_dm(alpha=2.0)(psr, 5)[2]
    assert np.allclose(np.asarray(F_old), np.asarray(F_new))


# ---------------------------------------------------------------------------
# Time-interpolation bases used by the FFT-covariance GPs
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_make_timeinterpbasis_dm_scaling(psr):
    """make_timeinterpbasis_dm scales the achromatic basis by (fref/freqs)**2."""
    fref = 1400.0
    T = signals.getspan(psr)
    _, _, B0 = signals.make_timeinterpbasis()(psr, 5, T)
    _, _, Bdm = signals.make_timeinterpbasis_dm(fref=fref)(psr, 5, T)
    expected = ((fref / psr.freqs) ** 2)[:, None] * np.asarray(B0)
    assert np.allclose(np.asarray(Bdm), expected)


@pytest.mark.unit
def test_make_timeinterpbasis_chromatic_scaling(psr):
    """make_timeinterpbasis_chromatic returns a closure scaling by (fref/freqs)**alpha."""
    fref, alpha = 1400.0, 3.0
    T = signals.getspan(psr)
    _, _, B0 = signals.make_timeinterpbasis()(psr, 5, T)
    _, _, Bfunc = signals.make_timeinterpbasis_chromatic(fref=fref)(psr, 5, T)
    assert callable(Bfunc)
    expected = ((fref / psr.freqs) ** alpha)[:, None] * np.asarray(B0)
    assert np.allclose(np.asarray(Bfunc(alpha)), expected)


@pytest.mark.unit
def test_make_dmtimeinterpbasis_deprecated(psr):
    """make_dmtimeinterpbasis raises a DeprecationWarning but still works."""
    with pytest.warns(DeprecationWarning, match="make_timeinterpbasis_dm"):
        basis = signals.make_dmtimeinterpbasis(alpha=2.0)
    # still produces a usable basis
    T = signals.getspan(psr)
    _, _, B = basis(psr, 5, T)
    assert np.asarray(B).shape[0] == len(psr.toas)


# ---------------------------------------------------------------------------
# FFT-covariance GPs
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_makegp_fftcov_dm(psr):
    """makegp_fftcov_dm builds a DM GP with the expected name and parameters."""
    gp = signals.makegp_fftcov_dm(psr, signals.powerlaw, components=5)
    assert gp.gpname == "dm_gp"
    assert np.asarray(gp.F).shape[0] == len(psr.toas)
    # fixed chromatic index -> static design matrix, no alpha parameter
    assert not callable(gp.F)
    assert sorted(gp.Phi.params) == sorted(
        [f"{psr.name}_dm_gp_log10_A", f"{psr.name}_dm_gp_gamma"]
    )
    # default name resolves against the standard DM-GP priors
    for par in gp.Phi.params:
        assert prior.getprior_uniform(par) is not None


@pytest.mark.unit
def test_makegp_fftcov_chrom(psr):
    """makegp_fftcov_chrom builds a chromatic GP with a free alpha parameter."""
    gp = signals.makegp_fftcov_chrom(psr, signals.powerlaw, components=5)
    assert gp.gpname == "chrom_gp"
    # variable chromatic index -> callable design matrix exposing alpha
    assert callable(gp.F)
    assert gp.F.params == [f"{psr.name}_chrom_gp_alpha"]
    # default name resolves against the standard chromatic-GP alpha prior
    assert prior.getprior_uniform(f"{psr.name}_chrom_gp_alpha") == [2.5, 14]


# ---------------------------------------------------------------------------
# Default priors for the new parameters
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("par,expected", [
    ("J0000_dm_gp_log10_A", [-20, -11]),
    ("J0000_dm_gp_gamma", [0, 7]),
    ("J0000_dm_gp_alpha", [1, 3]),
    ("J0000_chrom_gp_log10_A", [-20, -11]),
    ("J0000_chrom_gp_gamma", [0, 7]),
    ("J0000_chrom_gp_alpha", [2.5, 14]),
])
def test_chromatic_default_priors(par, expected):
    """The new DM/chromatic GP parameters have the expected default priors."""
    if not HAVE_DISCOVERY:
        pytest.skip("discovery package not installed")
    assert prior.getprior_uniform(par) == expected


# ---------------------------------------------------------------------------
# Extended single-EFAC white-noise model
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_measurement_simple_default_params(psr):
    """Default model uses one efac and a t2equad."""
    noise = signals.makenoise_measurement_simple(psr, noisedict={})
    assert isinstance(noise, matrix.NoiseMatrix1D_var)
    assert noise.params == [f"{psr.name}_efac", f"{psr.name}_log10_t2equad"]


@pytest.mark.unit
def test_measurement_simple_tnequad(psr):
    """tnequad=True switches the EQUAD parameter name/convention."""
    noise = signals.makenoise_measurement_simple(psr, noisedict={}, tnequad=True)
    assert noise.params == [f"{psr.name}_efac", f"{psr.name}_log10_tnequad"]


@pytest.mark.unit
def test_measurement_simple_no_equad(psr):
    """add_equad=False yields an EFAC-only model."""
    noise = signals.makenoise_measurement_simple(psr, noisedict={}, add_equad=False)
    assert noise.params == [f"{psr.name}_efac"]


@pytest.mark.unit
def test_measurement_simple_fixed_is_constant(psr):
    """A fully specified noisedict returns a constant (novar) matrix with correct values."""
    efac, log10_t2equad = 1.3, -6.0
    noisedict = {
        f"{psr.name}_efac": efac,
        f"{psr.name}_log10_t2equad": log10_t2equad,
    }
    noise = signals.makenoise_measurement_simple(psr, noisedict=noisedict)
    assert isinstance(noise, matrix.NoiseMatrix1D_novar)
    expected = efac**2 * (psr.toaerrs**2 + 10.0 ** (2.0 * log10_t2equad))
    assert np.allclose(np.asarray(noise.N), expected)
