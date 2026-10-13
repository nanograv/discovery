"""Band noise: the normalised band envelope, and the Fourier and FFT-covariance band bases
parametrised by band centre and log10 bandwidth."""

import numpy as np
import jax.numpy as jnp
import pytest
from pathlib import Path

try:
    import discovery as ds
    from discovery import signals
    HAVE_DISCOVERY = True
except ImportError:
    HAVE_DISCOVERY = False


@pytest.fixture
def data_dir():
    return Path(__file__).parent / "data"


@pytest.fixture
def psr(data_dir):
    if not HAVE_DISCOVERY:
        pytest.skip("discovery package not installed")
    return ds.Pulsar.read_feather(data_dir / "multi_backend_pulsar.feather")


@pytest.fixture
def fcenter(psr):
    return 0.5 * (float(psr.freqs.min()) + float(psr.freqs.max()))


@pytest.mark.unit
@pytest.mark.parametrize("log10_bw", [1.3, 1.7, 2.0, 2.5, 2.9])
def test_band_envelope_unit_rms(psr, fcenter, log10_bw):
    """The envelope has unit rms at any bandwidth."""
    env = signals._band_envelope(psr, fcenter, log10_bw)
    rms = float(jnp.sqrt(jnp.mean(env ** 2)))
    assert np.isclose(rms, 1.0, atol=1e-3)


@pytest.mark.unit
@pytest.mark.parametrize("log10_bw", [1.0, 2.0, 3.0])
def test_band_envelope_finite_and_positive(psr, fcenter, log10_bw):
    """Envelope is finite and non-negative everywhere, and never identically zero."""
    env = np.asarray(signals._band_envelope(psr, fcenter, log10_bw))
    assert np.all(np.isfinite(env))
    assert np.all(env >= 0.0)
    assert env.sum() > 0.0


@pytest.mark.unit
def test_band_envelope_centred(psr, fcenter):
    """In-band TOAs carry more weight than out-of-band ones."""
    freqs = np.asarray(psr.freqs)
    env = np.asarray(signals._band_envelope(psr, fcenter, np.log10(100.0)))  # 100 MHz band
    inband = np.abs(freqs - fcenter) < 50.0
    if inband.any() and (~inband).any():
        assert env[inband].mean() > env[~inband].mean()


@pytest.mark.unit
def test_fourierbasis_band(psr, fcenter):
    """fourierbasis_band returns a callable basis with the expected shape."""
    f, df, fmatfunc = signals.fourierbasis_band(psr, 5)
    assert callable(fmatfunc)
    F = np.asarray(fmatfunc(fcenter, 2.0))
    assert F.shape == (len(psr.toas), 10)
    assert np.all(np.isfinite(F))


@pytest.mark.unit
def test_makegp_fftcov_band(psr):
    """makegp_fftcov_band builds a band GP parametrised by (fcenter, log10_bw)."""
    gp = signals.makegp_fftcov_band(psr, signals.powerlaw, components=5)
    assert gp.gpname == "band_gp"
    assert callable(gp.F)
    assert gp.F.params == [f"{psr.name}_band_gp_fcenter", f"{psr.name}_band_gp_log10_bw"]
    assert sorted(gp.Phi.params) == sorted(
        [f"{psr.name}_band_gp_log10_A", f"{psr.name}_band_gp_gamma"]
    )


@pytest.mark.unit
def test_band_uses_centre_width_not_edges(psr):
    """The band is sampled as a centre and a width, not as two edges."""
    gp = signals.makegp_fftcov_band(psr, signals.powerlaw, components=5)
    joined = " ".join(gp.F.params)
    assert "fcenter" in joined and "log10_bw" in joined
    assert "flow" not in joined and "fhigh" not in joined


@pytest.mark.unit
def test_make_timeinterpbasis_band(psr):
    """make_timeinterpbasis_band builds a callable band basis over (fcenter, log10_bw)."""
    basis = signals.make_timeinterpbasis_band()
    t_coarse, dt_coarse, Bmat_func = basis(psr, 5, signals.getspan(psr))
    assert callable(Bmat_func)
    B = np.asarray(Bmat_func(1400.0, 2.0))
    assert B.shape[0] == len(psr.toas)
    assert np.all(np.isfinite(B))


@pytest.mark.unit
def test_band_envelope_is_the_normalised_product_of_two_sigmoids(psr):
    nu = np.asarray(psr.freqs, dtype=np.float64)
    fc, lbw = 1250.0, 2.2
    lo, hi, s = fc - 0.5 * 10 ** lbw, fc + 0.5 * 10 ** lbw, 0.1 * 10 ** lbw
    e = 1.0 / (1.0 + np.exp((lo - nu) / s)) / (1.0 + np.exp((nu - hi) / s))
    want = e / np.sqrt(np.mean(e ** 2) + 1e-12)
    assert np.allclose(np.asarray(signals._band_envelope(psr, fc, lbw)), want, rtol=1e-12, atol=0)


@pytest.mark.unit
def test_band_gp_is_the_fftcov_gp_under_the_envelope(psr, fcenter):
    """Same prior as makegp_fftcov, and a basis whose rows are scaled by the envelope."""
    band = signals.makegp_fftcov_band(psr, signals.powerlaw, components=11)
    plain = signals.makegp_fftcov(psr, signals.powerlaw, components=11, name='band_gp')
    p = {f'{psr.name}_band_gp_log10_A': -14.0, f'{psr.name}_band_gp_gamma': 3.0}
    assert np.array_equal(np.asarray(band.Phi.getN(p)), np.asarray(plain.Phi.getN(p)))
    env = np.asarray(signals._band_envelope(psr, fcenter, 2.0))
    F = np.asarray(band.F({f'{psr.name}_band_gp_fcenter': fcenter, f'{psr.name}_band_gp_log10_bw': 2.0}))
    assert np.allclose(F, np.asarray(plain.F) * env[:, None], rtol=1e-14, atol=0)


@pytest.mark.unit
def test_band_model_is_finite_under_data_bounded_priors(psr):
    import jax
    from discovery import prior
    band = signals.makegp_fftcov_band(psr, signals.powerlaw, components=11)
    m = ds.PulsarLikelihood([psr.residuals, signals.makenoise_measurement(psr),
                             signals.makegp_timing(psr, svd=True), band])
    fmin, fmax = float(psr.freqs.min()), float(psr.freqs.max())
    priordict = {f'{psr.name}_band_gp_fcenter': [fmin, fmax],
                 f'{psr.name}_band_gp_log10_bw': [np.log10(20.0), np.log10(fmax - fmin)]}
    assert prior.getprior_uniform(f'{psr.name}_band_gp_log10_A') == [-18, -11]
    assert prior.getprior_uniform(f'{psr.name}_band_gp_gamma') == [0, 7]
    t = prior.makelogtransform_uniform(m.logL, priordict=priordict)
    y = np.zeros(len(t.params))
    assert np.isfinite(float(t(y))) and np.all(np.isfinite(np.asarray(jax.grad(t)(y))))
